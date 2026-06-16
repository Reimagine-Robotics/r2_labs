"""Futures and arm-aware executor for concurrent robot behaviour execution."""

import dataclasses
import enum
import threading
import weakref
from collections import deque
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    FIRST_EXCEPTION,
    Executor,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from typing import Any, Callable, Deque, Generic, TypeVar

_T = TypeVar("_T")


@enum.unique
class ArmSelection(enum.Enum):
  """Which arm(s) a behaviour requires for execution."""

  LEFT = "left"
  RIGHT = "right"
  BOTH = "both"


# Backwards-compatible alias for older naming.
ArmSide = ArmSelection


@dataclasses.dataclass(slots=True)
class _QueuedTask(Generic[_T]):
  """Internal task representation for the scheduler queue.

  Attributes:
    requirement: Which arm(s) this task needs.
    fn: The callable to execute.
    args: Positional arguments for fn.
    kwargs: Keyword arguments for fn.
    future: Future to receive the result.
    cancel_callback: Optional callback invoked on cancellation.
  """

  requirement: ArmSelection
  fn: Callable[..., _T]
  args: tuple[Any, ...]
  kwargs: dict[str, Any]
  future: Future[_T]
  cancel_callback: Callable[[], None] | None


class CancellableFuture(Future[_T]):
  """Future with an optional cancellation hook.

  When cancelled, invokes the callback before the standard cancel logic.
  """

  def __init__(self, cancel_callback: Callable[[], None] | None = None):
    """Initialize the future.

    Args:
      cancel_callback: Optional callback invoked when cancel() is called.
    """
    super().__init__()
    self._cancel_callback = cancel_callback

  def cancel(self) -> bool:
    """Cancel the future and invoke the cancellation callback if set."""
    if self._cancel_callback is not None:
      try:
        self._cancel_callback()
      except Exception:  # pylint: disable=broad-except
        pass
    return super().cancel()


class ArmExecutor(Executor):
  """Arm-aware executor that can run left/right concurrently or lock both.

  Tasks for LEFT or RIGHT arms can run in parallel. Tasks requiring BOTH arms
  block until both are free and prevent other tasks from starting.
  """

  def __init__(self) -> None:
    """Initialize the executor with per-arm thread pools."""
    self._left_executor: ThreadPoolExecutor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="sdk-left"
    )
    self._right_executor: ThreadPoolExecutor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="sdk-right"
    )
    self._bimanual_executor: ThreadPoolExecutor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="sdk-bimanual"
    )
    self._tasks: Deque[_QueuedTask[Any]] = deque()
    self._future_arm: weakref.WeakKeyDictionary[Future[Any], ArmSelection] = (
        weakref.WeakKeyDictionary()
    )
    self._cv = threading.Condition()
    self._left_busy = False
    self._right_busy = False
    self._shutdown = False
    self._scheduler = threading.Thread(
        target=self._run_scheduler, name="sdk-arm-scheduler", daemon=True
    )
    self._scheduler.start()

  def submit(
      self, fn: Callable[..., _T], /, *args: Any, **kwargs: Any
  ) -> Future[_T]:
    """Submit a task for the left arm (default). See submit_for_arm."""
    return self.submit_for_arm(ArmSelection.LEFT, fn, *args, **kwargs)

  def submit_for_arm(
      self,
      arm: ArmSelection,
      fn: Callable[..., _T],
      /,
      *args: Any,
      cancel_callback: Callable[[], None] | None = None,
      **kwargs: Any,
  ) -> Future[_T]:
    """Submit a task requiring specific arm(s).

    Args:
      arm: Which arm(s) the task requires.
      fn: The callable to execute.
      *args: Positional arguments for fn.
      cancel_callback: Optional callback invoked if the future is cancelled.
      **kwargs: Keyword arguments for fn.

    Returns:
      A Future that will contain the result of fn(*args, **kwargs).

    Raises:
      RuntimeError: If the executor has been shut down.
    """
    if self._shutdown:
      raise RuntimeError("executor has been shut down")
    future: Future[_T] = CancellableFuture(cancel_callback=cancel_callback)
    task = _QueuedTask(
        requirement=arm,
        fn=fn,
        args=tuple(args),
        kwargs=dict(kwargs),
        future=future,
        cancel_callback=cancel_callback,
    )
    with self._cv:
      self._tasks.append(task)
      self._future_arm[future] = arm
      self._cv.notify()
    return future

  def arm_for(self, future: Future[Any]) -> ArmSelection | None:
    """Return which arm(s) a submitted future requires, or None if unknown."""
    return self._future_arm.get(future)

  def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
    """Shut down the executor.

    Args:
      wait: If True, block until all tasks complete.
      cancel_futures: If True, cancel pending tasks before shutdown.
    """
    with self._cv:
      self._shutdown = True
      if cancel_futures:
        for task in list(self._tasks):
          task.future.cancel()
        self._tasks.clear()
      self._cv.notify_all()
    if wait:
      self._scheduler.join()
    self._left_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
    self._right_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
    self._bimanual_executor.shutdown(wait=wait, cancel_futures=cancel_futures)

  def __enter__(self) -> "ArmExecutor":
    """Enter context manager."""
    return self

  def __exit__(self, exc_type, exc, tb) -> bool | None:
    """Exit context manager, shutting down the executor."""
    self.shutdown(wait=True, cancel_futures=False)
    return None

  def _run_scheduler(self) -> None:
    """Main scheduler loop that dispatches tasks when arms are available."""
    while True:
      with self._cv:
        while not self._shutdown and not self._tasks:
          self._cv.wait()
        if self._shutdown and not self._tasks:
          return
        task = self._pop_next_runnable()
        if task is None:
          self._cv.wait()
          continue
        self._mark_busy(task.requirement, busy=True)
      self._dispatch(task)

  def _pop_next_runnable(self) -> _QueuedTask[Any] | None:
    """Find and remove the next task that can run given arm availability."""
    # walk with manual indexing so deletions are safe while scanning
    idx = 0
    while idx < len(self._tasks):
      task = self._tasks[idx]
      if task.future.cancelled():
        del self._tasks[idx]
        continue
      if self._can_run(task.requirement):
        del self._tasks[idx]
        return task
      idx += 1
    return None

  def _dispatch(self, task: _QueuedTask[Any]) -> None:
    """Submit the task to the appropriate arm's thread pool."""
    executor = self._executor_for(task.requirement)
    executor.submit(self._run_task, task)

  def _run_task(self, task: _QueuedTask[Any]) -> None:
    """Execute the task and set the result or exception on its future."""
    try:
      if task.future.cancelled():
        return
      result = task.fn(*task.args, **task.kwargs)
      if not task.future.cancelled():
        task.future.set_result(result)
    except BaseException as exc:  # pylint: disable=broad-except
      if not task.future.cancelled():
        task.future.set_exception(exc)
      raise
    finally:
      self._release(task.requirement)

  def _executor_for(self, requirement: ArmSelection) -> ThreadPoolExecutor:
    """Return the thread pool for the given arm requirement."""
    if requirement is ArmSelection.LEFT:
      return self._left_executor
    if requirement is ArmSelection.RIGHT:
      return self._right_executor
    return self._bimanual_executor

  def _can_run(self, requirement: ArmSelection) -> bool:
    """Check if the required arm(s) are currently available."""
    if requirement is ArmSelection.LEFT:
      return not self._left_busy
    if requirement is ArmSelection.RIGHT:
      return not self._right_busy
    return not self._left_busy and not self._right_busy

  def _mark_busy(self, requirement: ArmSelection, *, busy: bool) -> None:
    """Set the busy state for the given arm(s)."""
    if requirement is ArmSelection.LEFT:
      self._left_busy = busy
    elif requirement is ArmSelection.RIGHT:
      self._right_busy = busy
    else:
      self._left_busy = busy
      self._right_busy = busy

  def _release(self, requirement: ArmSelection) -> None:
    """Mark the arm(s) as no longer busy and notify waiting tasks."""
    with self._cv:
      self._mark_busy(requirement, busy=False)
      self._cv.notify_all()


# Alias to preserve earlier naming while expanding capabilities.
SingleThreadExecutor = ArmExecutor

__all__ = [
    "ALL_COMPLETED",
    "ArmExecutor",
    "ArmSelection",
    "ArmSide",
    "FIRST_COMPLETED",
    "FIRST_EXCEPTION",
    "Future",
    "SingleThreadExecutor",
    "as_completed",
    "wait",
]
