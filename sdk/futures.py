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
  LEFT = "left"
  RIGHT = "right"
  BOTH = "both"


# Backwards-compatible alias for older naming.
ArmSide = ArmSelection


@dataclasses.dataclass(slots=True)
class _QueuedTask(Generic[_T]):
  requirement: ArmSelection
  fn: Callable[..., _T]
  args: tuple[Any, ...]
  kwargs: dict[str, Any]
  future: Future[_T]
  cancel_callback: Callable[[], None] | None


class CancellableFuture(Future[_T]):
  """Future with an optional cancellation hook."""

  def __init__(self, cancel_callback: Callable[[], None] | None = None):
    super().__init__()
    self._cancel_callback = cancel_callback

  def cancel(self) -> bool:
    if self._cancel_callback is not None:
      try:
        self._cancel_callback()
      except Exception:  # pylint: disable=broad-except
        pass
    return super().cancel()


class ArmExecutor(Executor):
  """Arm-aware executor that can run left/right concurrently or lock both."""

  def __init__(self) -> None:
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
    return self._future_arm.get(future)

  def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
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
    return self

  def __exit__(self, exc_type, exc, tb) -> bool | None:
    self.shutdown(wait=True, cancel_futures=False)
    return None

  def _run_scheduler(self) -> None:
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
    for idx, task in enumerate(self._tasks):
      if task.future.cancelled():
        del self._tasks[idx]
        continue
      if self._can_run(task.requirement):
        del self._tasks[idx]
        return task
    return None

  def _dispatch(self, task: _QueuedTask[Any]) -> None:
    executor = self._executor_for(task.requirement)
    executor.submit(self._run_task, task)

  def _run_task(self, task: _QueuedTask[Any]) -> None:
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
    if requirement is ArmSelection.LEFT:
      return self._left_executor
    if requirement is ArmSelection.RIGHT:
      return self._right_executor
    return self._bimanual_executor

  def _can_run(self, requirement: ArmSelection) -> bool:
    if requirement is ArmSelection.LEFT:
      return not self._left_busy
    if requirement is ArmSelection.RIGHT:
      return not self._right_busy
    return not self._left_busy and not self._right_busy

  def _mark_busy(self, requirement: ArmSelection, *, busy: bool) -> None:
    if requirement is ArmSelection.LEFT:
      self._left_busy = busy
    elif requirement is ArmSelection.RIGHT:
      self._right_busy = busy
    else:
      self._left_busy = busy
      self._right_busy = busy

  def _release(self, requirement: ArmSelection) -> None:
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
