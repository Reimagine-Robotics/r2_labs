"""The ping reply carries the server version, exposed on the client."""

import threading

import zmq

import r2_labs
from r2_labs.rpc import client as rpc_client
from r2_labs.rpc import server as rpc_server


def _serve(server: rpc_server.BaseServer) -> None:
  # The context teardown below force-closes the socket while run() is parked in
  # recv(); swallow the resulting ZMQError so it doesn't spam stderr.
  try:
    server.run()
  except zmq.ZMQError:
    pass


def test_client_reads_server_version_from_ping():
  context = zmq.Context()
  # Bind an ephemeral port and read it back from the server — no racy
  # pick-a-free-port-then-rebind gap.
  server = rpc_server.BaseServer(0, context)
  thread = threading.Thread(target=_serve, args=(server,), daemon=True)
  thread.start()
  try:
    # BaseClient pings on construction and stores the reply's version.
    client = rpc_client.BaseClient(
        f"tcp://localhost:{server.port}", timeout=2000
    )
    assert client.server_version == r2_labs.get_version()
  finally:
    # destroy(linger=0), not term(): the daemon server thread is parked in
    # recv() holding its socket, so term() would block forever.
    context.destroy(linger=0)
