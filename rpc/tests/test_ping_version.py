"""The ping reply carries the server version, exposed on the client."""

import threading

import zmq

import r2_labs
from r2_labs.rpc import client as rpc_client
from r2_labs.rpc import server as rpc_server


def test_client_reads_server_version_from_ping():
  context = zmq.Context()
  # Bind an ephemeral port and read it back from the server — no racy
  # pick-a-free-port-then-rebind gap.
  server = rpc_server.BaseServer(0, context)
  thread = threading.Thread(target=server.run, daemon=True)
  thread.start()
  try:
    # BaseClient pings on construction and stores the reply's version.
    client = rpc_client.BaseClient(
        f"tcp://localhost:{server.port}", timeout=2000
    )
    assert client.server_version == r2_labs.get_version()
  finally:
    # Stop the loop and join before destroying the context: closing sockets
    # under a parked recv aborts the process inside libzmq.
    server.stop()
    thread.join(timeout=5)
    context.destroy(linger=0)
