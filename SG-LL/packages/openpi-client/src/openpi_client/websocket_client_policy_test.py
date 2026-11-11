from __future__ import annotations

import openpi_client.websocket_client_policy as websocket_client_policy


class _FakeConnection:
    def recv(self):
        return websocket_client_policy.msgpack_numpy.packb({"action_horizon": 50})


def _uninitialized_policy(connect_kwargs):
    policy = object.__new__(websocket_client_policy.WebsocketClientPolicy)
    policy._uri = "ws://127.0.0.1:9999"
    policy._api_key = None
    policy._websocket_connect_kwargs = dict(connect_kwargs)
    return policy


def test_wait_for_server_preserves_default_websocket_options(monkeypatch) -> None:
    calls = []

    def fake_connect(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeConnection()

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", fake_connect)
    connection, metadata = _uninitialized_policy({})._wait_for_server()

    assert isinstance(connection, _FakeConnection)
    assert metadata == {"action_horizon": 50}
    assert calls[0][1]["compression"] is None
    assert calls[0][1]["max_size"] is None
    assert "ping_interval" not in calls[0][1]
    assert "ping_timeout" not in calls[0][1]


def test_wait_for_server_forwards_explicit_transport_override(monkeypatch) -> None:
    calls = []

    def fake_connect(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeConnection()

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", fake_connect)
    _uninitialized_policy({"ping_interval": None, "ping_timeout": None})._wait_for_server()

    assert calls[0][1]["ping_interval"] is None
    assert calls[0][1]["ping_timeout"] is None
