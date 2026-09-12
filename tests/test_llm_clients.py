import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_clients import FakeLLMClient, RawGeneration


def test_fake_client_returns_scripted_text_in_order():
    client = FakeLLMClient(["erste Antwort", "zweite Antwort"])
    first = client.generate("prompt 1")
    second = client.generate("prompt 2")
    assert first.status == "success"
    assert first.raw_text == "erste Antwort"
    assert second.raw_text == "zweite Antwort"


def test_fake_client_can_script_an_error():
    error = RawGeneration(status="error", error_type="timeout", raw_text=None, latency_ms=5.0)
    client = FakeLLMClient([error])
    result = client.generate("prompt")
    assert result.status == "error"
    assert result.error_type == "timeout"


def test_fake_client_raises_when_scenario_exhausted():
    client = FakeLLMClient(["only one"])
    client.generate("p")
    try:
        client.generate("p again")
        assert False, "phải raise khi hết kịch bản"
    except AssertionError as e:
        assert "hết kịch bản" in str(e)
