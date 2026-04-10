import json
from pathlib import Path

from tools import tts_tool


def test_check_tts_requirements_accepts_piper_with_named_model(monkeypatch):
    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"provider": "piper", "piper": {"binary_path": "piper", "model": "pl_PL-gosia-medium"}},
    )
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: False)
    monkeypatch.setattr(tts_tool, "_resolve_piper_binary", lambda cfg: "/usr/local/bin/piper")

    assert tts_tool.check_tts_requirements() is True


def test_check_tts_requirements_rejects_piper_without_binary(monkeypatch):
    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"provider": "piper", "piper": {"binary_path": "missing", "model": "pl_PL-gosia-medium"}},
    )
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr(tts_tool, "_check_neutts_available", lambda: False)
    monkeypatch.setattr(
        tts_tool,
        "_resolve_piper_binary",
        lambda cfg: (_ for _ in ()).throw(FileNotFoundError("Piper binary not found: missing")),
    )

    assert tts_tool.check_tts_requirements() is False


def test_generate_piper_uses_json_input_when_speaker_is_set(tmp_path, monkeypatch):
    wav_path = tmp_path / "sample.wav"
    captured = {}

    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None):
        captured["cmd"] = cmd
        captured["input"] = input
        wav_path.write_bytes(b"RIFFfake")

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        return Result()

    monkeypatch.setattr(tts_tool, "_resolve_piper_binary", lambda cfg: "/usr/local/bin/piper")
    monkeypatch.setattr(
        tts_tool,
        "_resolve_piper_model_paths",
        lambda cfg, allow_download=True: (tmp_path / "voice.onnx", tmp_path / "voice.onnx.json", "pl_PL-gosia-medium", tmp_path),
    )
    monkeypatch.setattr(tts_tool.subprocess, "run", fake_run)

    result = tts_tool._generate_piper(
        "hello",
        str(wav_path),
        {"piper": {"speaker": "2"}},
    )

    assert result == str(wav_path)
    assert "--json-input" in captured["cmd"]
    assert "\"speaker\": \"2\"" in captured["input"]


def test_text_to_speech_tool_uses_piper_and_converts_for_telegram(tmp_path, monkeypatch):
    wav_path = tmp_path / "tts.wav"
    ogg_path = tmp_path / "tts.ogg"

    def fake_generate(text, output_path, _tts_config):
        Path(output_path).write_bytes(b"RIFFfake")
        return output_path

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "piper", "piper": {"model": "pl_PL-gosia-medium"}})
    monkeypatch.setattr(tts_tool, "_generate_piper", fake_generate)
    monkeypatch.setattr(tts_tool, "_convert_to_opus", lambda _path: str(ogg_path))
    ogg_path.write_bytes(b"OggSfake")

    result = json.loads(tts_tool.text_to_speech_tool("Czesc", output_path=str(wav_path)))

    assert result["success"] is True
    assert result["provider"] == "piper"
    assert result["voice_compatible"] is True
    assert result["file_path"] == str(ogg_path)
    assert result["media_tag"].startswith("[[audio_as_voice]]")


def test_text_to_speech_tool_reports_missing_piper_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "piper", "piper": {"model": "pl_PL-gosia-medium"}})
    monkeypatch.setattr(
        tts_tool,
        "_generate_piper",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("Piper binary not found: piper")),
    )

    result = json.loads(tts_tool.text_to_speech_tool("Hello", output_path=str(tmp_path / "out.wav")))

    assert result["success"] is False
    assert "TTS dependency missing (piper)" in result["error"]
