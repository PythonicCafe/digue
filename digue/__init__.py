"""Local speech-to-text dictation and transcription using whisper.cpp."""

from __future__ import annotations

__version__ = "0.1.0"

DEFAULT_PORT = 8178
DEFAULT_LANGUAGE = "auto"
DEFAULT_MODELS = {"nvidia": "large-v3-turbo", "amd": "large-v3-turbo", "intel": "large-v3-turbo", "cpu": "small"}
AVAILABLE_MODELS = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")
DEFAULT_MAX_RECORD_SECONDS = 300

from digue.config import load_config  # noqa: E402
from digue.recording import record_to  # noqa: E402
from digue.transcribe import transcribe_file  # noqa: E402

__all__ = [
    "__version__",
    "load_config",
    "record_to",
    "transcribe_file",
]
