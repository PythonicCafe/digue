"""Local speech-to-text dictation and transcription using whisper.cpp."""

from __future__ import annotations

__version__ = "0.1.0"

DEFAULT_PORT = 8178
DEFAULT_LANGUAGE = "auto"
DEFAULT_MODELS = {"nvidia": "large-v3-turbo", "amd": "large-v3-turbo", "intel": "large-v3-turbo", "cpu": "small"}
# Every ggml model published in huggingface.co/ggerganov/whisper.cpp (the
# download source), in size order per family: f16, then q8_0, then q5_x.
# "-q8_0"/"-q5_0"/"-q5_1" are integer-quantized copies (smaller file and RAM,
# usually faster on CPU, slightly lower accuracy at q5); ".en" are English-only.
AVAILABLE_MODELS = (
    "tiny",
    "tiny-q8_0",
    "tiny-q5_1",
    "tiny.en",
    "tiny.en-q8_0",
    "tiny.en-q5_1",
    "base",
    "base-q8_0",
    "base-q5_1",
    "base.en",
    "base.en-q8_0",
    "base.en-q5_1",
    "small",
    "small-q8_0",
    "small-q5_1",
    "small.en",
    "small.en-q8_0",
    "small.en-q5_1",
    "medium",
    "medium-q8_0",
    "medium-q5_0",
    "medium.en",
    "medium.en-q8_0",
    "medium.en-q5_0",
    "large-v1",
    "large-v2",
    "large-v2-q8_0",
    "large-v2-q5_0",
    "large-v3",
    "large-v3-q5_0",
    "large-v3-turbo",
    "large-v3-turbo-q8_0",
    "large-v3-turbo-q5_0",
)
DEFAULT_MAX_RECORD_SECONDS = 300
# whisper-server answers only after transcribing the whole file, so this bounds
# the file length a CPU can handle; [transcribe] timeout overrides it.
DEFAULT_TRANSCRIPTION_TIMEOUT = 600

from digue.config import load_config  # noqa: E402
from digue.recording import record_to  # noqa: E402
from digue.transcribe import transcribe_file  # noqa: E402

__all__ = [
    "__version__",
    "load_config",
    "record_to",
    "transcribe_file",
]
