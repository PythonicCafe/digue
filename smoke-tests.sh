#!/usr/bin/env bash
# digue smoke tests (pre-release). Complements TESTES-MANUAIS.md: the human checklist covers GUI/mic/recovery behavior
# a script cannot automate; this runs everything a script can -- CLI commands, config validation, transcription formats,
# conversion, batch, server lifecycle, remote-backend refusals -- against real Docker containers on cpu/amd backends.
#
# Usage:
#   ./smoke-tests.sh [backend] [model ...]     # backend: cpu or amd (default: digue detect, then cpu)
#   ./smoke-tests.sh cpu small-q8_0 large-v3-turbo-q8_0
#
# Env toggles:
#   DIGUE_BIN=/path/to/digue   binary to test (default: digue in PATH)
#   FAIL_FAST=1                stop at the first failure
#   KEEP_ON_FAIL=1             keep the temp dir when the run fails (debugging)
#   DIGUE_SMOKE_CACHE=dir      model cache (default: ~/.cache/digue-smoke-models; empty disables)
#
# Dictation tests need a microphone and are interactive: the script says what to speak (language, duration), waits for
# ENTER, and only then starts recording. They run first among the functional tests and cover both pw-record and arecord
# (one short sentence each). Everything else is hands-off.
#
# Everything runs in a mktemp dir (config, data, runtime, audio). The whisper-server container uses a dedicated name
# (digue-smoke) and is removed at the end, even on Ctrl+c. Models are downloaded to the temp data dir (~250-834 MB each)
# but cached for reuse. Requires: docker without sudo, ffmpeg, curl, network.

set -o pipefail

# NOTE: deliberately no `set -e`: failing commands are the test data here (every check inspects $?), and an early abort
# would skip cleanup. Each dg call captures rc explicitly.

SCRIPT_BACKEND="${1:-}"
if [ -n "$2" ]; then
    shift
    MODELS=("$@")
else
    MODELS=(small-q8_0 large-v3-turbo-q8_0)
fi

CONTAINER_NAME="digue-smoke"
JFK_URL="https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"
# Where downloaded models are kept between runs (models are the only state worth keeping; set
# DIGUE_SMOKE_CACHE="" to force re-downloading).
CACHE_DIR="${DIGUE_SMOKE_CACHE:-$HOME/.cache/digue-smoke-models}"
PORT=18378

DIGUE_BIN="${DIGUE_BIN:-digue}"
command -v "$DIGUE_BIN" >/dev/null 2>&1 || {
    echo "error: $DIGUE_BIN not found in PATH (set DIGUE_BIN=/path/to/digue)" >&2
    exit 2
}
command -v docker >/dev/null 2>&1 || { echo "error: docker not found" >&2; exit 2; }
command -v ffmpeg >/dev/null 2>&1 || { echo "error: ffmpeg not found" >&2; exit 2; }
docker ps >/dev/null 2>&1 || { echo "error: docker needs to work without sudo" >&2; exit 2; }

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/digue-smoke-XXXXXX")"
DATA_DIR="$TEST_ROOT/data"
CONFIG="$TEST_ROOT/config.toml"
RUNTIME="$TEST_ROOT/runtime"
mkdir -p "$DATA_DIR" "$RUNTIME"

FAILS=()
SKIPPED=0
PASSED=0
CURRENT=""
FAIL_FAST="${FAIL_FAST:-0}"
KEEP_ON_FAIL="${KEEP_ON_FAIL:-0}"

cleanup() {
    local rc=$?
    trap - INT TERM EXIT
    # the container itself and any benchmark backup left by a Ctrl+c mid-benchmark
    docker ps -aq --filter "name=^${CONTAINER_NAME}" | xargs -r docker rm -f >/dev/null 2>&1 || true
    if [ "$rc" -ne 0 ] && [ "$KEEP_ON_FAIL" = "1" ]; then
        echo >&2
        echo "KEEP_ON_FAIL=1: state kept at $TEST_ROOT" >&2
    else
        rm -rf "$TEST_ROOT"
    fi
    if [ "$rc" -ne 0 ]; then
        echo >&2
        echo "SMOKE TESTS FAILED (see failures above)" >&2
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

config_set() { # config_set section key value
    python3 - "$CONFIG" "$1" "$2" "$3" <<'EOF'
import sys
from pathlib import Path

path, section, key, value = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
text = path.read_text()
lines = text.splitlines()
out, in_section, replaced = [], False, False
for line in lines:
    if line.startswith("[") and not line.startswith(f"[{section}]"):
        in_section = False
    if line.strip() == f"[{section}]":
        in_section = True
    if in_section and (line.split("=")[0].strip() == key or line.split("=")[0].strip() == key.replace("_", "-")):
        is_plain = value.isdigit() or value in ("true", "false")
        rendered = f"{key} = {value}" if is_plain else f'{key} = "{value}"'
        out.append(rendered)
        replaced = True
        continue
    out.append(line)
if not replaced:
    if f"[{section}]" not in text:
        out.append("")
        out.append(f"[{section}]")
    is_plain = value.isdigit() or value in ("true", "false")
    rendered = f"{key} = {value}" if is_plain else f'{key} = "{value}"'
    out.append(rendered)
path.write_text("\n".join(out) + "\n")
EOF
}

dg() { "$DIGUE_BIN" -c "$CONFIG" "$@"; }

pass() { PASSED=$((PASSED + 1)); echo "  ok: $1"; }
skip() { SKIPPED=$((SKIPPED + 1)); echo "  SKIP: $1"; }
fail() {
    FAILS+=("${CURRENT}: $1")
    echo "  FAIL: $1" >&2
}
check() { # check "description" expected_rc actual_rc [output]
    local desc="$1" expected="$2" actual="$3" output="${4:-}"
    if [ "$actual" -eq "$expected" ]; then
        pass "$desc"
    else
        fail "$desc (expected rc=$expected, got rc=$actual)"
        [ -n "$output" ] && echo "    output: $(echo "$output" | head -3)" >&2
    fi
    if [ "${#FAILS[@]}" -gt 0 ] && [ "$FAIL_FAST" = "1" ]; then
        echo "FAIL_FAST=1: stopping at first failure" >&2
        exit 1
    fi
}
check_output() { # check_output "description" actual_output expected_substring
    local desc="$1" output="$2" needle="$3"
    if [[ "$output" == *"$needle"* ]]; then
        pass "$desc"
    else
        fail "$desc (output lacks \"$needle\")"
        echo "    output: $(echo "$output" | head -3)" >&2
    fi
}
section() { CURRENT="$1"; echo; echo "== $1"; }

# Setup

BACKEND="$SCRIPT_BACKEND"
if [ -z "$BACKEND" ]; then
    BACKEND="$("$DIGUE_BIN" -c "$CONFIG" detect 2>/dev/null)"
    case "$BACKEND" in
        amd) ;;
        *) BACKEND="cpu" ;;
    esac
fi
case "$BACKEND" in
    cpu | amd) ;;
    *) echo "error: backend must be cpu or amd (got: $BACKEND); nvidia/intel are out of scope here" >&2; exit 2 ;;
esac
export XDG_RUNTIME_DIR="$RUNTIME"
echo "backend: $BACKEND  container: $CONTAINER_NAME  root: $TEST_ROOT"

mkdir -p "$CACHE_DIR"
for model in "${MODELS[@]}"; do
    cache_file="$CACHE_DIR/ggml-$model.bin"
    if [ -f "$cache_file" ]; then
        mkdir -p "$DATA_DIR/models"
        cp "$cache_file" "$DATA_DIR/models/"
        echo "model $model: reused from cache"
    fi
done

cat >"$CONFIG" <<EOF
[server]
data-dir = "$DATA_DIR"
backend = "$BACKEND"
bind-ip = "127.0.0.1"
port = $PORT
container-name = "$CONTAINER_NAME"

[transcribe]
language = "en"
prompt = ""
output-format = "text"
max-line-length = 42
max-lines = 2
timeout = 600

[dictate]
audio-dir = "$DATA_DIR/audio"
display-server = "auto"
input-mode = "paste"
recorder = "auto"
max-duration = 300
save-audio = true
audio-format = "flac"
EOF

# CLI basics

section "CLI basics"
out="$(dg --version 2>/dev/null)"
rc=$?
check "--version exits 0" 0 "$rc" "$out"
check_output "--version prints version" "$out" "digue "
dg >/dev/null 2>&1
check "bare digue exits 1 (no accidental toggle)" 1 $?
out="$(dg --help 2>&1)"
missing=""
help_cmds="detect detect-language download server dictate transcribe convert batch-transcribe batch-simplify-vtt"
help_cmds="$help_cmds benchmark config doctor clean models"
for cmd in $help_cmds; do
    [[ "$out" == *"$cmd"* ]] || missing="$missing $cmd"
done
if [ -z "$missing" ]; then pass "--help lists all subcommands"; else fail "--help lacks:$missing"; fi
dg detect >/dev/null 2>&1
check "detect exits 0" 0 $?

# Config

section "Config: show/init/validation"
dg config show >/dev/null 2>&1
check "config show exits 0" 0 $?
out="$(dg config show)"
check_output "config show resolves data-dir" "$out" "data-dir = \"$DATA_DIR\""
check_output "config show resolves backend" "$out" "backend = \"$BACKEND\""
dg config show -f json 2>/dev/null | python3 -m json.tool >/dev/null 2>&1
check "config show -f json is valid JSON" 0 $?
dg config show -f json >/dev/null 2>&1
check "config show -f json exits 0" 0 $?
dg models >/dev/null 2>&1
check "models lists AVAILABLE_MODELS" 0 $?
out="$(dg models)"
check_output "models lists a quantized model" "$out" "small-q8_0"
check_output "models stars the selected model" "$out" "* "

init_path="$TEST_ROOT/created.toml"
dg config init -o "$init_path" >/dev/null 2>&1
check "config init -o creates file" 0 $?
dg config init -o "$init_path" >/dev/null 2>&1
check "config init refuses existing file" 1 $?
dg config init -o "$init_path" -f >/dev/null 2>&1
check "config init -f overwrites" 0 $?

bad_configs=(
    'bad-toml:[server'
    'bad-port:[server]
port = 70000'
    'bad-backend:[server]
backend = "wrong"'
    'unknown-key:[server]
prot = 8178'
    'conflicting-keys:[server]
data-dir = "a"
data_dir = "b"'
    'unknown-section:[servers]
port = 8178'
    'unknown-model:[models]
cpu = "gigante"'
    'bad-duration:[dictate]
max-duration = -1'
)
for entry in "${bad_configs[@]}"; do
    name="${entry%%:*}"
    body="${entry#*:}"
    file="$TEST_ROOT/$name.toml"
    printf '%s\n' "$body" >"$file"
    out="$("$DIGUE_BIN" -c "$file" config show 2>&1)"
    rc=$?
    check "invalid config rejected: $name" 1 "$rc" "$out"
    check_output "  error message present" "$out" "Error"
    [[ "$out" != *"Traceback"* ]] && pass "  no traceback: $name" || fail "  traceback leaked: $name"
done
out="$("$DIGUE_BIN" -c "$TEST_ROOT/unknown-key.toml" config show 2>&1)"
check_output "typo suggests the right key" "$out" 'did you mean "port"'

host_short="$(hostname | cut -d. -f1)"
cat >>"$CONFIG" <<EOF

[host.$host_short.dictate]
max-duration = 42
EOF
out="$(dg config show)"
check_output "host override applies (max-duration = 42)" "$out" "max-duration = 42"
python3 - "$CONFIG" <<'EOF'
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
text = text.split("\n[host.")[0]
Path(sys.argv[1]).write_text(text + "\n")
EOF
out="$(dg config show)"
[[ "$out" != *"max-duration = 42"* ]] && pass "host override removed" || fail "host override removal"

# Doctor

section "doctor"
out="$(dg doctor 2>&1)"
rc=$?
check "doctor exits 0" 0 "$rc" "$out"
check_output "doctor shows the config path" "$out" "$CONFIG"
[[ "$out" != *"Traceback"* ]] && pass "doctor without traceback" || fail "doctor traceback"
if [[ "$out" == *"[SKIP] not pulled"* || "$out" == *"[OK]"* ]]; then
    pass "doctor image section present"
else
    fail "doctor image section missing"
fi

# Model download

section "Model download"
first_model="${MODELS[0]}"
out="$(dg download "$first_model" 2>&1)"
rc=$?
check "download $first_model exits 0" 0 "$rc" "$out"
model_file="$DATA_DIR/models/ggml-$first_model.bin"
[ -f "$model_file" ] && pass "model file exists" || fail "model file missing: ggml-$first_model.bin"
[ -f "$DATA_DIR/models/ggml-silero-v6.2.0.bin" ] && pass "VAD model downloaded" || fail "VAD model missing"
out="$(dg download "$first_model" 2>&1)"
check_output "re-download says Already exists" "$out" "Already exists"

# Server lifecycle

section "Server lifecycle (real container)"
dg server destroy >/dev/null 2>&1 || true
out="$(dg server status 2>&1)"
rc=$?
check "status without container exits 1" 1 "$rc" "$out"
out="$(dg server start 2>&1)"
rc=$?
check "server start exits 0" 0 "$rc" "$out"
check_output "start waits for the model" "$out" "Server ready"
out="$(dg server status 2>&1)"
rc=$?
check "status running exits 0" 0 "$rc" "$out"
check_output "status reports responding" "$out" "responding"
out="$(curl --fail --silent "http://127.0.0.1:$PORT/" 2>&1)"
rc=$?
check "HTTP probe answers on 127.0.0.1:$PORT" 0 "$rc" "$out"
out="$(dg server start 2>&1)"
rc=$?
check "second start is a no-op (exit 0)" 0 "$rc" "$out"
check_output "second start message" "$out" "already running"
out="$(dg server stop 2>&1)"
rc=$?
check "server stop exits 0" 0 "$rc" "$out"
out="$(dg server status 2>&1)"
rc=$?
check "status after stop exits 1" 1 "$rc"
out="$(dg server start 2>&1)"
rc=$?
check "restart after stop exits 0" 0 "$rc" "$out"
name_in_docker="$(docker ps --format '{{.Names}}' | grep -x "$CONTAINER_NAME" || true)"
[ -n "$name_in_docker" ] && pass "container visible in docker ps" || fail "container not in docker ps"
if command -v ss >/dev/null 2>&1; then
    bind="$(ss -ltn 2>/dev/null | grep ":$PORT" || true)"
    [[ "$bind" == *"127.0.0.1:$PORT"* ]] && pass "port bound to 127.0.0.1" || fail "bind is not loopback: $bind"
else
    skip "port bind check (ss not installed)"
fi

section "server start --image and -n"
current_image="$(docker inspect --format '{{.Config.Image}}' "$CONTAINER_NAME" 2>/dev/null)"
out="$(dg server start --image "$current_image" 2>&1)"
rc=$?
check "start --image with the same image is a no-op" 0 "$rc" "$out"
check_output "  says already running" "$out" "already running"
out="$(dg server start -n "bad name" 2>&1)"
rc=$?
check "start -n with an invalid docker name fails cleanly" 1 "$rc" "$out"
[[ "$out" != *"Traceback"* ]] && pass "  no traceback" || fail "  traceback leaked"
docker ps -a --format '{{.Names}}' | grep -qx "bad name" && fail "  container with a bad name was created" || pass "  nothing created"

section "benchmark (one run, same container name, restores the server)"
out="$(dg benchmark "$JFK" -b "$BACKEND" -m "${MODELS[0]}" -r 1 --json 2>"$TEST_ROOT/bench-stderr.log")"
rc=$?
check "benchmark exits 0" 0 "$rc" "$(cat "$TEST_ROOT/bench-stderr.log")"
echo "$out" | python3 -m json.tool >/dev/null 2>&1 && pass "benchmark --json is valid JSON" || fail "benchmark JSON invalid"
check_output "benchmark summary printed" "$(cat "$TEST_ROOT/bench-stderr.log")" "Summary"
dg server status >/dev/null 2>&1
check "server restored and running after benchmark" 0 $?
docker ps -a --format '{{.Names}}' | grep -q "benchmark-backup" && fail "benchmark backup container left behind" || pass "no backup container left"

# Dictation (interactive: speak when asked)

mic_take() { # mic_take <recorder>: one interactive take (daemon in background, ENTER to start and to stop)
    local recorder="$1"
    config_set dictate recorder "$recorder"
    if ! command -v "$recorder" >/dev/null 2>&1; then
        skip "dictation with $recorder (binary not installed)"
        return
    fi
    local daemon_pid="" waited=0
    echo "  recorder: $recorder"
    echo "  >>> Press ENTER, then speak ONE SHORT SENTENCE IN ENGLISH for about 5 seconds <<<"
    read -r
    dg dictate >/dev/null 2>&1 &
    daemon_pid=$!
    # wait up to 10s for the daemon to publish its recording state (model already cached, server up)
    while [ "$waited" -lt 50 ]; do
        grep -q " recording " "$RUNTIME/digue-daemon.pid" 2>/dev/null && break
        sleep 0.2
        waited=$((waited + 1))
    done
    if grep -q " recording " "$RUNTIME/digue-daemon.pid" 2>/dev/null; then
        pass "$recorder: daemon reached recording state"
    else
        fail "$recorder: daemon never reached recording state"
    fi
    echo "  >>> Recording... press ENTER to stop <<<"
    read -r
    dg dictate >/dev/null 2>&1
    # delivery (transcribe + paste/archive) can take a while on cpu: cap at 300s then give up loudly
    waited=0
    while kill -0 "$daemon_pid" 2>/dev/null && [ "$waited" -lt 300 ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "$daemon_pid" 2>/dev/null; then
        fail "$recorder: daemon did not finish in 300s"
        kill -9 "$daemon_pid" 2>/dev/null
        return
    fi
    wait "$daemon_pid" 2>/dev/null
    local rc_daemon=$?
    if [ "$rc_daemon" -eq 0 ] || [ "$rc_daemon" -eq 1 ]; then
        pass "$recorder: daemon finished (exit $rc_daemon; 1 = paste into the focused window failed)"
    else
        fail "$recorder: unexpected daemon exit $rc_daemon"
    fi
    local latest_txt latest_audio
    latest_txt="$(find "$DATA_DIR/audio" -name '*.txt' -newer "$CONFIG" | sort | tail -1)"
    latest_audio="$(find "$DATA_DIR/audio" \( -name '*.flac' -o -name '*.wav' \) -newer "$CONFIG" | sort | tail -1)"
    if [ -n "$latest_txt" ] && [ -s "$latest_txt" ]; then
        pass "$recorder: transcript saved with content ($latest_txt)"
    else
        fail "$recorder: transcript missing or empty"
    fi
    if [ -n "$latest_audio" ]; then
        pass "$recorder: audio archived ($latest_audio)"
    else
        fail "$recorder: no audio archived"
    fi
}

section "Dictation with microphone (pw-record, then arecord)"
echo "This part is interactive: two short takes, one per recorder. The script prompts before recording."
mic_take pw-record
mic_take arecord
# cleanup: the daemon wrote its state inside $RUNTIME; anything left after both takes is a leak
runtime_pattern="$RUNTIME/digue-take-* $RUNTIME/digue-daemon* $RUNTIME/digue-*.wav $RUNTIME/digue-*.flac"
leftover_runtime=$(find $runtime_pattern 2>/dev/null | wc -l)
if [ "$leftover_runtime" -eq 0 ]; then
    pass "runtime dir clean after dictation"
else
    fail "runtime leftovers: $leftover_runtime"
fi
recorders_alive="$(pgrep -x pw-record; pgrep -x arecord)"
if [ -z "$recorders_alive" ]; then
    pass "no recorder process left"
else
    fail "recorder process still alive: $recorders_alive"
fi

# Transcription (JFK sample)

JFK="$TEST_ROOT/jfk.wav"
if [ -f "$CACHE_DIR/jfk.wav" ]; then
    cp "$CACHE_DIR/jfk.wav" "$JFK"
else
    curl --fail --silent --location --max-time 120 -o "$JFK" "$JFK_URL" || {
        echo "error: could not download the JFK sample" >&2
        exit 2
    }
    cp "$JFK" "$CACHE_DIR/jfk.wav"
fi
[ -s "$JFK" ] && pass "JFK sample ready" || fail "JFK sample is empty"

section "Transcription (JFK sample)"
stderr_file="$TEST_ROOT/transcribe-stderr.log"
out="$(dg transcribe "$JFK" 2>"$stderr_file")"
rc=$?
check "transcribe wav exits 0" 0 "$rc" "$out"
check_output "stdout is plain text" "$out" "country"
stderr_size=$(wc -c <"$stderr_file")
if [ "$stderr_size" -eq 0 ]; then
    pass "quiet mode: empty stderr"
else
    fail "stderr not empty in quiet mode ($stderr_size bytes)"
fi

out_file="$TEST_ROOT/out.txt"
dg transcribe "$JFK" -o "$out_file" >/dev/null 2>&1
check "-o writes the file" 0 $?
last_char="$(tail -c 1 "$out_file" | od -An -c | tr -d ' ')"
[ "$last_char" = '\n' ] && pass "output file ends with exactly one newline" || fail "output file newline: '$last_char'"
lines="$(wc -l <"$out_file")"
[ "$lines" -eq 1 ] && pass "text format is one line" || fail "text format has $lines lines"

# The model name is baked into the container's command line: changing [models] only
# takes effect after destroy + start (which downloads a missing model first).
for model in "${MODELS[@]:1}"; do
    config_set models "$BACKEND" "$model"
    out="$(dg server destroy 2>&1 && dg server start 2>&1)"
    rc=$?
    check "server recreated with model $model" 0 "$rc" "$out"
    [ -f "$DATA_DIR/models/ggml-$model.bin" ] && pass "model $model downloaded" || fail "ggml-$model.bin missing"
    docker inspect --format '{{join .Args " "}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "ggml-$model.bin" \
        && pass "container runs ggml-$model.bin" || fail "container command line lacks ggml-$model.bin"
    out="$(dg transcribe "$JFK" 2>&1)"
    rc=$?
    check "transcribe with model $model exits 0" 0 "$rc" "$out"
    check_output "  model $model transcribes the sample" "$out" "country"
done
config_set models "$BACKEND" "${MODELS[0]}"
dg server destroy >/dev/null 2>&1
dg server start >/dev/null 2>&1
check "server back on ${MODELS[0]}" 0 $?
for model in "${MODELS[@]}"; do
    cache_file="$CACHE_DIR/ggml-$model.bin"
    if [ -f "$DATA_DIR/models/ggml-$model.bin" ] && [ ! -f "$cache_file" ]; then
        cp "$DATA_DIR/models/ggml-$model.bin" "$cache_file"
    fi
done

dg transcribe "$JFK" -l en >/dev/null 2>&1
check "-l en exits 0" 0 $?
dg transcribe "$JFK" -p "test prompt" >/dev/null 2>&1
check "-p prompt exits 0" 0 $?
out="$(dg transcribe "$TEST_ROOT/no-such-file.wav" 2>&1)"
rc=$?
check "missing file exits 1" 1 "$rc" "$out"
check_output "missing file message" "$out" "file not found"
out="$(dg transcribe "$DATA_DIR" 2>&1)"
rc=$?
check "directory as input exits 1" 1 "$rc"
head -c 1000 /dev/urandom >"$TEST_ROOT/corrupt.mp3"
out="$(dg transcribe "$TEST_ROOT/corrupt.mp3" 2>&1)"
rc=$?
check "corrupt file exits 1" 1 "$rc"
[[ "$out" != *"Traceback"* ]] && pass "corrupt file without traceback" || fail "corrupt file traceback"

section "Transcription: formats"
dg transcribe "$JFK" -f text -o "$TEST_ROOT/f.txt" >/dev/null 2>&1
check "format text" 0 $?
dg transcribe "$JFK" -f timestamps -o "$TEST_ROOT/f-ts.txt" >/dev/null 2>&1
check "format timestamps" 0 $?
first_line="$(head -1 "$TEST_ROOT/f-ts.txt")"
if echo "$first_line" | grep -qE '^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\] '; then
    pass "timestamps lines start with [HH:MM:SS]"
else
    fail "timestamps format: $first_line"
fi
dg transcribe "$JFK" -f vtt -o "$TEST_ROOT/f.vtt" >/dev/null 2>&1
check "format vtt" 0 $?
head -1 "$TEST_ROOT/f.vtt" | grep -q '^WEBVTT$' && pass "vtt starts with WEBVTT" || fail "vtt header"
! head -3 "$TEST_ROOT/f.vtt" | grep -q '^ ' && pass "vtt cues not indented" || fail "vtt cues indented"
dg transcribe "$JFK" -f srt -o "$TEST_ROOT/f.srt" >/dev/null 2>&1
check "format srt" 0 $?
grep -q -- '-->' "$TEST_ROOT/f.srt" && pass "srt has cue arrows" || fail "srt missing -->"
head -1 "$TEST_ROOT/f.srt" | grep -q '^[0-9]*$' && pass "srt starts with an index" || fail "srt index"

section "Transcription: ffmpeg conversion (in memory)"
ffmpeg -v error -i "$JFK" -c:a aac -b:a 64k "$TEST_ROOT/jfk.m4a" -y 2>/dev/null
[ -f "$TEST_ROOT/jfk.m4a" ] && pass "m4a sample created" || fail "m4a sample creation"
ffmpeg -v error -i "$JFK" -c:a libopus -b:a 24k -vn "$TEST_ROOT/jfk-opus.ogg" -y 2>/dev/null
[ -f "$TEST_ROOT/jfk-opus.ogg" ] && pass "opus-in-ogg sample created" || fail "opus sample creation"
out="$(dg transcribe "$TEST_ROOT/jfk.m4a" -v 2>&1)"
rc=$?
check "m4a transcribes (ffmpeg upfront)" 0 "$rc" "$out"
check_output "  conversion message shown" "$out" "with ffmpeg"
out="$(dg transcribe "$TEST_ROOT/jfk-opus.ogg" -v 2>&1)"
rc=$?
check "opus-in-ogg transcribes (HTTP 400 retry)" 0 "$rc" "$out"
check_output "  400-retry message shown" "$out" "HTTP 400"
after_wavs="$(find "$TEST_ROOT" -name '*.wav' ! -name 'jfk.wav' \
    | grep -v "$DATA_DIR/models" | grep -v "$RUNTIME" | wc -l)"
if [ "$after_wavs" -eq 0 ]; then
    pass "no wav left beside the input (in-memory conversion)"
else
    fail "conversion left wav files: $after_wavs"
fi

section "detect-language"
out="$(dg detect-language "$JFK" 2>/dev/null)"
rc=$?
check "detect-language exits 0" 0 "$rc" "$out"
[ ${#out} -eq 2 ] && pass "prints a two-letter code: $out" || fail "detect-language output: '$out'"
dg detect-language "$JFK" --json 2>/dev/null | python3 -m json.tool >/dev/null 2>&1
check "detect-language --json is valid JSON" 0 $?

# Convert

section "convert"
out="$(dg convert "$TEST_ROOT/f.vtt" 2>/dev/null)"
rc=$?
check "vtt to stdout (text)" 0 "$rc"
check_output "converted text has content" "$out" "country"
dg convert "$TEST_ROOT/f.vtt" "$TEST_ROOT/c-ts.txt" >/dev/null 2>&1
check "vtt -> timestamps file (ext-inferred)" 0 $?
grep -q '^\[' "$TEST_ROOT/c-ts.txt" && pass "timestamps output format" || fail "vtt->timestamps format"
dg convert "$TEST_ROOT/f.srt" "$TEST_ROOT/c.vtt" >/dev/null 2>&1
check "srt -> vtt" 0 $?
head -1 "$TEST_ROOT/c.vtt" | grep -q '^WEBVTT$' && pass "srt->vtt header" || fail "srt->vtt header"
dg convert "$TEST_ROOT/f.vtt" "$TEST_ROOT/c.srt" >/dev/null 2>&1
check "vtt -> srt" 0 $?
out="$(printf 'WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nhello <c>world</c>\n' | dg convert -f vtt - 2>/dev/null)"
rc=$?
check "stdin -> stdout" 0 "$rc"
check_output "vtt tags stripped" "$out" "hello world"
out="$(printf 'hello\n' | dg convert - 2>&1)"
rc=$?
check "stdin without -f exits 1" 1 "$rc"
printf '[00:99:00] bad\n' >"$TEST_ROOT/bad-ts.txt"
out="$(dg convert "$TEST_ROOT/bad-ts.txt" "$TEST_ROOT/bad.vtt" 2>&1)"
rc=$?
check "malformed timestamp exits 1" 1 "$rc"
out="$(dg convert "$TEST_ROOT/no-such.vtt" 2>&1)"
rc=$?
check "missing input exits 1" 1 "$rc"
dg convert "$TEST_ROOT/f.vtt" "$TEST_ROOT/roundtrip.srt" >/dev/null 2>&1
dg convert "$TEST_ROOT/roundtrip.srt" "$TEST_ROOT/roundtrip.vtt" >/dev/null 2>&1
if diff <(grep -v '^WEBVTT' "$TEST_ROOT/f.vtt") <(grep -v '^WEBVTT' "$TEST_ROOT/roundtrip.vtt") >/dev/null 2>&1; then
    pass "vtt -> srt -> vtt roundtrip preserves cues"
else
    fail "vtt roundtrip changed cues"
fi

# Batch

section "batch-transcribe"
mkdir -p "$TEST_ROOT/batch-in" "$TEST_ROOT/batch-out"
cp "$JFK" "$TEST_ROOT/batch-in/one.wav"
ffmpeg -v error -i "$JFK" -c:a libmp3lame -q:a 4 "$TEST_ROOT/batch-in/two.mp3" -y 2>/dev/null
cp "$TEST_ROOT/corrupt.mp3" "$TEST_ROOT/batch-in/three.mp3"
printf 'ignored\n' >"$TEST_ROOT/batch-in/note.txt"
out="$(dg batch-transcribe "$TEST_ROOT/batch-in" "$TEST_ROOT/batch-out" 2>&1)"
rc=$?
check "batch with a bad file still processes the rest (exit 1)" 1 "$rc" "$out"
check_output "batch summary counts failures" "$out" "1 failed"
check_output "batch summary counts successes" "$out" "2 succeeded"
[ -f "$TEST_ROOT/batch-out/one.txt" ] && pass "one.wav transcribed" || fail "one.txt missing"
[ -f "$TEST_ROOT/batch-out/two.txt" ] && pass "two.mp3 transcribed" || fail "two.txt missing"
if [ ! -f "$TEST_ROOT/batch-out/three.txt" ]; then
    pass "corrupt file produced no output"
else
    fail "corrupt file produced three.txt"
fi
leftover="$(find "$TEST_ROOT/batch-out" -name '*.tmp' | wc -l)"
[ "$leftover" -eq 0 ] && pass "no .tmp leftovers" || fail ".tmp leftovers: $leftover"
out="$(dg batch-transcribe "$TEST_ROOT/batch-in" "$TEST_ROOT/batch-out" 2>&1)"
check_output "rerun skips completed files" "$out" "Skipping 2"
dg transcribe "$JFK" -f vtt -o "$TEST_ROOT/batch-out/one.vtt" >/dev/null 2>&1
mkdir -p "$TEST_ROOT/simplified"
out="$(dg batch-simplify-vtt "$TEST_ROOT/batch-out" "$TEST_ROOT/simplified" 2>&1)"
rc=$?
check "batch-simplify-vtt exits 0" 0 "$rc" "$out"
[ -f "$TEST_ROOT/simplified/one.txt" ] && pass "vtt simplified to txt" || fail "simplified/one.txt missing"

# Server auto-start

section "transcribe starts the server itself"
dg server stop >/dev/null 2>&1
dg transcribe "$JFK" >/dev/null 2>&1
check "transcribe with server down auto-starts it" 0 $?
dg server status >/dev/null 2>&1
check "server is running again" 0 $?

# Remote backend

section "remote backend (refusals, no container touched)"
containers_before="$(docker ps -a --format '{{.Names}}' | sort)"
cp "$CONFIG" "$TEST_ROOT/remote.toml"
python3 - "$TEST_ROOT/remote.toml" <<EOF
from pathlib import Path

text = Path("$TEST_ROOT/remote.toml").read_text()
text = text.replace('backend = "$BACKEND"', 'backend = "remote"')
text = text.replace("port = $PORT", "port = $((PORT + 1))")
Path("$TEST_ROOT/remote.toml").write_text(text)
EOF
dgr() { "$DIGUE_BIN" -c "$TEST_ROOT/remote.toml" "$@"; }
out="$(dgr server status 2>&1)"
rc=$?
check "remote status without server exits 1" 1 "$rc" "$out"
check_output "remote status message" "$out" "remote backend, not responding"
out="$(dgr server start 2>&1)"
rc=$?
check "remote start refuses (exit 1)" 1 "$rc" "$out"
check_output "remote start suggests ssh tunnel" "$out" "ssh -NfL"
out="$(dgr server stop 2>&1)"
rc=$?
check "remote stop refuses" 1 "$rc"
out="$(dgr server destroy 2>&1)"
rc=$?
check "remote destroy refuses" 1 "$rc"
out="$(dgr benchmark "$JFK" 2>&1)"
rc=$?
check "remote benchmark refuses" 1 "$rc"
out="$(dgr download 2>&1)"
rc=$?
check "remote download without model refuses" 1 "$rc"
check_output "  remote download message" "$out" "remote machine"
out="$(dgr transcribe "$JFK" 2>&1)"
rc=$?
check "remote transcribe fails cleanly" 1 "$rc"
check_output "  suggests the tunnel" "$out" "ssh -NfL"
containers_after="$(docker ps -a --format '{{.Names}}' | sort)"
if [ "$containers_before" = "$containers_after" ]; then
    pass "remote mode touched no container"
else
    fail "remote mode changed containers"
fi

# Clean

section "clean"
out="$(dg clean -f 2>&1)"
rc=$?
check "clean -f exits 0" 0 "$rc" "$out"
remaining="$(find "$DATA_DIR/audio" \( -name '*.txt' -o -name '*.flac' -o -name '*.wav' \) 2>/dev/null | wc -l)"
if [ "$remaining" -eq 0 ]; then
    pass "clean removed the dictation files"
else
    fail "clean left $remaining files"
fi
mkdir -p "$DATA_DIR/audio/2026/09"
out="$(dg clean -f 2>&1)"
rc=$?
check "clean on empty dir exits 0" 0 "$rc"
check_output "clean says nothing to remove" "$out" "Nothing to remove"
rmdir "$DATA_DIR/audio/2026/09" "$DATA_DIR/audio/2026" "$DATA_DIR/audio" 2>/dev/null
out="$(dg clean -f 2>&1)"
rc=$?
check "clean without audio dir exits 0" 0 "$rc"
check_output "clean reports missing dir" "$out" "Audio directory does not exist"

# Teardown

section "Teardown"
out="$(dg server destroy 2>&1)"
rc=$?
check "server destroy exits 0" 0 "$rc" "$out"
docker ps -a --format '{{.Names}}' | grep -x "$CONTAINER_NAME" >/dev/null 2>&1
if [ $? -eq 0 ]; then
    fail "container still exists after destroy"
else
    pass "container removed"
fi
out="$(dg server status 2>&1)"
rc=$?
check "status after destroy exits 1" 1 "$rc"

# Report

echo
echo "==========================================="
echo "Smoke tests: $PASSED passed, $SKIPPED skipped, ${#FAILS[@]} failed"
echo "==========================================="
if [ "${#FAILS[@]}" -gt 0 ]; then
    for f in "${FAILS[@]}"; do echo "FAIL: $f"; done
    exit 1
fi
echo "All green."
exit 0
