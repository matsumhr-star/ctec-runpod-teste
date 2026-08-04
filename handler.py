import base64
import json
import math
import os
import re
import subprocess
import tempfile
import time
import threading
import urllib.request
import urllib.parse
import difflib
import statistics
from pathlib import Path
from typing import Any

import runpod
import torch
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from num2words import num2words

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_VERSION = os.getenv("CTEC_CHATTERBOX_MODEL", "v3").strip().lower() or "v3"
MAX_TEXT_CHARS = int(os.getenv("CTEC_MAX_TEXT_CHARS", "120000"))
MAX_REFERENCE_BYTES = int(os.getenv("CTEC_MAX_REFERENCE_BYTES", str(30 * 1024 * 1024)))
MAX_RESULT_BASE64_BYTES = int(os.getenv("CTEC_MAX_RESULT_BASE64_BYTES", str(14 * 1024 * 1024)))
WORKER_CONTRACT_VERSION = 2

_MODEL: ChatterboxMultilingualTTS | None = None
_MODEL_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()
_WHISPER = None
_WHISPER_LOCK = threading.Lock()


PROFILES: dict[str, dict[str, Any]] = {
    "law_formal": {
        "speed": 0.94,
        "exaggeration": 0.35,
        "cfg_weight": 0.62,
        "temperature": 0.72,
        "repetition_penalty": 1.25,
        "min_p": 0.05,
        "top_p": 0.95,
        "pause_sentence_ms": 330,
        "pause_paragraph_ms": 650,
        "pitch_semitones": 0.0,
        "gain_db": 0.0,
        "normalize": True,
        "trim_silence": False,
        "text_mode": "law",
    },
    "law_natural": {
        "speed": 0.98,
        "exaggeration": 0.48,
        "cfg_weight": 0.50,
        "temperature": 0.78,
        "repetition_penalty": 1.20,
        "min_p": 0.05,
        "top_p": 1.0,
        "pause_sentence_ms": 260,
        "pause_paragraph_ms": 520,
        "pitch_semitones": 0.0,
        "gain_db": 0.0,
        "normalize": True,
        "trim_silence": False,
        "text_mode": "law",
    },
    "professor": {
        "speed": 0.98,
        "exaggeration": 0.55,
        "cfg_weight": 0.46,
        "temperature": 0.80,
        "repetition_penalty": 1.20,
        "min_p": 0.05,
        "top_p": 1.0,
        "pause_sentence_ms": 240,
        "pause_paragraph_ms": 440,
        "pitch_semitones": 0.0,
        "gain_db": 0.5,
        "normalize": True,
        "trim_silence": False,
        "text_mode": "general",
    },
    "podcast_calm": {
        "speed": 0.93,
        "exaggeration": 0.43,
        "cfg_weight": 0.54,
        "temperature": 0.76,
        "repetition_penalty": 1.20,
        "min_p": 0.05,
        "top_p": 0.98,
        "pause_sentence_ms": 300,
        "pause_paragraph_ms": 520,
        "pitch_semitones": -0.3,
        "gain_db": 0.0,
        "normalize": True,
        "trim_silence": False,
        "text_mode": "general",
    },
    "podcast_energetic": {
        "speed": 1.04,
        "exaggeration": 0.72,
        "cfg_weight": 0.34,
        "temperature": 0.88,
        "repetition_penalty": 1.18,
        "min_p": 0.04,
        "top_p": 1.0,
        "pause_sentence_ms": 180,
        "pause_paragraph_ms": 340,
        "pitch_semitones": 0.4,
        "gain_db": 0.8,
        "normalize": True,
        "trim_silence": False,
        "text_mode": "general",
    },
    "summary_fast": {
        "speed": 1.15,
        "exaggeration": 0.44,
        "cfg_weight": 0.42,
        "temperature": 0.76,
        "repetition_penalty": 1.22,
        "min_p": 0.05,
        "top_p": 0.96,
        "pause_sentence_ms": 130,
        "pause_paragraph_ms": 240,
        "pitch_semitones": 0.0,
        "gain_db": 0.0,
        "normalize": True,
        "trim_silence": True,
        "text_mode": "general",
    },
    "question_explained": {
        "speed": 0.97,
        "exaggeration": 0.60,
        "cfg_weight": 0.43,
        "temperature": 0.82,
        "repetition_penalty": 1.20,
        "min_p": 0.05,
        "top_p": 1.0,
        "pause_sentence_ms": 250,
        "pause_paragraph_ms": 470,
        "pitch_semitones": 0.0,
        "gain_db": 0.4,
        "normalize": True,
        "trim_silence": False,
        "text_mode": "general",
    },
    "institutional": {
        "speed": 0.96,
        "exaggeration": 0.34,
        "cfg_weight": 0.64,
        "temperature": 0.70,
        "repetition_penalty": 1.25,
        "min_p": 0.05,
        "top_p": 0.94,
        "pause_sentence_ms": 280,
        "pause_paragraph_ms": 480,
        "pitch_semitones": -0.2,
        "gain_db": 0.3,
        "normalize": True,
        "trim_silence": False,
        "text_mode": "general",
    },
}

SUPPORTED_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
    "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "sim", "on"}


def load_voice_library() -> dict[str, str]:
    raw = os.getenv("CTEC_VOICE_LIBRARY_JSON", "{}").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return {str(k): str(v) for k, v in parsed.items() if str(v).strip()}
    except json.JSONDecodeError as exc:
        print(f"[CTEC] CTEC_VOICE_LIBRARY_JSON inválido: {exc}", flush=True)
        return {}


def get_model() -> ChatterboxMultilingualTTS:
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                print(
                    f"[CTEC] Carregando Chatterbox Multilingual {MODEL_VERSION} em {DEVICE}...",
                    flush=True,
                )
                _MODEL = ChatterboxMultilingualTTS.from_pretrained(
                    device=DEVICE,
                )
                print("[CTEC] Modelo carregado com sucesso.", flush=True)
    return _MODEL



ROMAN_VALUES = {
    "I": 1, "V": 5, "X": 10, "L": 50,
    "C": 100, "D": 500, "M": 1000,
}


def roman_to_int(value: str) -> int:
    value = value.upper().strip()
    total = 0
    previous = 0
    for char in reversed(value):
        current = ROMAN_VALUES.get(char, 0)
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def number_words(value: int, ordinal: bool = False) -> str:
    try:
        return num2words(value, lang="pt_BR", to="ordinal" if ordinal else "cardinal")
    except Exception:
        return str(value)


def apply_pronunciation_dictionary(
    text: str,
    custom_dictionary: list[dict[str, Any]] | None,
) -> str:
    items = custom_dictionary or []
    items = sorted(
        items,
        key=lambda item: len(str(item.get("source") or "")),
        reverse=True,
    )
    for item in items:
        source = str(item.get("source") or "").strip()
        spoken = str(item.get("spoken") or "").strip()
        if not source or not spoken:
            continue
        text = re.sub(re.escape(source), spoken, text, flags=re.IGNORECASE)
    return text


def normalize_law_text(
    text: str,
    custom_dictionary: list[dict[str, Any]] | None = None,
) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^[,;:\s]+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    text = apply_pronunciation_dictionary(text, custom_dictionary)

    text = re.sub(r"§\s*único", "Parágrafo único", text, flags=re.IGNORECASE)
    text = re.sub(
        r"§\s*(\d+)\s*[º°]?",
        lambda match: f"Parágrafo {number_words(int(match.group(1)), True)}",
        text,
    )
    text = re.sub(
        r"\bArts?\.\s*(\d+)\s*[º°]?",
        lambda match: (
            f"Artigo {number_words(int(match.group(1)), True)}"
            if int(match.group(1)) <= 9
            else f"Artigo {number_words(int(match.group(1)))}"
        ),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bArtigo\s+(\d+)\s*[º°]?",
        lambda match: (
            f"Artigo {number_words(int(match.group(1)), True)}"
            if int(match.group(1)) <= 9
            else f"Artigo {number_words(int(match.group(1)))}"
        ),
        text,
        flags=re.IGNORECASE,
    )

    def roman_line(match: re.Match[str]) -> str:
        value = roman_to_int(match.group(1))
        return f"{match.group(2)}Inciso {number_words(value)}. "

    text = re.sub(
        r"(?m)^\s*([IVXLCDM]{1,12})\s*[—–-]\s*",
        lambda match: f"Inciso {number_words(roman_to_int(match.group(1)))}. ",
        text,
    )
    text = re.sub(
        r"\binciso\s+([IVXLCDM]{1,12})\b",
        lambda match: f"inciso {number_words(roman_to_int(match.group(1)))}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^\s*([a-z])\)\s*",
        lambda match: f"Alínea {match.group(1)}. ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^\s*(\d+)\.\s+",
        lambda match: f"Item {number_words(int(match.group(1)))}. ",
        text,
    )

    fixed = [
        (r"\bcaput\b", "cáput"),
        (r"\bn[.º°]\s*", "número "),
        (r"\bc/c\b", "combinado com"),
        (r"\bCF/88\b", "Constituição Federal de mil novecentos e oitenta e oito"),
        (r"\bCRFB/88\b", "Constituição da República Federativa do Brasil de mil novecentos e oitenta e oito"),
        (r"\bCPP\b", "Código de Processo Penal"),
        (r"\bCP\b", "Código Penal"),
        (r"\bCPC\b", "Código de Processo Civil"),
        (r"\bSTF\b", "Supremo Tribunal Federal"),
        (r"\bSTJ\b", "Superior Tribunal de Justiça"),
    ]
    for pattern, spoken in fixed:
        text = re.sub(pattern, spoken, text, flags=re.IGNORECASE)

    text = re.sub(
        r"\bLei\s+número\s+(\d{1,6})(?:\.(\d{3}))?/(\d{2,4})\b",
        lambda match: (
            "Lei número "
            + number_words(int((match.group(1) or "") + (match.group(2) or "")))
            + ", de "
            + number_words(
                int(match.group(3))
                if len(match.group(3)) == 4
                else 2000 + int(match.group(3))
            )
        ),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(\d+(?:[.,]\d+)?)%",
        lambda match: f"{number_words(int(float(match.group(1).replace(',', '.'))))} por cento",
        text,
    )

    text = re.sub(r"\s*;\s*", ";\n", text)
    text = re.sub(r"\s*:\s*", ":\n", text)
    text = re.sub(r"\s*[—–]\s*", ". ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +([,.;:])", r"\1", text)
    return text.strip()



def prepare_text(text: str, mode: str, custom_dictionary=None) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if mode == "law":
        text = normalize_law_text(text, custom_dictionary)
    return text


def split_long_unit(value: str, limit: int) -> list[str]:
    """Divide uma frase longa primeiro por orações e só então por palavras."""
    clauses = [
        item.strip()
        for item in re.split(r"(?<=[,;:])\s+", value)
        if item.strip()
    ]
    output: list[str] = []
    current = ""
    for clause in clauses:
        if len(clause) <= limit:
            candidate = f"{current} {clause}".strip()
            if current and len(candidate) > limit:
                output.append(current)
                current = clause
            else:
                current = candidate
            continue
        if current:
            output.append(current)
            current = ""
        piece = ""
        for word in clause.split():
            candidate = f"{piece} {word}".strip()
            if piece and len(candidate) > limit:
                output.append(piece)
                piece = word
            else:
                piece = candidate
        if piece:
            output.append(piece)
    if current:
        output.append(current)
    return output


def split_text(
    text: str,
    limit: int,
    *,
    preserve_complete_sentences: bool = True,
    split_by_legal_structure: bool = True,
    context_margin_words: int = 0,
) -> list[tuple[str, bool]]:
    """Retorna (trecho, fim_de_paragrafo) sem perder nem repetir palavras."""
    margin = int(clamp(context_margin_words, 0, 20)) * 7
    effective_limit = max(100, int(limit) - margin)
    source = text if split_by_legal_structure else re.sub(r"\s*\n+\s*", " ", text)
    paragraphs = [p.strip() for p in re.split(r"\n+", source) if p.strip()]
    chunks: list[tuple[str, bool]] = []

    for paragraph in paragraphs:
        sentences = (
            re.split(r"(?<=[.!?;:])\s+", paragraph)
            if preserve_complete_sentences
            else [paragraph]
        )
        current = ""
        local: list[str] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > effective_limit:
                if current:
                    local.append(current)
                    current = ""
                local.extend(split_long_unit(sentence, effective_limit))
                continue
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > effective_limit:
                local.append(current)
                current = sentence
            else:
                current = candidate

        if current:
            local.append(current)
        for index, item in enumerate(local):
            chunks.append((item, index == len(local) - 1))

    return chunks


def download_url(url: str, destination: Path) -> None:
    parsed = urllib.parse.urlparse(url)
    allowed_hosts = {
        "firebasestorage.googleapis.com",
        "storage.googleapis.com",
    }
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError("A URL da referência não pertence ao Firebase Storage.")
    request = urllib.request.Request(url, headers={"User-Agent": "CTEC-Voice-Worker/2.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_REFERENCE_BYTES:
            raise ValueError("O áudio de referência excede o limite permitido.")
        data = response.read(MAX_REFERENCE_BYTES + 1)
    if len(data) > MAX_REFERENCE_BYTES:
        raise ValueError("O áudio de referência excede o limite permitido.")
    destination.write_bytes(data)


def save_reference_audio(data: dict[str, Any], root: Path) -> Path:
    voice_id = str(data.get("voice_id") or "").strip()
    url = str(data.get("reference_audio_url") or "").strip()
    b64 = str(data.get("reference_audio_base64") or "").strip()

    if voice_id and not url and not b64:
        url = load_voice_library().get(voice_id, "")
        if not url:
            raise ValueError(f"voice_id não encontrado na biblioteca: {voice_id}")

    if url:
        suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}:
            suffix = ".wav"
        path = root / f"reference{suffix}"
        download_url(url, path)
        return path

    if b64:
        match = re.match(r"^data:audio/([^;]+);base64,", b64)
        mime = match.group(1).lower() if match else "wav"
        suffix_map = {
            "wav": ".wav", "x-wav": ".wav", "mpeg": ".mp3", "mp3": ".mp3",
            "mp4": ".m4a", "m4a": ".m4a", "ogg": ".ogg", "flac": ".flac",
            "aac": ".aac",
        }
        path = root / f"reference{suffix_map.get(mime, '.wav')}"
        cleaned = re.sub(r"^data:audio/[^;]+;base64,", "", b64)
        raw = base64.b64decode(cleaned, validate=True)
        if len(raw) > MAX_REFERENCE_BYTES:
            raise ValueError("O áudio de referência excede o limite permitido.")
        path.write_bytes(raw)
        return path

    raise ValueError(
        "Informe voice_id, reference_audio_url ou reference_audio_base64."
    )


def prepare_reference_audio(source: Path, root: Path) -> tuple[Path, dict[str, Any]]:
    """Decodifica e equilibra a amostra sem alterar o arquivo original."""
    decoded = root / "reference_decoded.wav"
    prepared = root / "reference_prepared.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(decoded),
        ],
        check=True,
    )
    metrics = analyze_reference_audio(decoded)
    duration = float(metrics.get("durationSeconds") or 0)
    if duration < 3:
        raise ValueError(
            "A amostra precisa ter pelo menos 3 segundos; use de 15 a 60 segundos."
        )
    if duration > 300:
        raise ValueError("A amostra não pode ultrapassar cinco minutos.")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(decoded),
            "-af",
            "highpass=f=60,lowpass=f=11500,"
            "loudnorm=I=-20:TP=-3:LRA=7",
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(prepared),
        ],
        check=True,
    )
    return prepared, metrics


def resolve_settings(data: dict[str, Any]) -> dict[str, Any]:
    profile_name = str(data.get("profile") or "law_natural").strip().lower()
    base = dict(PROFILES.get(profile_name, PROFILES["law_natural"]))
    base.setdefault("stability", 0.72)
    base.setdefault("voice_fidelity", 0.78)
    base.setdefault("pause_comma_ms", 250)
    base.setdefault("pause_colon_ms", 380)
    base.setdefault("initial_silence_ms", 180)
    base.setdefault("final_silence_ms", 260)
    base.setdefault("chunk_overlap_words", 8)
    base.setdefault("preserve_complete_sentences", True)
    base.setdefault("split_by_legal_structure", True)

    numeric_limits = {
        "speed": (0.70, 1.35),
        "exaggeration": (0.0, 1.0),
        "cfg_weight": (0.0, 1.0),
        "temperature": (0.1, 1.5),
        "repetition_penalty": (1.0, 2.0),
        "min_p": (0.0, 1.0),
        "top_p": (0.05, 1.0),
        "pause_sentence_ms": (0, 2500),
        "pause_paragraph_ms": (0, 5000),
        "pause_comma_ms": (0, 1200),
        "pause_colon_ms": (0, 1600),
        "initial_silence_ms": (0, 2000),
        "final_silence_ms": (0, 3000),
        "chunk_overlap_words": (0, 20),
        "stability": (0.0, 1.0),
        "voice_fidelity": (0.0, 1.0),
        "pitch_semitones": (-6.0, 6.0),
        "gain_db": (-12.0, 12.0),
        "chunk_limit": (100, 600),
    }

    for key, (minimum, maximum) in numeric_limits.items():
        if key in data:
            base[key] = clamp(float(data[key]), minimum, maximum)

    for key in (
        "normalize",
        "trim_silence",
        "preserve_complete_sentences",
        "split_by_legal_structure",
    ):
        if key in data:
            base[key] = to_bool(data[key], bool(base.get(key, False)))

    if "text_mode" in data:
        base["text_mode"] = str(data["text_mode"]).strip().lower()

    base["profile"] = profile_name
    base["chunk_limit"] = int(base.get("chunk_limit", 320))
    for key in (
        "pause_sentence_ms",
        "pause_paragraph_ms",
        "pause_comma_ms",
        "pause_colon_ms",
        "initial_silence_ms",
        "final_silence_ms",
        "chunk_overlap_words",
    ):
        base[key] = int(base[key])
    return base


def public_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Contrato devolvido ao Flutter e persistido nas Functions."""
    return {
        "contractVersion": WORKER_CONTRACT_VERSION,
        "profile": settings["profile"],
        "speed": settings["speed"],
        "exaggeration": settings["exaggeration"],
        "cfg_weight": settings["cfg_weight"],
        "temperature": settings["temperature"],
        "repetition_penalty": settings["repetition_penalty"],
        "min_p": settings["min_p"],
        "top_p": settings["top_p"],
        "stability": settings["stability"],
        "voiceFidelity": settings["voice_fidelity"],
        "pause_sentence_ms": settings["pause_sentence_ms"],
        "pause_paragraph_ms": settings["pause_paragraph_ms"],
        "commaPauseMs": settings["pause_comma_ms"],
        "colonPauseMs": settings["pause_colon_ms"],
        "initialSilenceMs": settings["initial_silence_ms"],
        "finalSilenceMs": settings["final_silence_ms"],
        "chunk_limit": settings["chunk_limit"],
        "chunkOverlapWords": settings["chunk_overlap_words"],
        "preserveCompleteSentences": settings["preserve_complete_sentences"],
        "splitByLegalStructure": settings["split_by_legal_structure"],
        "text_mode": settings["text_mode"],
        "normalize": settings["normalize"],
        "trim_silence": settings["trim_silence"],
        "pitch_semitones": settings["pitch_semitones"],
        "gain_db": settings["gain_db"],
    }


def build_ffmpeg_filter(
    settings: dict[str, Any],
    *,
    include_edge_silence: bool = True,
) -> str:
    filters: list[str] = []

    speed = float(settings["speed"])
    filters.append(f"atempo={speed:.4f}")

    pitch = float(settings["pitch_semitones"])
    if abs(pitch) > 0.001:
        ratio = math.pow(2.0, pitch / 12.0)
        filters.append(f"rubberband=pitch={ratio:.8f}")

    gain = float(settings["gain_db"])
    if abs(gain) > 0.001:
        filters.append(f"volume={gain:.2f}dB")

    if settings.get("trim_silence"):
        filters.append(
            "silenceremove=start_periods=1:start_duration=0.08:start_threshold=-55dB"
        )
        filters.append("areverse")
        filters.append(
            "silenceremove=start_periods=1:start_duration=0.08:start_threshold=-55dB"
        )
        filters.append("areverse")

    if settings.get("normalize"):
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    if include_edge_silence:
        initial_ms = int(settings.get("initial_silence_ms", 0))
        final_ms = int(settings.get("final_silence_ms", 0))
        if initial_ms > 0:
            filters.append(f"adelay={initial_ms}:all=1")
        if final_ms > 0:
            filters.append(f"apad=pad_dur={final_ms / 1000.0:.3f}")

    return ",".join(filters)


def encode_output(path: Path) -> str:
    raw = path.read_bytes()
    if len(raw) > MAX_RESULT_BASE64_BYTES:
        raise ValueError(
            "O áudio final ficou grande demais para retorno em Base64. "
            "Divida o texto ou integre o worker ao Firebase Storage."
        )
    return base64.b64encode(raw).decode("ascii")



def get_whisper():
    global _WHISPER
    if WhisperModel is None:
        return None
    if _WHISPER is None:
        with _WHISPER_LOCK:
            if _WHISPER is None:
                compute_type = "float16" if DEVICE == "cuda" else "int8"
                _WHISPER = WhisperModel(
                    os.getenv("CTEC_WHISPER_MODEL", "small"),
                    device=DEVICE,
                    compute_type=compute_type,
                )
    return _WHISPER


def normalize_compare_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^\w\sáàâãéêíóôõúüç]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def transcription_similarity(expected: str, actual: str) -> float:
    a = normalize_compare_text(expected)
    b = normalize_compare_text(actual)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def transcribe_audio(path: Path) -> str:
    model = get_whisper()
    if model is None:
        return ""
    segments, _ = model.transcribe(
        str(path),
        language="pt",
        beam_size=3,
        vad_filter=True,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def analyze_reference_audio(path: Path) -> dict[str, Any]:
    waveform, sample_rate = torchaudio.load(str(path))
    mono = waveform.mean(dim=0)
    if mono.numel() == 0 or sample_rate <= 0:
        raise ValueError("A amostra de voz está vazia ou corrompida.")
    duration = float(mono.numel() / sample_rate)
    rms = float(torch.sqrt(torch.mean(mono ** 2) + 1e-9))
    peak = float(torch.max(torch.abs(mono)))
    frame = max(1, int(sample_rate * 0.03))
    usable = mono[: (mono.numel() // frame) * frame]
    if usable.numel() == 0:
        silence_ratio = 0.0
    else:
        energies = torch.sqrt(
            torch.mean(usable.reshape(-1, frame) ** 2, dim=1) + 1e-9
        )
        threshold = max(0.003, float(torch.median(energies)) * 0.22)
        silence_ratio = float((energies < threshold).float().mean())
    clipping_ratio = float((torch.abs(mono) >= 0.995).float().mean())
    quality = 100.0
    quality -= min(35.0, silence_ratio * 45.0)
    quality -= min(30.0, clipping_ratio * 5000.0)
    if duration < 12:
        quality -= 25
    if duration > 180:
        quality -= 5
    if rms < 0.008:
        quality -= 20
    if rms > 0.35:
        quality -= 10
    return {
        "durationSeconds": round(duration, 2),
        "sampleRate": sample_rate,
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "silenceRatio": round(silence_ratio, 4),
        "clippingRatio": round(clipping_ratio, 6),
        "qualityScore": round(clamp(quality, 0, 100), 1),
    }


def write_generated_candidate(
    model: ChatterboxMultilingualTTS,
    text: str,
    reference_path: Path,
    settings: dict[str, Any],
    destination: Path,
) -> float:
    audio = generate_chunk_with_retry(
        model,
        text,
        language_id="pt",
        reference_path=reference_path,
        settings=settings,
        chunk_index=1,
        total_chunks=1,
    )
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    raw_path = destination.with_name(f"{destination.stem}_raw.wav")
    torchaudio.save(str(raw_path), audio, model.sr)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path)]
    audio_filter = build_ffmpeg_filter(settings, include_edge_silence=False)
    if audio_filter:
        command += ["-filter:a", audio_filter]
    command += ["-c:a", "pcm_s16le", str(destination)]
    subprocess.run(command, check=True)
    processed, sample_rate = torchaudio.load(str(destination))
    return float(processed.shape[1] / sample_rate)


def calibrate(job: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    raw_calibration_text = str(
        data.get("calibration_text")
        or "Artigo quinto. Todos são iguais perante a lei, sem distinção de qualquer natureza. Parágrafo primeiro. As normas definidoras dos direitos e garantias fundamentais têm aplicação imediata."
    ).strip()
    dictionary = data.get("pronunciation_dictionary")
    calibration_text = prepare_text(
        raw_calibration_text,
        "law",
        dictionary if isinstance(dictionary, list) else [],
    )
    # A calibração compara uma amostra representativa, não o documento inteiro.
    # Isso mantém a prova A/B rápida, previsível e pequena o bastante para
    # retornar pelas Cloud Functions sem cortar o áudio.
    preview_text = calibration_text[:450].strip()
    if len(calibration_text) > 450:
        boundaries = list(re.finditer(r"[.!?;:](?:\s|$)", preview_text))
        if boundaries and boundaries[-1].end() >= 200:
            preview_text = preview_text[:boundaries[-1].end()].strip()
        else:
            preview_text = preview_text.rsplit(" ", 1)[0].strip()
    if len(preview_text) < 20:
        raise ValueError("O texto preparado ficou curto demais para calibrar a voz.")
    target_profile = str(data.get("target_profile") or "law_natural").strip()
    base = resolve_settings({"profile": target_profile})

    candidates = [
        {"id": "candidate_a", "name": "A", "exaggeration": 0.30, "cfg_weight": 0.68, "temperature": 0.46, "speed": 0.94, "stability": 0.86},
        {"id": "candidate_b", "name": "B", "exaggeration": 0.40, "cfg_weight": 0.60, "temperature": 0.54, "speed": 0.97, "stability": 0.78},
    ]

    with _GENERATION_LOCK, tempfile.TemporaryDirectory(prefix="ctec_calibration_") as tmp:
        root = Path(tmp)
        reference_source = save_reference_audio(data, root)
        reference_path, reference_metrics = prepare_reference_audio(
            reference_source,
            root,
        )
        model = get_model()
        results = []

        for index, candidate in enumerate(candidates, start=1):
            runpod.serverless.progress_update(
                job, f"Calibração automática: teste {index} de {len(candidates)}"
            )
            settings = dict(base)
            settings.update(candidate)
            wav_path = root / f"candidate_{candidate['name']}.wav"
            duration = write_generated_candidate(
                model, preview_text, reference_path, settings, wav_path
            )
            transcript = transcribe_audio(wav_path)
            similarity = transcription_similarity(preview_text, transcript)
            expected_duration = max(2.0, len(preview_text.split()) / 2.7)
            duration_ratio = duration / expected_duration
            rhythm_score = 1.0 - min(1.0, abs(duration_ratio - 1.0) / 0.55)
            completeness = similarity
            score = (
                completeness * 72.0
                + rhythm_score * 18.0
                + (reference_metrics["qualityScore"] / 100.0) * 10.0
            )
            preview_mp3 = root / f"candidate_{candidate['name']}.mp3"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
                    "-codec:a", "libmp3lame", "-b:a", "128k", str(preview_mp3),
                ],
                check=True,
            )
            preview_b64 = base64.b64encode(preview_mp3.read_bytes()).decode("ascii")
            results.append({
                "id": candidate["id"],
                "name": candidate["name"],
                "score": round(score, 1),
                "completeness": round(completeness * 100, 1),
                "rhythmStability": round(rhythm_score * 100, 1),
                "durationSeconds": round(duration, 2),
                "transcript": transcript,
                "expectedText": preview_text,
                "settings": public_settings(settings),
                "previewAudioBase64": preview_b64,
                "previewMimeType": "audio/mpeg",
            })

        best = max(results, key=lambda item: item["score"])
        best_settings = dict(best["settings"])
        best_settings.update({
            "stability": round(best["rhythmStability"] / 100.0, 3),
            "voiceFidelity": round(clamp(
                reference_metrics["qualityScore"] / 100.0, 0.55, 0.95
            ), 3),
            "cfgWeight": best_settings["cfg_weight"],
            "commaPauseMs": 250 if target_profile != "law_formal" else 300,
            "periodPauseMs": 520 if target_profile != "law_formal" else 620,
            "paragraphPauseMs": 760 if target_profile != "law_formal" else 900,
            "maxChunkCharacters": 500 if target_profile != "law_formal" else 430,
        })
        return {
            "status": "ok",
            "action": "calibrate",
            "score": best["score"],
            "completeness": best["completeness"],
            "rhythmStability": best["rhythmStability"],
            "recommendedSettings": best_settings,
            "preparedText": calibration_text,
            "referenceMetrics": reference_metrics,
            "bestCandidate": best["id"],
            "bestCandidateName": best["name"],
            "candidates": results,
        }


def capabilities() -> dict[str, Any]:
    return {
        "status": "ok",
        "worker": "CTEC Estúdio de Voz",
        "version": "5.0.0",
        "contract_version": WORKER_CONTRACT_VERSION,
        "device": DEVICE,
        "model": f"Chatterbox Multilingual {MODEL_VERSION}",
        "profiles": PROFILES,
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        "reference_inputs": ["voice_id", "reference_audio_url", "reference_audio_base64"],
        "output_formats": ["mp3", "wav"],
        "legal_normalization": True,
        "custom_pronunciation_dictionary": True,
        "automatic_calibration": True,
        "ab_candidates": True,
        "whisper_verification": WhisperModel is not None,
        "long_projects": True,
        "punctuation_pauses": True,
        "reference_audio_balancing": True,
    }



def clean_generation_chunk(value: str) -> str:
    value = str(value or "")
    value = value.replace("\u200b", " ").replace("\ufeff", " ")
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^[,;:.!?—–\-\s]+", "", value)
    value = re.sub(r"[—–\-\s]+$", "", value)
    if value and not re.search(r"[,;:.!?]$", value):
        value += "."
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    return value.strip()


def pause_for_chunk(
    chunk: str,
    paragraph_end: bool,
    settings: dict[str, Any],
) -> int:
    cleaned = chunk.rstrip()
    if cleaned.endswith(","):
        return int(settings["pause_comma_ms"])
    if cleaned.endswith((":", ";")):
        return int(settings["pause_colon_ms"])
    if paragraph_end:
        return int(settings["pause_paragraph_ms"])
    return int(settings["pause_sentence_ms"])


def is_valid_generation_chunk(value: str) -> bool:
    cleaned = clean_generation_chunk(value)
    letters_or_numbers = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", cleaned)
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", cleaned)
    return len(letters_or_numbers) >= 3 and len(words) >= 1


def merge_tiny_chunks(
    chunks: list[tuple[str, bool]],
    minimum_chars: int = 18,
) -> list[tuple[str, bool]]:
    merged: list[tuple[str, bool]] = []
    pending = ""

    for raw_chunk, paragraph_end in chunks:
        chunk = clean_generation_chunk(raw_chunk)
        if not is_valid_generation_chunk(chunk):
            continue

        if len(chunk) < minimum_chars:
            pending = f"{pending} {chunk}".strip()
            if paragraph_end and pending:
                if merged:
                    previous, _ = merged.pop()
                    merged.append((f"{previous} {pending}".strip(), True))
                else:
                    merged.append((pending, True))
                pending = ""
            continue

        if pending:
            chunk = f"{pending} {chunk}".strip()
            pending = ""

        merged.append((chunk, paragraph_end))

    if pending:
        if merged:
            previous, paragraph_end = merged.pop()
            merged.append((f"{previous} {pending}".strip(), paragraph_end))
        elif is_valid_generation_chunk(pending):
            merged.append((pending, True))

    return merged


def generate_chunk_with_retry(
    model: ChatterboxMultilingualTTS,
    chunk: str,
    *,
    language_id: str,
    reference_path: Path,
    settings: dict[str, Any],
    chunk_index: int,
    total_chunks: int,
) -> torch.Tensor:
    stability = float(settings.get("stability", 0.72))
    fidelity = float(settings.get("voice_fidelity", 0.78))
    effective_temperature = clamp(
        float(settings["temperature"]) * (1.18 - stability * 0.34),
        0.10,
        1.50,
    )
    effective_exaggeration = clamp(
        float(settings["exaggeration"]) * (1.10 - stability * 0.18),
        0.0,
        1.0,
    )
    effective_cfg = clamp(
        float(settings["cfg_weight"]) + (fidelity - 0.5) * 0.12,
        0.0,
        1.0,
    )
    attempts = [
        {
            "text": clean_generation_chunk(chunk),
            "temperature": effective_temperature,
            "exaggeration": effective_exaggeration,
            "cfg_weight": effective_cfg,
        },
        {
            "text": clean_generation_chunk(chunk).rstrip(".") + ".",
            "temperature": min(effective_temperature, 0.52),
            "exaggeration": min(effective_exaggeration, 0.40),
            "cfg_weight": max(effective_cfg, 0.60),
        },
    ]

    last_error: Exception | None = None
    for attempt_index, attempt in enumerate(attempts, start=1):
        candidate = attempt["text"]
        if not is_valid_generation_chunk(candidate):
            raise ValueError(
                f"Trecho {chunk_index}/{total_chunks} ficou vazio ou inválido "
                "depois da preparação jurídica."
            )

        try:
            print(
                f"[CTEC] Trecho {chunk_index}/{total_chunks}, "
                f"tentativa {attempt_index}: {candidate[:180]!r}",
                flush=True,
            )
            audio = model.generate(
                candidate,
                language_id=language_id,
                audio_prompt_path=str(reference_path),
                exaggeration=attempt["exaggeration"],
                cfg_weight=attempt["cfg_weight"],
                temperature=attempt["temperature"],
                repetition_penalty=float(settings["repetition_penalty"]),
                min_p=float(settings["min_p"]),
                top_p=float(settings["top_p"]),
            ).detach().cpu()

            if audio.numel() == 0:
                raise RuntimeError("O modelo devolveu um tensor de áudio vazio.")
            return audio
        except (IndexError, RuntimeError) as error:
            last_error = error
            print(
                f"[CTEC] Falha no trecho {chunk_index}/{total_chunks}, "
                f"tentativa {attempt_index}: {type(error).__name__}: {error}",
                flush=True,
            )

    raise RuntimeError(
        f"O Chatterbox falhou ao gerar o trecho {chunk_index}/{total_chunks}. "
        f"Trecho: {clean_generation_chunk(chunk)[:220]!r}. "
        f"Erro final: {last_error}"
    )



def upload_file_to_signed_url(
    file_path: Path,
    signed_url: str,
    content_type: str,
) -> None:
    parsed = urllib.parse.urlparse(signed_url)
    allowed_hosts = {
        "firebasestorage.googleapis.com",
        "storage.googleapis.com",
    }
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError("A URL de upload não pertence ao Firebase Storage.")
    request = urllib.request.Request(
        signed_url,
        data=file_path.read_bytes(),
        method="PUT",
        headers={
            "Content-Type": content_type,
            "Content-Length": str(file_path.stat().st_size),
        },
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        if response.status not in {200, 201}:
            raise RuntimeError(
                f"Upload do áudio final falhou com HTTP {response.status}."
            )


def generate_long_project(
    job: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    text = str(data.get("text") or "").strip()
    if len(text) < 100:
        raise ValueError("O projeto longo precisa ter pelo menos 100 caracteres.")
    if len(text) > 180000:
        raise ValueError("O projeto longo excede 180.000 caracteres.")

    upload_url = str(data.get("final_upload_url") or "").strip()
    if not upload_url:
        raise ValueError("URL assinada de upload final não foi enviada.")

    language_id = str(data.get("language_id") or "pt").strip().lower()
    settings = resolve_settings(data)
    custom_dictionary = data.get("pronunciation_dictionary")
    prepared = prepare_text(
        text,
        settings["text_mode"],
        custom_dictionary if isinstance(custom_dictionary, list) else [],
    )

    chunks = merge_tiny_chunks(split_text(
        prepared,
        int(settings["chunk_limit"]),
        preserve_complete_sentences=settings["preserve_complete_sentences"],
        split_by_legal_structure=settings["split_by_legal_structure"],
        context_margin_words=settings["chunk_overlap_words"],
    ))
    if not chunks:
        raise ValueError("Nenhum trecho válido foi produzido.")

    output_format = "mp3"
    mp3_bitrate = str(data.get("mp3_bitrate") or "160k")
    model = get_model()
    started = time.time()

    with _GENERATION_LOCK, tempfile.TemporaryDirectory(
        prefix="ctec_long_voice_"
    ) as tmp:
        root = Path(tmp)
        reference_source = save_reference_audio(data, root)
        reference_path, _ = prepare_reference_audio(reference_source, root)
        chunks_dir = root / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        concat_file = root / "concat.txt"
        final_path = root / "ctec-audio-longo.mp3"

        marker_list = []
        concat_lines = []
        cumulative_seconds = 0.0
        total = len(chunks)

        for index, (chunk, paragraph_end) in enumerate(chunks, start=1):
            percent = 8 + int((index - 1) / max(1, total) * 82)
            runpod.serverless.progress_update(
                job,
                json.dumps({
                    "stage": "generating",
                    "stageLabel": f"Gerando trecho {index} de {total}",
                    "progress": percent / 100,
                    "currentChunk": index,
                    "totalChunks": total,
                    "elapsedSeconds": int(time.time() - started),
                }),
            )

            audio = generate_chunk_with_retry(
                model,
                chunk,
                language_id=language_id,
                reference_path=reference_path,
                settings=settings,
                chunk_index=index,
                total_chunks=total,
            )
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)

            wav_path = chunks_dir / f"{index:05d}.wav"
            mp3_path = chunks_dir / f"{index:05d}.mp3"
            torchaudio.save(str(wav_path), audio, model.sr)

            pause_ms = (
                int(settings["final_silence_ms"])
                if index == total
                else pause_for_chunk(chunk, paragraph_end, settings)
            )
            filters = []
            standard_filter = build_ffmpeg_filter(
                settings,
                include_edge_silence=False,
            )
            if standard_filter:
                filters.append(standard_filter)
            if index == 1 and int(settings["initial_silence_ms"]) > 0:
                filters.append(
                    f"adelay={int(settings['initial_silence_ms'])}:all=1"
                )
            if pause_ms > 0:
                filters.append(f"apad=pad_dur={pause_ms / 1000.0:.3f}")
            command = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(wav_path),
            ]
            if filters:
                command += ["-af", ",".join(filters)]
            command += [
                "-codec:a", "libmp3lame", "-b:a", mp3_bitrate, str(mp3_path)
            ]
            subprocess.run(command, check=True)
            concat_lines.append(f"file '{mp3_path.as_posix()}'")

            duration = float(audio.shape[1] / model.sr) / float(settings["speed"])
            marker_list.append({
                "index": index,
                "startSeconds": round(cumulative_seconds, 2),
                "durationSeconds": round(duration, 2),
                "text": chunk[:500],
            })
            if index == 1:
                cumulative_seconds += int(settings["initial_silence_ms"]) / 1000.0
            cumulative_seconds += duration + pause_ms / 1000.0

        runpod.serverless.progress_update(
            job,
            json.dumps({
                "stage": "joining",
                "stageLabel": "Unindo todos os trechos",
                "progress": 0.93,
                "currentChunk": total,
                "totalChunks": total,
                "elapsedSeconds": int(time.time() - started),
            }),
        )

        concat_file.write_text("\n".join(concat_lines), encoding="utf-8")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_file),
                "-codec:a", "copy",
                str(final_path),
            ],
            check=True,
        )

        runpod.serverless.progress_update(
            job,
            json.dumps({
                "stage": "uploading",
                "stageLabel": "Salvando o áudio final",
                "progress": 0.98,
                "currentChunk": total,
                "totalChunks": total,
                "elapsedSeconds": int(time.time() - started),
            }),
        )
        upload_file_to_signed_url(
            final_path,
            upload_url,
            "audio/mpeg",
        )

        return {
            "status": "ok",
            "action": "generate_long_project",
            "file_name": final_path.name,
            "mime_type": "audio/mpeg",
            "size_bytes": final_path.stat().st_size,
            "duration_seconds": round(cumulative_seconds, 2),
            "chunks": total,
            "markers": marker_list,
            "settings": public_settings(settings),
            "elapsed_seconds": int(time.time() - started),
        }


def generate(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("input") or {}
    action = str(data.get("action") or "generate").strip().lower()

    if action in {"capabilities", "config", "health"}:
        return capabilities()

    if action == "calibrate":
        return calibrate(job, data)

    if action == "generate_long_project":
        return generate_long_project(job, data)

    if action == "normalize_legal_text":
        original = str(data.get("text") or "")
        dictionary = data.get("pronunciation_dictionary")
        prepared = normalize_law_text(
            original,
            dictionary if isinstance(dictionary, list) else [],
        )
        return {
            "status": "ok",
            "action": "normalize_legal_text",
            "prepared_text": prepared,
            "original_characters": len(original),
            "prepared_characters": len(prepared),
        }

    text = str(data.get("text") or "").strip()
    if len(text) < 3:
        raise ValueError("O texto precisa ter pelo menos 3 caracteres.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"O texto excede o limite de {MAX_TEXT_CHARS} caracteres.")

    language_id = str(data.get("language_id") or "pt").strip().lower()
    if language_id not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Idioma não suportado: {language_id}")

    settings = resolve_settings(data)
    custom_dictionary = data.get("pronunciation_dictionary")
    text = prepare_text(
        text,
        settings["text_mode"],
        custom_dictionary if isinstance(custom_dictionary, list) else [],
    )

    if to_bool(data.get("preview"), False):
        preview_chars = int(clamp(float(data.get("preview_chars", 450)), 80, 1500))
        text = text[:preview_chars].rsplit(" ", 1)[0].strip() or text[:preview_chars]

    chunks = split_text(
        text,
        int(settings["chunk_limit"]),
        preserve_complete_sentences=settings["preserve_complete_sentences"],
        split_by_legal_structure=settings["split_by_legal_structure"],
        context_margin_words=settings["chunk_overlap_words"],
    )
    chunks = merge_tiny_chunks(chunks)
    if not chunks:
        raise ValueError(
            "O texto não gerou nenhum trecho válido depois da preparação jurídica."
        )

    output_format = str(data.get("output_format") or "mp3").strip().lower()
    if output_format not in {"mp3", "wav"}:
        raise ValueError("output_format deve ser mp3 ou wav.")
    mp3_bitrate = str(data.get("mp3_bitrate") or "192k").strip().lower()
    if mp3_bitrate not in {"96k", "128k", "160k", "192k", "256k", "320k"}:
        mp3_bitrate = "192k"

    model = get_model()

    with _GENERATION_LOCK, tempfile.TemporaryDirectory(prefix="ctec_voice_") as tmp:
        root = Path(tmp)
        reference_source = save_reference_audio(data, root)
        reference_path, reference_metrics = prepare_reference_audio(
            reference_source,
            root,
        )
        raw_wav = root / "raw.wav"
        final_path = root / f"ctec-voz-neural.{output_format}"

        segments: list[torch.Tensor] = []
        total = len(chunks)

        for index, (chunk, paragraph_end) in enumerate(chunks, start=1):
            runpod.serverless.progress_update(job, f"Gerando trecho {index} de {total}")
            print(f"[CTEC] Gerando trecho {index}/{total}", flush=True)

            audio = generate_chunk_with_retry(
                model,
                chunk,
                language_id=language_id,
                reference_path=reference_path,
                settings=settings,
                chunk_index=index,
                total_chunks=total,
            )

            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            segments.append(audio)

            if index < total:
                pause_ms = pause_for_chunk(chunk, paragraph_end, settings)
                if pause_ms > 0:
                    segments.append(torch.zeros(
                        1,
                        int(model.sr * pause_ms / 1000.0),
                        dtype=torch.float32,
                    ))

        combined = torch.cat(segments, dim=1)
        torchaudio.save(str(raw_wav), combined, model.sr)

        command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_wav)]
        audio_filter = build_ffmpeg_filter(settings)
        if audio_filter:
            command += ["-filter:a", audio_filter]

        if output_format == "mp3":
            command += ["-codec:a", "libmp3lame", "-b:a", mp3_bitrate, str(final_path)]
            mime_type = "audio/mpeg"
        else:
            command += ["-codec:a", "pcm_s16le", str(final_path)]
            mime_type = "audio/wav"

        subprocess.run(command, check=True)
        encoded = encode_output(final_path)
        recognized_text = ""
        if to_bool(data.get("verify_transcript"), True):
            recognized_text = transcribe_audio(final_path)

        duration_seconds = round(
            combined.shape[1] / model.sr / float(settings["speed"])
            + int(settings["initial_silence_ms"]) / 1000.0
            + int(settings["final_silence_ms"]) / 1000.0,
            2,
        )

        return {
            "status": "ok",
            "action": "generate",
            "contract_version": WORKER_CONTRACT_VERSION,
            "audio_base64": encoded,
            "mime_type": mime_type,
            "file_name": final_path.name,
            "sample_rate": model.sr,
            "duration_seconds_estimate": duration_seconds,
            "device": DEVICE,
            "model": MODEL_VERSION,
            "chunks": total,
            "voice_id": str(data.get("voice_id") or "") or None,
            "settings": public_settings(settings),
            "prepared_text": text,
            "reference_metrics": reference_metrics,
            "recognized_text": recognized_text,
            "transcription_similarity": round(
                transcription_similarity(text, recognized_text) * 100,
                1,
            ) if recognized_text else None,
        }


if __name__ == "__main__":
    print("[CTEC] Iniciando CTEC Estúdio de Voz Worker 5.0...", flush=True)
    print(f"[CTEC] Device: {DEVICE} | Modelo: {MODEL_VERSION}", flush=True)
    runpod.serverless.start({"handler": generate})
