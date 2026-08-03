import base64
import difflib
import inspect
import json
import math
import os
import random
import re
import subprocess
import tempfile
import threading
import time
import unicodedata
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import runpod
import torch
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from num2words import num2words

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


# ============================================================
# CONFIGURAÇÃO GLOBAL
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_VERSION = (os.getenv("CTEC_CHATTERBOX_MODEL", "v3").strip().lower() or "v3")
if MODEL_VERSION not in {"v2", "v3"}:
    raise RuntimeError(
        f"CTEC_CHATTERBOX_MODEL inválido: {MODEL_VERSION!r}. Use somente v2 ou v3."
    )

MAX_TEXT_CHARS = int(os.getenv("CTEC_MAX_TEXT_CHARS", "180000"))
MAX_REFERENCE_BYTES = int(
    os.getenv("CTEC_MAX_REFERENCE_BYTES", str(64 * 1024 * 1024))
)
MAX_RESULT_BASE64_BYTES = int(
    os.getenv("CTEC_MAX_RESULT_BASE64_BYTES", str(14 * 1024 * 1024))
)
WHISPER_MODEL_NAME = os.getenv("CTEC_WHISPER_MODEL", "small").strip() or "small"
DEFAULT_SEED = int(os.getenv("CTEC_SEED_BASE", "20260803"))
DEFAULT_SIMILARITY = float(os.getenv("CTEC_MIN_SIMILARITY", "0.92"))
DEFAULT_WORD_COVERAGE = float(os.getenv("CTEC_MIN_WORD_COVERAGE", "0.94"))
DEFAULT_MAX_ATTEMPTS = int(os.getenv("CTEC_MAX_CHUNK_ATTEMPTS", "3"))
DEFAULT_CHUNK_TARGET = int(os.getenv("CTEC_CHUNK_TARGET", "250"))
DEFAULT_CHUNK_MAX = int(os.getenv("CTEC_CHUNK_MAX", "300"))

_MODEL: ChatterboxMultilingualTTS | None = None
_MODEL_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()
_WHISPER = None
_WHISPER_LOCK = threading.Lock()


# ============================================================
# PERFIS
# ============================================================

CONTROL_PROFILE = {
    "speed": 1.0,
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8,
    "pause_continuation_ms": 100,
    "pause_sentence_ms": 220,
    "pause_legal_item_ms": 280,
    "pause_paragraph_ms": 430,
    "pause_article_ms": 620,
    "pitch_semitones": 0.0,
    "gain_db": 0.0,
    "normalize": True,
    "trim_silence": False,
    "text_mode": "law",
    "chunk_target": 250,
    "chunk_limit": 300,
    "seed": DEFAULT_SEED,
    "verify_fidelity": True,
    "min_similarity": DEFAULT_SIMILARITY,
    "min_word_coverage": DEFAULT_WORD_COVERAGE,
    "max_chunk_attempts": DEFAULT_MAX_ATTEMPTS,
}

PROFILES: dict[str, dict[str, Any]] = {
    "law_control": dict(CONTROL_PROFILE),
    "law_natural": {
        **CONTROL_PROFILE,
        "speed": 0.98,
        "exaggeration": 0.48,
        "cfg_weight": 0.50,
        "temperature": 0.78,
    },
    "law_formal": {
        **CONTROL_PROFILE,
        "speed": 0.95,
        "exaggeration": 0.38,
        "cfg_weight": 0.54,
        "temperature": 0.72,
        "pause_sentence_ms": 240,
        "pause_legal_item_ms": 310,
        "pause_paragraph_ms": 470,
        "pause_article_ms": 680,
    },
    "professor": {
        **CONTROL_PROFILE,
        "speed": 0.98,
        "exaggeration": 0.55,
        "cfg_weight": 0.46,
        "temperature": 0.80,
        "text_mode": "general",
    },
    "podcast_calm": {
        **CONTROL_PROFILE,
        "speed": 0.94,
        "exaggeration": 0.43,
        "cfg_weight": 0.54,
        "temperature": 0.76,
        "text_mode": "general",
    },
    "podcast_energetic": {
        **CONTROL_PROFILE,
        "speed": 1.04,
        "exaggeration": 0.66,
        "cfg_weight": 0.40,
        "temperature": 0.84,
        "pause_sentence_ms": 180,
        "pause_paragraph_ms": 360,
        "text_mode": "general",
    },
    "summary_fast": {
        **CONTROL_PROFILE,
        "speed": 1.12,
        "exaggeration": 0.44,
        "cfg_weight": 0.44,
        "temperature": 0.76,
        "pause_sentence_ms": 150,
        "pause_paragraph_ms": 280,
        "text_mode": "general",
    },
    "question_explained": {
        **CONTROL_PROFILE,
        "speed": 0.97,
        "exaggeration": 0.58,
        "cfg_weight": 0.46,
        "temperature": 0.80,
        "text_mode": "general",
    },
    "institutional": {
        **CONTROL_PROFILE,
        "speed": 0.96,
        "exaggeration": 0.38,
        "cfg_weight": 0.56,
        "temperature": 0.72,
        "text_mode": "general",
    },
}

SUPPORTED_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
    "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}

ALIASES = {
    "cfgWeight": "cfg_weight",
    "maxChunkCharacters": "chunk_limit",
    "chunkLimit": "chunk_limit",
    "chunkTarget": "chunk_target",
    "commaPauseMs": "pause_continuation_ms",
    "periodPauseMs": "pause_sentence_ms",
    "legalItemPauseMs": "pause_legal_item_ms",
    "paragraphPauseMs": "pause_paragraph_ms",
    "articlePauseMs": "pause_article_ms",
    "pitchSemitones": "pitch_semitones",
    "gainDb": "gain_db",
    "trimSilence": "trim_silence",
    "verifyFidelity": "verify_fidelity",
    "minSimilarity": "min_similarity",
    "minWordCoverage": "min_word_coverage",
    "maxChunkAttempts": "max_chunk_attempts",
    "seedBase": "seed",
}


# ============================================================
# MODELOS DE DADOS
# ============================================================

@dataclass
class PreparedChunk:
    index: int
    original_text: str
    spoken_text: str
    boundary: str
    source_start: int
    source_end: int


@dataclass
class VerifiedChunk:
    index: int
    original_text: str
    spoken_text: str
    expected_text: str
    transcript: str
    similarity: float
    word_coverage: float
    missing_words: list[str]
    extra_words: list[str]
    attempts: int
    seed: int
    duration_seconds: float
    pause_after_seconds: float
    status: str
    wav_name: str
    start_seconds: float = 0.0


# ============================================================
# UTILITÁRIOS
# ============================================================

def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "sim", "on"}


def safe_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def progress(job: dict[str, Any], **payload: Any) -> None:
    try:
        runpod.serverless.progress_update(job, json.dumps(payload, ensure_ascii=False))
    except Exception as exc:
        print(f"[CTEC] Não foi possível atualizar progresso: {exc}", flush=True)


# ============================================================
# CARREGAMENTO REAL DO CHATTERBOX V2/V3
# ============================================================

def get_model() -> ChatterboxMultilingualTTS:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        loader = ChatterboxMultilingualTTS.from_pretrained
        signature = inspect.signature(loader)
        supports_t3_model = "t3_model" in signature.parameters

        print(
            f"[CTEC] Solicitado Chatterbox Multilingual {MODEL_VERSION}; "
            f"device={DEVICE}; from_pretrained={signature}",
            flush=True,
        )

        if MODEL_VERSION == "v3" and not supports_t3_model:
            raise RuntimeError(
                "A versão instalada de chatterbox-tts não aceita t3_model. "
                "Não é seguro anunciar V3. Atualize para uma versão compatível "
                "ou selecione explicitamente CTEC_CHATTERBOX_MODEL=v2."
            )

        kwargs: dict[str, Any] = {"device": DEVICE}
        if supports_t3_model:
            kwargs["t3_model"] = MODEL_VERSION

        try:
            _MODEL = loader(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao carregar Chatterbox {MODEL_VERSION} com parâmetros "
                f"{sorted(kwargs)}: {type(exc).__name__}: {exc}"
            ) from exc

        setattr(_MODEL, "_ctec_requested_model_version", MODEL_VERSION)
        setattr(_MODEL, "_ctec_t3_model_argument_used", supports_t3_model)
        print(
            f"[CTEC] Modelo efetivamente carregado: {MODEL_VERSION}; "
            f"t3_model enviado={supports_t3_model}; sample_rate={_MODEL.sr}",
            flush=True,
        )
        return _MODEL


# ============================================================
# NORMALIZAÇÃO JURÍDICA SEM REORDENAR OU APAGAR CONTEÚDO
# ============================================================

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
        return num2words(
            value,
            lang="pt_BR",
            to="ordinal" if ordinal else "cardinal",
        )
    except Exception:
        return str(value)


def apply_pronunciation_dictionary(
    text: str,
    custom_dictionary: list[dict[str, Any]] | None,
) -> str:
    result = text
    items = sorted(
        custom_dictionary or [],
        key=lambda item: len(str(item.get("source") or "")),
        reverse=True,
    )
    for item in items:
        source = str(item.get("source") or "").strip()
        spoken = str(item.get("spoken") or "").strip()
        if source and spoken:
            result = re.sub(re.escape(source), spoken, result, flags=re.IGNORECASE)
    return result


def normalize_pdf_line_breaks(text: str) -> str:
    """Quebra simples vira espaço; duas ou mais preservam parágrafo real."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n[ \t]*\n+", text)
    normalized: list[str] = []
    structural = re.compile(
        r"^\s*(?:"
        r"(?:Art(?:s)?\.\s*\d+)|"
        r"(?:§\s*(?:\d+|único))|"
        r"(?:Parágrafo\s+único)|"
        r"(?:[IVXLCDM]{1,12}\s*[—–-])|"
        r"(?:[a-z]\))|"
        r"(?:TÍTULO|CAPÍTULO|SEÇÃO|SUBSEÇÃO|LIVRO)\b"
        r")",
        flags=re.IGNORECASE,
    )

    for paragraph in paragraphs:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in paragraph.split("\n")]
        lines = [line for line in lines if line]
        if not lines:
            continue
        merged = lines[0]
        for line in lines[1:]:
            # Mantém a unidade estrutural em nova linha para o parser,
            # mas não a converte automaticamente em parágrafo longo.
            separator = "\n" if structural.match(line) else " "
            merged = f"{merged}{separator}{line}"
        normalized.append(merged.strip())
    return "\n\n".join(normalized).strip()


def normalize_law_text(
    text: str,
    custom_dictionary: list[dict[str, Any]] | None = None,
) -> str:
    original = text
    text = normalize_pdf_line_breaks(text)
    text = apply_pronunciation_dictionary(text, custom_dictionary)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Estruturas jurídicas.
    text = re.sub(r"§\s*único", "Parágrafo único", text, flags=re.IGNORECASE)
    text = re.sub(
        r"§\s*(\d+)\s*[º°]?",
        lambda m: f"Parágrafo {number_words(int(m.group(1)), True)}",
        text,
    )
    text = re.sub(
        r"\bArts?\.\s*(\d+)\s*[º°]?",
        lambda m: (
            f"Artigo {number_words(int(m.group(1)), True)}"
            if int(m.group(1)) <= 9
            else f"Artigo {number_words(int(m.group(1)))}"
        ),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bArtigo\s+(\d+)\s*[º°]?",
        lambda m: (
            f"Artigo {number_words(int(m.group(1)), True)}"
            if int(m.group(1)) <= 9
            else f"Artigo {number_words(int(m.group(1)))}"
        ),
        text,
        flags=re.IGNORECASE,
    )

    # Inciso estrutural: remove apenas o travessão separador, sem criar ponto
    # artificial no conteúdo.
    text = re.sub(
        r"(?m)^\s*([IVXLCDM]{1,12})\s*[—–-]\s*",
        lambda m: f"Inciso {number_words(roman_to_int(m.group(1)))}: ",
        text,
    )
    text = re.sub(
        r"\binciso\s+([IVXLCDM]{1,12})\b",
        lambda m: f"inciso {number_words(roman_to_int(m.group(1)))}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^\s*([a-z])\)\s*",
        lambda m: f"Alínea {m.group(1)}: ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^\s*(\d+)\.\s+",
        lambda m: f"Item {number_words(int(m.group(1)))}: ",
        text,
    )

    fixed = [
        (r"\bcaput\b", "cáput"),
        (r"\bn[.º°]\s*", "número "),
        (r"\bc/c\b", "combinado com"),
        (r"\bCF/88\b", "Constituição Federal de mil novecentos e oitenta e oito"),
        (
            r"\bCRFB/88\b",
            "Constituição da República Federativa do Brasil de mil novecentos e oitenta e oito",
        ),
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
        lambda m: (
            "Lei número "
            + number_words(int((m.group(1) or "") + (m.group(2) or "")))
            + ", de "
            + number_words(
                int(m.group(3))
                if len(m.group(3)) == 4
                else 2000 + int(m.group(3))
            )
        ),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(\d+(?:[.,]\d+)?)%",
        lambda m: (
            f"{number_words(int(float(m.group(1).replace(',', '.'))))} por cento"
        ),
        text,
    )

    # Preserva ; : e travessões não estruturais.
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if original.strip() and not text.strip():
        raise RuntimeError("A normalização jurídica apagou todo o texto.")
    return text


def prepare_text(
    text: str,
    mode: str,
    custom_dictionary: list[dict[str, Any]] | None = None,
) -> str:
    normalized = normalize_pdf_line_breaks(text)
    if mode == "law":
        normalized = normalize_law_text(normalized, custom_dictionary)
    else:
        normalized = apply_pronunciation_dictionary(
            normalized,
            custom_dictionary,
        )
    return normalized.strip()


# ============================================================
# DIVISÃO HIERÁRQUICA E AUDITÁVEL
# ============================================================

STRUCTURE_RE = re.compile(
    r"^(?:"
    r"Artigo\b|Parágrafo\b|Inciso\b|Alínea\b|Item\b|"
    r"TÍTULO\b|CAPÍTULO\b|SEÇÃO\b|SUBSEÇÃO\b|LIVRO\b"
    r")",
    flags=re.IGNORECASE,
)


def boundary_kind(text: str, paragraph_end: bool) -> str:
    stripped = text.strip()
    if re.match(r"^(?:TÍTULO|CAPÍTULO|SEÇÃO|SUBSEÇÃO|LIVRO|Artigo)\b", stripped, re.I):
        return "article"
    if re.match(r"^(?:Parágrafo|Inciso|Alínea|Item)\b", stripped, re.I):
        return "legal_item"
    if paragraph_end:
        return "paragraph"
    if re.search(r"[.!?][\"')\]]*$", stripped):
        return "sentence"
    return "continuation"


def _safe_split_position(value: str, maximum: int, target: int) -> int:
    if len(value) <= maximum:
        return len(value)

    candidates: list[tuple[int, int]] = []
    priorities = [
        (r"\n\n+", 6),
        (r"(?<=[.!?])\s+", 5),
        (r"(?<=;)\s+", 4),
        (r"(?<=:)\s+", 3),
        (r"(?<=,)\s+", 2),
        (r"\s+", 1),
    ]
    for pattern, priority in priorities:
        for match in re.finditer(pattern, value[: maximum + 1]):
            pos = match.end()
            if pos < 40:
                continue
            distance = abs(pos - target)
            candidates.append((priority * 10000 - distance, pos))
        if candidates:
            best_priority = max(score // 10000 for score, _ in candidates)
            same = [(score, pos) for score, pos in candidates if score // 10000 == best_priority]
            return max(same)[1]

    # Último recurso, nunca corta palavra.
    fallback = value.rfind(" ", 0, maximum + 1)
    if fallback <= 0:
        raise RuntimeError(
            "Não foi possível dividir um bloco sem cortar uma palavra ou referência."
        )
    return fallback + 1


def split_text_audited(
    original_text: str,
    prepared_text: str,
    target: int,
    maximum: int,
) -> list[PreparedChunk]:
    target = int(clamp(target, 180, maximum))
    maximum = int(clamp(maximum, 220, 300))

    # Parágrafos reais são apenas duas ou mais quebras.
    paragraphs = [
        p.strip()
        for p in re.split(r"\n[ \t]*\n+", prepared_text)
        if p.strip()
    ]
    chunks: list[PreparedChunk] = []
    source_cursor = 0
    index = 1

    for paragraph_idx, paragraph in enumerate(paragraphs):
        # Linhas estruturais são unidades lógicas; linhas comuns já foram
        # reagrupadas por normalize_pdf_line_breaks.
        units = [u.strip() for u in paragraph.split("\n") if u.strip()]
        if not units:
            continue

        for unit_idx, unit in enumerate(units):
            remaining = unit
            while remaining:
                split_at = _safe_split_position(remaining, maximum, target)
                piece = remaining[:split_at].strip()
                remaining = remaining[split_at:].strip()

                if not piece:
                    raise RuntimeError(
                        f"Divisão produziu bloco vazio no índice {index}."
                    )
                if len(piece) > maximum:
                    raise RuntimeError(
                        f"Bloco {index} excedeu o limite máximo: {len(piece)} > {maximum}."
                    )

                paragraph_end = (
                    paragraph_idx == len(paragraphs) - 1
                    or (unit_idx == len(units) - 1 and not remaining)
                )
                boundary = boundary_kind(piece, paragraph_end)

                # Mapeamento aproximado e monotônico para auditoria.
                lookup = re.sub(r"\s+", " ", piece).strip()
                haystack = re.sub(r"\s+", " ", original_text[source_cursor:])
                found = haystack.lower().find(lookup[: min(50, len(lookup))].lower())
                start = source_cursor if found < 0 else source_cursor + found
                end = min(len(original_text), start + len(piece))
                source_cursor = max(source_cursor, end)

                chunks.append(
                    PreparedChunk(
                        index=index,
                        original_text=piece,
                        spoken_text=piece,
                        boundary=boundary,
                        source_start=start,
                        source_end=end,
                    )
                )
                index += 1

    if not chunks:
        raise RuntimeError("Nenhum bloco foi preparado.")

    reconstructed = " ".join(c.spoken_text for c in chunks)
    prepared_tokens = comparison_tokens(prepared_text)
    reconstructed_tokens = comparison_tokens(reconstructed)
    coverage = multiset_coverage(prepared_tokens, reconstructed_tokens)
    if coverage < 0.999:
        raise RuntimeError(
            f"A divisão perdeu conteúdo: cobertura={coverage:.4f}. "
            "O projeto foi interrompido."
        )
    return chunks


# ============================================================
# ÁUDIO DE REFERÊNCIA CANÔNICO
# ============================================================

def download_url(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CTEC-Voice-Worker/5.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_REFERENCE_BYTES:
            raise ValueError("O áudio de referência excede o limite permitido.")
        data = response.read(MAX_REFERENCE_BYTES + 1)
    if len(data) > MAX_REFERENCE_BYTES:
        raise ValueError("O áudio de referência excede o limite permitido.")
    destination.write_bytes(data)


def receive_reference_audio(data: dict[str, Any], root: Path) -> Path:
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
            suffix = ".bin"
        path = root / f"reference_original{suffix}"
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
        path = root / f"reference_original{suffix_map.get(mime, '.bin')}"
        cleaned = re.sub(r"^data:audio/[^;]+;base64,", "", b64)
        raw = base64.b64decode(cleaned, validate=True)
        if len(raw) > MAX_REFERENCE_BYTES:
            raise ValueError("O áudio de referência excede o limite permitido.")
        path.write_bytes(raw)
        return path

    raise ValueError(
        "Informe voice_id, reference_audio_url ou reference_audio_base64."
    )


def canonicalize_reference(
    source: Path,
    root: Path,
    target_sample_rate: int,
) -> Path:
    destination = root / "reference_canonical.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(source),
            "-ac", "1",
            "-ar", str(target_sample_rate),
            "-c:a", "pcm_s16le",
            str(destination),
        ],
        check=True,
    )
    if not destination.exists() or destination.stat().st_size < 1024:
        raise RuntimeError("A conversão da referência produziu arquivo inválido.")
    return destination


def analyze_audio_tensor(
    waveform: torch.Tensor,
    sample_rate: int,
) -> dict[str, Any]:
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.float().cpu()
    if waveform.numel() == 0:
        raise RuntimeError("Áudio vazio.")
    finite = bool(torch.isfinite(waveform).all())
    peak = float(torch.max(torch.abs(waveform)))
    rms = float(torch.sqrt(torch.mean(waveform ** 2) + 1e-12))
    duration = float(waveform.numel() / sample_rate)
    clipping = float((torch.abs(waveform) >= 0.995).float().mean())
    dc_offset = float(torch.mean(waveform))

    frame = max(1, int(sample_rate * 0.03))
    usable = waveform[: (waveform.numel() // frame) * frame]
    if usable.numel():
        energies = torch.sqrt(
            torch.mean(usable.reshape(-1, frame) ** 2, dim=1) + 1e-12
        )
        silence_ratio = float((energies < 0.003).float().mean())
    else:
        silence_ratio = 0.0

    return {
        "durationSeconds": round(duration, 3),
        "sampleRate": sample_rate,
        "channels": 1,
        "finite": finite,
        "peak": round(peak, 7),
        "rms": round(rms, 7),
        "clippingRatio": round(clipping, 7),
        "silenceRatio": round(silence_ratio, 5),
        "dcOffset": round(dc_offset, 7),
    }


def analyze_reference_audio(path: Path) -> dict[str, Any]:
    waveform, sample_rate = torchaudio.load(str(path))
    channels = int(waveform.shape[0])
    mono = waveform.mean(dim=0, keepdim=True)
    metrics = analyze_audio_tensor(mono, sample_rate)
    metrics["originalChannels"] = channels
    metrics["referenceQuality"] = round(
        clamp(
            100
            - min(25, metrics["clippingRatio"] * 5000)
            - (20 if metrics["durationSeconds"] < 6 else 0)
            - (10 if metrics["peak"] < 0.02 else 0),
            0,
            100,
        ),
        1,
    )
    return metrics


# ============================================================
# CONFIGURAÇÕES E ALIASES
# ============================================================

def resolve_settings(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    for old, new in ALIASES.items():
        if old in normalized and new not in normalized:
            normalized[new] = normalized[old]

    profile_name = str(
        normalized.get("profile") or "law_natural"
    ).strip().lower()
    base = dict(PROFILES.get(profile_name, PROFILES["law_natural"]))

    numeric_limits = {
        "speed": (0.70, 1.35),
        "exaggeration": (0.0, 1.0),
        "cfg_weight": (0.0, 1.0),
        "temperature": (0.1, 1.5),
        "pause_continuation_ms": (0, 800),
        "pause_sentence_ms": (80, 700),
        "pause_legal_item_ms": (100, 900),
        "pause_paragraph_ms": (150, 1200),
        "pause_article_ms": (200, 1800),
        "pitch_semitones": (-6.0, 6.0),
        "gain_db": (-12.0, 12.0),
        "chunk_target": (180, 280),
        "chunk_limit": (220, 300),
        "seed": (0, 2_147_483_647),
        "min_similarity": (0.75, 1.0),
        "min_word_coverage": (0.75, 1.0),
        "max_chunk_attempts": (1, 5),
    }
    for key, (minimum, maximum) in numeric_limits.items():
        if key in normalized:
            base[key] = clamp(
                safe_float(normalized[key], float(base.get(key, minimum))),
                minimum,
                maximum,
            )

    for key in ("normalize", "trim_silence", "verify_fidelity"):
        if key in normalized:
            base[key] = to_bool(normalized[key], bool(base.get(key, False)))

    if "text_mode" in normalized:
        base["text_mode"] = str(normalized["text_mode"]).strip().lower()

    base["profile"] = profile_name
    for key in (
        "pause_continuation_ms", "pause_sentence_ms",
        "pause_legal_item_ms", "pause_paragraph_ms", "pause_article_ms",
        "chunk_target", "chunk_limit", "seed", "max_chunk_attempts",
    ):
        base[key] = int(base[key])

    if base["chunk_target"] > base["chunk_limit"]:
        base["chunk_target"] = base["chunk_limit"] - 20
    return base


# ============================================================
# WHISPER E FIDELIDADE
# ============================================================

def get_whisper(required: bool = False):
    global _WHISPER
    if WhisperModel is None:
        if required:
            raise RuntimeError(
                "Faster Whisper não está disponível, mas a verificação de "
                "fidelidade foi exigida."
            )
        return None

    if _WHISPER is None:
        with _WHISPER_LOCK:
            if _WHISPER is None:
                compute_type = "float16" if DEVICE == "cuda" else "int8"
                print(
                    f"[CTEC] Carregando Whisper {WHISPER_MODEL_NAME} "
                    f"em {DEVICE}/{compute_type}...",
                    flush=True,
                )
                _WHISPER = WhisperModel(
                    WHISPER_MODEL_NAME,
                    device=DEVICE,
                    compute_type=compute_type,
                )
    return _WHISPER


def strip_accents(value: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", value)
        if unicodedata.category(c) != "Mn"
    )


def normalize_compare_text(value: str) -> str:
    value = strip_accents(value.lower())
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def comparison_tokens(value: str) -> list[str]:
    return normalize_compare_text(value).split()


def token_counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def multiset_coverage(expected: list[str], actual: list[str]) -> float:
    if not expected:
        return 1.0 if not actual else 0.0
    e = token_counts(expected)
    a = token_counts(actual)
    matched = sum(min(count, a.get(word, 0)) for word, count in e.items())
    return matched / len(expected)


def compare_transcript(expected: str, actual: str) -> dict[str, Any]:
    expected_norm = normalize_compare_text(expected)
    actual_norm = normalize_compare_text(actual)
    expected_tokens = expected_norm.split()
    actual_tokens = actual_norm.split()

    similarity = (
        difflib.SequenceMatcher(None, expected_norm, actual_norm).ratio()
        if expected_norm and actual_norm
        else 0.0
    )
    coverage = multiset_coverage(expected_tokens, actual_tokens)

    e_counts = token_counts(expected_tokens)
    a_counts = token_counts(actual_tokens)
    missing: list[str] = []
    extra: list[str] = []
    for word, count in e_counts.items():
        missing.extend([word] * max(0, count - a_counts.get(word, 0)))
    for word, count in a_counts.items():
        extra.extend([word] * max(0, count - e_counts.get(word, 0)))

    return {
        "similarity": similarity,
        "word_coverage": coverage,
        "missing_words": missing[:50],
        "extra_words": extra[:50],
    }


def transcribe_audio(path: Path, vad_filter: bool = False) -> str:
    model = get_whisper(required=True)
    segments, _ = model.transcribe(
        str(path),
        language="pt",
        beam_size=4,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


# ============================================================
# GERAÇÃO, VALIDAÇÃO E RETENTATIVAS
# ============================================================

def supported_generate_kwargs(model: Any, settings: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(model.generate)
    official = {
        "repetition_penalty": 1.2,
        "min_p": 0.05,
        "top_p": 1.0,
    }
    kwargs: dict[str, Any] = {
        "exaggeration": float(settings["exaggeration"]),
        "cfg_weight": float(settings["cfg_weight"]),
        "temperature": float(settings["temperature"]),
    }
    for key, value in official.items():
        if key in signature.parameters:
            kwargs[key] = value
    return kwargs


def validate_waveform(
    audio: torch.Tensor,
    sample_rate: int,
    spoken_text: str,
) -> dict[str, Any]:
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.numel() == 0:
        raise RuntimeError("O modelo devolveu tensor vazio.")
    metrics = analyze_audio_tensor(audio, sample_rate)
    if not metrics["finite"]:
        raise RuntimeError("O áudio contém NaN ou Inf.")
    if metrics["peak"] <= 0.0005 or metrics["rms"] <= 0.00005:
        raise RuntimeError("O áudio está vazio ou praticamente silencioso.")
    if metrics["peak"] > 1.5:
        raise RuntimeError("O áudio possui pico inválido.")
    expected_minimum = max(0.35, len(spoken_text.split()) * 0.12)
    if metrics["durationSeconds"] < expected_minimum:
        raise RuntimeError(
            f"Duração incompatível: {metrics['durationSeconds']:.2f}s; "
            f"mínimo esperado {expected_minimum:.2f}s."
        )
    return metrics


def generate_verified_chunk(
    model: ChatterboxMultilingualTTS,
    chunk: PreparedChunk,
    reference_path: Path,
    language_id: str,
    settings: dict[str, Any],
    destination_dir: Path,
) -> tuple[torch.Tensor, VerifiedChunk]:
    max_attempts = int(settings["max_chunk_attempts"])
    verify = bool(settings["verify_fidelity"])
    last_report: dict[str, Any] = {}
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        seed = int(settings["seed"]) + chunk.index * 100 + attempt
        set_seed(seed)

        attempt_settings = dict(settings)
        if attempt > 1:
            attempt_settings["temperature"] = max(
                0.45,
                float(settings["temperature"]) - 0.10 * (attempt - 1),
            )
            attempt_settings["exaggeration"] = min(
                0.55,
                max(0.40, float(settings["exaggeration"])),
            )
            attempt_settings["cfg_weight"] = min(
                0.58,
                max(0.45, float(settings["cfg_weight"])),
            )

        wav_path = destination_dir / f"{chunk.index:05d}_attempt_{attempt}.wav"
        try:
            print(
                f"[CTEC] Bloco {chunk.index}, tentativa {attempt}, seed={seed}, "
                f"chars={len(chunk.spoken_text)}",
                flush=True,
            )
            audio = model.generate(
                chunk.spoken_text,
                language_id=language_id,
                audio_prompt_path=str(reference_path),
                **supported_generate_kwargs(model, attempt_settings),
            ).detach().float().cpu()

            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            metrics = validate_waveform(audio, model.sr, chunk.spoken_text)
            torchaudio.save(str(wav_path), audio, model.sr)

            transcript = ""
            comparison = {
                "similarity": 1.0,
                "word_coverage": 1.0,
                "missing_words": [],
                "extra_words": [],
            }
            if verify:
                transcript = transcribe_audio(wav_path, vad_filter=False)
                comparison = compare_transcript(chunk.spoken_text, transcript)

            last_report = {
                **comparison,
                "transcript": transcript,
                "metrics": metrics,
                "seed": seed,
                "attempt": attempt,
                "settings": attempt_settings,
            }
            approved = (
                comparison["similarity"] >= float(settings["min_similarity"])
                and comparison["word_coverage"] >= float(
                    settings["min_word_coverage"]
                )
            )
            if approved:
                final_wav = destination_dir / f"{chunk.index:05d}.wav"
                wav_path.replace(final_wav)
                verified = VerifiedChunk(
                    index=chunk.index,
                    original_text=chunk.original_text,
                    spoken_text=chunk.spoken_text,
                    expected_text=chunk.spoken_text,
                    transcript=transcript,
                    similarity=round(comparison["similarity"], 5),
                    word_coverage=round(comparison["word_coverage"], 5),
                    missing_words=comparison["missing_words"],
                    extra_words=comparison["extra_words"],
                    attempts=attempt,
                    seed=seed,
                    duration_seconds=float(metrics["durationSeconds"]),
                    pause_after_seconds=0.0,
                    status="approved",
                    wav_name=final_wav.name,
                )
                return audio, verified

            print(
                f"[CTEC] Bloco {chunk.index} reprovado: "
                f"similaridade={comparison['similarity']:.3f}; "
                f"cobertura={comparison['word_coverage']:.3f}; "
                f"faltantes={comparison['missing_words'][:12]}",
                flush=True,
            )
        except Exception as exc:
            last_error = exc
            print(
                f"[CTEC] Bloco {chunk.index} falhou na tentativa {attempt}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    raise RuntimeError(
        "Não foi possível gerar um bloco fiel após todas as tentativas. "
        f"bloco={chunk.index}; esperado={chunk.spoken_text!r}; "
        f"transcrição={last_report.get('transcript', '')!r}; "
        f"similaridade={last_report.get('similarity')}; "
        f"cobertura={last_report.get('word_coverage')}; "
        f"seed={last_report.get('seed')}; erro={last_error}"
    )


# ============================================================
# PCM, PAUSAS, MARCADORES E PROCESSAMENTO FINAL
# ============================================================

def trailing_silence_seconds(
    audio: torch.Tensor,
    sample_rate: int,
    threshold: float = 0.003,
) -> float:
    mono = audio.mean(dim=0) if audio.ndim == 2 else audio
    if mono.numel() == 0:
        return 0.0
    mask = torch.abs(mono) > threshold
    indices = torch.nonzero(mask, as_tuple=False)
    if indices.numel() == 0:
        return float(mono.numel() / sample_rate)
    last = int(indices[-1].item())
    return max(0.0, (mono.numel() - last - 1) / sample_rate)


def target_pause_seconds(boundary: str, settings: dict[str, Any]) -> float:
    key = {
        "continuation": "pause_continuation_ms",
        "sentence": "pause_sentence_ms",
        "legal_item": "pause_legal_item_ms",
        "paragraph": "pause_paragraph_ms",
        "article": "pause_article_ms",
    }.get(boundary, "pause_sentence_ms")
    return float(settings[key]) / 1000.0


def append_natural_pause(
    audio: torch.Tensor,
    sample_rate: int,
    boundary: str,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, float]:
    target = target_pause_seconds(boundary, settings)
    existing = trailing_silence_seconds(audio, sample_rate)
    required = max(0.0, target - existing)
    if required <= 0.001:
        return audio, 0.0
    silence = torch.zeros(
        1,
        int(round(required * sample_rate)),
        dtype=audio.dtype,
    )
    return torch.cat([audio, silence], dim=1), required


def build_final_filter(settings: dict[str, Any]) -> str:
    filters: list[str] = []

    speed = float(settings["speed"])
    filters.append(f"atempo={speed:.6f}")

    pitch = float(settings["pitch_semitones"])
    if abs(pitch) > 0.001:
        ratio = math.pow(2.0, pitch / 12.0)
        filters.append(f"rubberband=pitch={ratio:.8f}")

    gain = float(settings["gain_db"])
    if abs(gain) > 0.001:
        filters.append(f"volume={gain:.2f}dB")

    # Trim apenas quando explicitamente solicitado; limiares conservadores.
    if settings.get("trim_silence"):
        filters.append(
            "silenceremove=start_periods=1:start_duration=0.02:"
            "start_threshold=-55dB:stop_periods=1:stop_duration=0.10:"
            "stop_threshold=-55dB"
        )

    if settings.get("normalize"):
        filters.append("loudnorm=I=-17:TP=-1:LRA=8")

    filters.append("alimiter=limit=0.891")
    return ",".join(filters)


def process_final_audio(
    raw_wav: Path,
    output_path: Path,
    output_format: str,
    bitrate: str,
    settings: dict[str, Any],
) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(raw_wav),
        "-ac", "1",
        "-filter:a", build_final_filter(settings),
    ]
    if output_format == "mp3":
        command += [
            "-codec:a", "libmp3lame",
            "-b:a", bitrate,
            str(output_path),
        ]
    else:
        command += [
            "-codec:a", "pcm_s16le",
            str(output_path),
        ]
    subprocess.run(command, check=True)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def recalculate_markers(
    reports: list[VerifiedChunk],
    processed_duration: float,
    raw_duration: float,
) -> None:
    scale = (
        processed_duration / raw_duration
        if raw_duration > 0 and processed_duration > 0
        else 1.0
    )
    cursor = 0.0
    for report in reports:
        report.start_seconds = round(cursor * scale, 4)
        cursor += report.duration_seconds + report.pause_after_seconds
        report.duration_seconds = round(report.duration_seconds * scale, 4)
        report.pause_after_seconds = round(report.pause_after_seconds * scale, 4)


def encode_output(path: Path) -> str:
    raw = path.read_bytes()
    if len(raw) > MAX_RESULT_BASE64_BYTES:
        raise ValueError(
            "O áudio final ficou grande demais para retorno em Base64. "
            "Use upload por URL assinada para projetos longos."
        )
    return base64.b64encode(raw).decode("ascii")


def upload_file_to_signed_url(
    file_path: Path,
    signed_url: str,
    content_type: str,
) -> None:
    request = urllib.request.Request(
        signed_url,
        data=file_path.read_bytes(),
        method="PUT",
        headers={
            "Content-Type": content_type,
            "Content-Length": str(file_path.stat().st_size),
        },
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        if response.status not in {200, 201}:
            raise RuntimeError(
                f"Upload final falhou com HTTP {response.status}."
            )


# ============================================================
# PIPELINE ÚNICO
# ============================================================

def run_pipeline(
    job: dict[str, Any],
    data: dict[str, Any],
    *,
    long_mode: bool,
) -> dict[str, Any]:
    original_text = str(data.get("text") or "").strip()
    minimum = 100 if long_mode else 3
    if len(original_text) < minimum:
        raise ValueError(
            f"O texto precisa ter pelo menos {minimum} caracteres."
        )
    if len(original_text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"O texto excede {MAX_TEXT_CHARS} caracteres."
        )

    language_id = str(data.get("language_id") or "pt").strip().lower()
    if language_id not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Idioma não suportado: {language_id}")

    settings = resolve_settings(data)
    dictionary = data.get("pronunciation_dictionary")
    dictionary = dictionary if isinstance(dictionary, list) else []
    prepared_text = prepare_text(
        original_text,
        settings["text_mode"],
        dictionary,
    )

    if to_bool(data.get("preview"), False):
        preview_chars = int(
            clamp(safe_float(data.get("preview_chars"), 450), 80, 1500)
        )
        prepared_text = (
            prepared_text[:preview_chars].rsplit(" ", 1)[0].strip()
            or prepared_text[:preview_chars]
        )

    chunks = split_text_audited(
        original_text,
        prepared_text,
        int(settings["chunk_target"]),
        int(settings["chunk_limit"]),
    )

    output_format = (
        "mp3" if long_mode
        else str(data.get("output_format") or "mp3").strip().lower()
    )
    if output_format not in {"mp3", "wav"}:
        raise ValueError("output_format deve ser mp3 ou wav.")
    bitrate = str(data.get("mp3_bitrate") or "128k").strip().lower()
    if bitrate not in {"96k", "128k", "160k"}:
        bitrate = "128k"

    if bool(settings["verify_fidelity"]):
        get_whisper(required=True)

    model = get_model()
    started = time.time()

    with _GENERATION_LOCK, tempfile.TemporaryDirectory(
        prefix="ctec_voice_v5_"
    ) as tmp:
        root = Path(tmp)
        chunks_dir = root / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        original_reference = receive_reference_audio(data, root)
        canonical_reference = canonicalize_reference(
            original_reference,
            root,
            model.sr,
        )
        reference_metrics = analyze_reference_audio(canonical_reference)

        segments: list[torch.Tensor] = []
        reports: list[VerifiedChunk] = []
        retry_count = 0

        for chunk in chunks:
            progress(
                job,
                stage="generating",
                stageLabel=f"Gerando e verificando trecho {chunk.index} de {len(chunks)}",
                progress=0.08 + (chunk.index - 1) / len(chunks) * 0.82,
                currentChunk=chunk.index,
                totalChunks=len(chunks),
                elapsedSeconds=int(time.time() - started),
            )

            audio, verified = generate_verified_chunk(
                model,
                chunk,
                canonical_reference,
                language_id,
                settings,
                chunks_dir,
            )
            retry_count += max(0, verified.attempts - 1)

            if chunk.index < len(chunks):
                audio, added_pause = append_natural_pause(
                    audio,
                    model.sr,
                    chunk.boundary,
                    settings,
                )
                verified.pause_after_seconds = round(added_pause, 4)

            segments.append(audio)
            reports.append(verified)

        prepared_count = len(chunks)
        generated_count = len(reports)
        verified_count = sum(1 for item in reports if item.status == "approved")
        assembled_count = len(segments)

        if not (
            prepared_count
            == generated_count
            == verified_count
            == assembled_count
        ):
            raise RuntimeError(
                "Contagens divergentes; áudio final bloqueado. "
                f"preparados={prepared_count}; gerados={generated_count}; "
                f"verificados={verified_count}; montados={assembled_count}"
            )

        progress(
            job,
            stage="joining",
            stageLabel="Unindo PCM e processando arquivo final",
            progress=0.93,
            currentChunk=len(chunks),
            totalChunks=len(chunks),
            elapsedSeconds=int(time.time() - started),
        )

        combined = torch.cat(segments, dim=1)
        raw_wav = root / "ctec_raw_pcm.wav"
        torchaudio.save(str(raw_wav), combined, model.sr)
        raw_duration = float(combined.shape[1] / model.sr)

        final_path = root / f"ctec-voz-final.{output_format}"
        process_final_audio(
            raw_wav,
            final_path,
            output_format,
            bitrate,
            settings,
        )
        final_duration = probe_duration(final_path)
        recalculate_markers(reports, final_duration, raw_duration)

        average_similarity = (
            sum(r.similarity for r in reports) / len(reports)
            if reports else 0.0
        )
        minimum_similarity = (
            min(r.similarity for r in reports)
            if reports else 0.0
        )

        mime_type = (
            "audio/mpeg" if output_format == "mp3" else "audio/wav"
        )
        upload_url = str(data.get("final_upload_url") or "").strip()
        audio_base64 = None

        if upload_url:
            progress(
                job,
                stage="uploading",
                stageLabel="Enviando arquivo final",
                progress=0.98,
                currentChunk=len(chunks),
                totalChunks=len(chunks),
                elapsedSeconds=int(time.time() - started),
            )
            upload_file_to_signed_url(final_path, upload_url, mime_type)
        elif long_mode:
            raise ValueError(
                "Projeto longo exige final_upload_url para upload direto."
            )
        else:
            audio_base64 = encode_output(final_path)

        return {
            "status": "ok",
            "action": "generate_long_project" if long_mode else "generate",
            "audio_base64": audio_base64,
            "mime_type": mime_type,
            "file_name": final_path.name,
            "size_bytes": final_path.stat().st_size,
            "sample_rate": model.sr,
            "final_duration": round(final_duration, 3),
            "duration_seconds": round(final_duration, 3),
            "device": DEVICE,
            "model_version": MODEL_VERSION,
            "language_id": language_id,
            "voice_id": str(data.get("voice_id") or "") or None,
            "settings": settings,
            "reference_metrics": reference_metrics,
            "original_text": original_text,
            "prepared_text": prepared_text,
            "original_characters": len(original_text),
            "prepared_characters": len(prepared_text),
            "total_chunks": prepared_count,
            "generated_chunks": generated_count,
            "verified_chunks": verified_count,
            "failed_chunks": 0,
            "retry_count": retry_count,
            "seed_base": int(settings["seed"]),
            "average_similarity": round(average_similarity, 5),
            "minimum_similarity": round(minimum_similarity, 5),
            "elapsed_seconds": int(time.time() - started),
            "chunks": [asdict(item) for item in reports],
            "markers": [
                {
                    "index": item.index,
                    "startSeconds": item.start_seconds,
                    "durationSeconds": item.duration_seconds,
                    "pauseAfterSeconds": item.pause_after_seconds,
                    "text": item.spoken_text,
                }
                for item in reports
            ],
        }


# ============================================================
# CALIBRAÇÃO CORRIGIDA
# ============================================================

CALIBRATION_TEXTS = [
    "A administração pública deverá obedecer aos princípios previstos na Constituição Federal.",
    "Artigo quinto. Todos são iguais perante a lei, sem distinção de qualquer natureza.",
    (
        "Artigo quinto. Parágrafo primeiro: as normas definidoras dos direitos "
        "e garantias fundamentais têm aplicação imediata. Inciso primeiro: "
        "homens e mulheres são iguais em direitos e obrigações; Alínea a: "
        "aplicação imediata."
    ),
    (
        "São objetivos do teste: preservar a ordem; respeitar o ponto e vírgula; "
        "e manter a leitura integral."
    ),
    (
        "Artigo primeiro. Esta é a primeira unidade. Parágrafo primeiro: esta é "
        "a segunda unidade. Inciso primeiro: esta é a terceira unidade. "
        "Artigo segundo. Esta é a quarta unidade, usada para testar vários blocos."
    ),
]


def apply_speed_to_preview(
    source: Path,
    destination: Path,
    speed: float,
) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(source),
            "-filter:a", f"atempo={speed:.6f}",
            "-codec:a", "pcm_s16le",
            str(destination),
        ],
        check=True,
    )


def calibrate(job: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    target_profile = str(
        data.get("target_profile") or "law_natural"
    ).strip().lower()
    base = dict(PROFILES.get(target_profile, PROFILES["law_natural"]))
    base.update(resolve_settings(data))
    fixed_seed = int(base["seed"])

    candidates = [
        {"name": "Controle", **CONTROL_PROFILE},
        {
            "name": "Natural",
            "speed": 0.98,
            "exaggeration": 0.48,
            "cfg_weight": 0.50,
            "temperature": 0.78,
        },
        {
            "name": "Estável",
            "speed": 0.96,
            "exaggeration": 0.44,
            "cfg_weight": 0.52,
            "temperature": 0.68,
        },
    ]

    model = get_model()
    get_whisper(required=True)

    with _GENERATION_LOCK, tempfile.TemporaryDirectory(
        prefix="ctec_calibration_v5_"
    ) as tmp:
        root = Path(tmp)
        original_reference = receive_reference_audio(data, root)
        reference = canonicalize_reference(
            original_reference,
            root,
            model.sr,
        )
        reference_metrics = analyze_reference_audio(reference)
        results: list[dict[str, Any]] = []

        for candidate_index, candidate in enumerate(candidates, start=1):
            settings = dict(base)
            settings.update(candidate)
            settings["seed"] = fixed_seed
            candidate_reports = []
            preview_b64 = ""

            for text_index, test_text in enumerate(CALIBRATION_TEXTS, start=1):
                progress(
                    job,
                    stage="calibrating",
                    stageLabel=(
                        f"Calibração {candidate_index}/{len(candidates)} — "
                        f"teste {text_index}/{len(CALIBRATION_TEXTS)}"
                    ),
                    progress=(
                        ((candidate_index - 1) * len(CALIBRATION_TEXTS) + text_index)
                        / (len(candidates) * len(CALIBRATION_TEXTS))
                    ),
                )
                prepared = prepare_text(
                    test_text,
                    settings["text_mode"],
                    data.get("pronunciation_dictionary")
                    if isinstance(data.get("pronunciation_dictionary"), list)
                    else [],
                )
                chunks = split_text_audited(
                    test_text,
                    prepared,
                    int(settings["chunk_target"]),
                    int(settings["chunk_limit"]),
                )

                test_segments = []
                test_reports = []
                for chunk in chunks:
                    audio, report = generate_verified_chunk(
                        model,
                        chunk,
                        reference,
                        "pt",
                        settings,
                        root,
                    )
                    test_segments.append(audio)
                    test_reports.append(report)

                raw = torch.cat(test_segments, dim=1)
                raw_path = root / f"cal_{candidate_index}_{text_index}_raw.wav"
                sped_path = root / f"cal_{candidate_index}_{text_index}_speed.wav"
                torchaudio.save(str(raw_path), raw, model.sr)
                apply_speed_to_preview(
                    raw_path,
                    sped_path,
                    float(settings["speed"]),
                )
                duration = probe_duration(sped_path)
                similarity = sum(r.similarity for r in test_reports) / len(test_reports)
                coverage = sum(r.word_coverage for r in test_reports) / len(test_reports)
                candidate_reports.append({
                    "test": text_index,
                    "similarity": similarity,
                    "wordCoverage": coverage,
                    "durationSeconds": duration,
                    "chunks": len(chunks),
                    "transcripts": [r.transcript for r in test_reports],
                })

                if text_index == 3:
                    mp3 = root / f"preview_{candidate_index}.mp3"
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-loglevel", "error",
                            "-i", str(sped_path),
                            "-codec:a", "libmp3lame",
                            "-b:a", "128k",
                            str(mp3),
                        ],
                        check=True,
                    )
                    preview_b64 = base64.b64encode(mp3.read_bytes()).decode("ascii")

            avg_similarity = sum(
                item["similarity"] for item in candidate_reports
            ) / len(candidate_reports)
            avg_coverage = sum(
                item["wordCoverage"] for item in candidate_reports
            ) / len(candidate_reports)
            score = (avg_similarity * 0.60 + avg_coverage * 0.40) * 100

            results.append({
                "name": candidate["name"],
                "score": round(score, 2),
                "textFidelity": round(avg_similarity * 100, 2),
                "wordCoverage": round(avg_coverage * 100, 2),
                "referenceQuality": reference_metrics["referenceQuality"],
                "settings": {
                    key: settings[key]
                    for key in CONTROL_PROFILE.keys()
                    if key in settings
                },
                "tests": candidate_reports,
                "previewAudioBase64": preview_b64,
                "previewMimeType": "audio/mpeg",
            })

        best = max(results, key=lambda item: item["score"])
        recommended = dict(best["settings"])
        # Aliases apenas por compatibilidade; o núcleo usa snake_case.
        recommended.update({
            "cfgWeight": recommended["cfg_weight"],
            "maxChunkCharacters": recommended["chunk_limit"],
            "periodPauseMs": recommended["pause_sentence_ms"],
            "paragraphPauseMs": recommended["pause_paragraph_ms"],
        })
        return {
            "status": "ok",
            "action": "calibrate",
            "score": best["score"],
            "completeness": best["wordCoverage"],
            "rhythmStability": None,
            "referenceQuality": reference_metrics["referenceQuality"],
            "recommendedSettings": recommended,
            "bestCandidate": best["name"],
            "candidates": results,
            "note": (
                "Ritmo não é chamado de estabilidade sem medição específica. "
                "Fidelidade vocal não é informada sem speaker embedding."
            ),
        }


# ============================================================
# AUTOTESTES SEM GPU
# ============================================================

def run_self_tests() -> dict[str, Any]:
    legal = (
        "Art. 5º Todos são iguais perante a lei, sem distinção de qualquer natureza.\n\n"
        "§ 1º As normas definidoras dos direitos e garantias fundamentais têm aplicação imediata.\n\n"
        "I – homens e mulheres são iguais em direitos e obrigações;\n"
        "II – ninguém será obrigado a fazer ou deixar de fazer alguma coisa senão em virtude de lei;\n"
        "a) aplicação imediata;\n"
        "b) respeito à legalidade.\n\n"
        "Lei nº 8.112/1990."
    )
    prepared = normalize_law_text(legal, [])
    required = [
        "Artigo quinto",
        "Parágrafo primeiro",
        "Inciso um:",
        "Inciso dois:",
        "Alínea a:",
        "Alínea b:",
        ";",
        "Lei número oito mil, cento e doze",
    ]
    missing = [item for item in required if item not in prepared]
    if missing:
        raise AssertionError(f"Normalização não produziu: {missing}")

    pdf = (
        "A administração pública deverá obedecer\n"
        "aos princípios previstos na Constituição Federal.\n\n"
        "Novo parágrafo real."
    )
    pdf_prepared = normalize_pdf_line_breaks(pdf)
    if "obedecer aos princípios" not in pdf_prepared:
        raise AssertionError("Quebra visual de PDF não foi reunida.")
    if "\n\nNovo parágrafo" not in pdf_prepared:
        raise AssertionError("Parágrafo real não foi preservado.")

    long_text = " ".join(
        f"Artigo {i}. Esta é uma frase de teste suficientemente completa;"
        for i in range(1, 30)
    )
    chunks = split_text_audited(long_text, long_text, 250, 300)
    if not chunks:
        raise AssertionError("Divisão não produziu blocos.")
    if any(len(c.spoken_text) > 300 for c in chunks):
        raise AssertionError("Há bloco acima de 300 caracteres.")
    if [c.index for c in chunks] != list(range(1, len(chunks) + 1)):
        raise AssertionError("Ordem numérica incorreta.")

    expected = comparison_tokens(long_text)
    assembled = comparison_tokens(" ".join(c.spoken_text for c in chunks))
    coverage = multiset_coverage(expected, assembled)
    if coverage < 0.999:
        raise AssertionError(f"Cobertura insuficiente: {coverage}")

    return {
        "status": "ok",
        "tests": {
            "legal_normalization": "passed",
            "pdf_line_breaks": "passed",
            "hierarchical_chunking": "passed",
            "order_manifest": "passed",
            "content_coverage": round(coverage, 5),
        },
        "prepared_example": prepared,
        "chunk_count": len(chunks),
    }


# ============================================================
# CAPACIDADES E DISPATCH
# ============================================================

def capabilities() -> dict[str, Any]:
    loader_signature = str(
        inspect.signature(ChatterboxMultilingualTTS.from_pretrained)
    )
    generate_signature = str(
        inspect.signature(ChatterboxMultilingualTTS.generate)
    )
    return {
        "status": "ok",
        "worker": "CTEC Estúdio de Voz",
        "version": "5.0.0",
        "device": DEVICE,
        "requested_model_version": MODEL_VERSION,
        "model_loaded": _MODEL is not None,
        "from_pretrained_signature": loader_signature,
        "generate_signature": generate_signature,
        "profiles": PROFILES,
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        "reference_inputs": [
            "voice_id",
            "reference_audio_url",
            "reference_audio_base64",
        ],
        "output_formats": ["mp3", "wav"],
        "legal_normalization": True,
        "custom_pronunciation_dictionary": True,
        "whisper_verification": WhisperModel is not None,
        "pcm_assembly": True,
        "single_final_encoding": True,
    }


def generate(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("input") or {}
    action = str(data.get("action") or "generate").strip().lower()

    if action in {"capabilities", "config", "health"}:
        return capabilities()
    if action in {"self_test", "self_tests", "test"}:
        return run_self_tests()
    if action == "calibrate":
        return calibrate(job, data)
    if action == "generate_long_project":
        return run_pipeline(job, data, long_mode=True)
    if action == "normalize_legal_text":
        original = str(data.get("text") or "")
        dictionary = data.get("pronunciation_dictionary")
        prepared = normalize_law_text(
            original,
            dictionary if isinstance(dictionary, list) else [],
        )
        chunks = split_text_audited(
            original,
            prepared,
            DEFAULT_CHUNK_TARGET,
            DEFAULT_CHUNK_MAX,
        )
        return {
            "status": "ok",
            "action": "normalize_legal_text",
            "original_text": original,
            "prepared_text": prepared,
            "original_characters": len(original),
            "prepared_characters": len(prepared),
            "chunks": [asdict(chunk) for chunk in chunks],
        }
    return run_pipeline(job, data, long_mode=False)


if __name__ == "__main__":
    print("[CTEC] Iniciando CTEC Estúdio de Voz Worker 5.0...", flush=True)
    print(
        f"[CTEC] Device={DEVICE}; modelo solicitado={MODEL_VERSION}; "
        f"Whisper={WHISPER_MODEL_NAME}",
        flush=True,
    )
    runpod.serverless.start({"handler": generate})
