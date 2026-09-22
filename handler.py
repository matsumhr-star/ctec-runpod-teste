import base64
import inspect
import json
import logging
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
import hashlib
import unicodedata
import wave
from contextlib import contextmanager
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
SAFE_TTS_FALLBACK_CHARS = int(os.getenv("CTEC_SAFE_TTS_FALLBACK_CHARS", "140"))
MIN_TTS_SUBCHUNK_CHARS = int(os.getenv("CTEC_MIN_TTS_SUBCHUNK_CHARS", "72"))
MAX_ADAPTIVE_FALLBACK_DEPTH = int(os.getenv("CTEC_MAX_FALLBACK_DEPTH", "3"))
EARLY_EOS_MIN_WORDS = int(os.getenv("CTEC_EARLY_EOS_MIN_WORDS", "10"))
EARLY_EOS_NARRATION_WPM = float(os.getenv("CTEC_EARLY_EOS_WPM", "155"))
EARLY_EOS_MIN_DURATION_RATIO = float(
    os.getenv("CTEC_EARLY_EOS_MIN_DURATION_RATIO", "0.45")
)
WORKER_CONTRACT_VERSION = 2

_LEGAL_NUMBER_WORDS = (
    "zero|um|uma|dois|duas|três|tres|quatro|cinco|seis|sete|oito|nove|dez|"
    "onze|doze|treze|catorze|quatorze|quinze|dezesseis|dezessete|dezoito|"
    "dezenove|vinte|trinta|quarenta|cinquenta|sessenta|setenta|oitenta|"
    "noventa|cem|cento|duzentos|duzentas|trezentos|trezentas|quatrocentos|"
    "quatrocentas|quinhentos|quinhentas|seiscentos|seiscentas|setecentos|"
    "setecentas|oitocentos|oitocentas|novecentos|novecentas|mil|milhão|"
    "milhao|milhões|milhoes|primeiro|primeira|segundo|segunda|terceiro|"
    "terceira|quarto|quarta|quinto|quinta|sexto|sexta|sétimo|setimo|sétima|"
    "setima|oitavo|oitava|nono|nona|décimo|decimo|décima|decima"
)
_LEGAL_NUMBER_EXPRESSION = (
    rf"(?:{_LEGAL_NUMBER_WORDS})"
    rf"(?:\s+(?:e\s+)?(?:{_LEGAL_NUMBER_WORDS})){{0,12}}"
)
_LEGAL_MONTHS = (
    "janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|"
    "setembro|outubro|novembro|dezembro"
)

_MODEL: ChatterboxMultilingualTTS | None = None
_LOADED_MODEL_VERSION: str | None = None
_MODEL_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()
_WHISPER = None
_WHISPER_LOCK = threading.Lock()


class _ChatterboxGenerationLogCapture(logging.Handler):
    """Observa os avisos do Chatterbox sem alterar a biblioteca instalada."""

    _REPETITION_RE = re.compile(
        r"Detected\s+(\d+)x\s+repetition\s+of\s+token\s+([^\s,;]+)",
        re.IGNORECASE,
    )
    _EOS_STEP_RE = re.compile(
        r"Stopping\s+generation\s+at\s+step\s+(\d+)",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []
        self.repetition_count: int | None = None
        self.repeated_token: str | None = None
        self.token_repetition = False
        self.long_tail = False
        self.alignment_repetition = False
        self.eos_detected = False
        self.eos_step: int | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.feed(record.getMessage())
        except Exception:
            return

    def feed(self, message: str) -> None:
        clean = str(message or "").strip()
        if not clean:
            return
        relevant = (
            "repetition of token" in clean
            or "forcing EOS token" in clean
            or "EOS token detected" in clean
        )
        if not relevant:
            return
        self.messages.append(clean[:600])
        repetition = self._REPETITION_RE.search(clean)
        if repetition:
            self.repetition_count = int(repetition.group(1))
            self.repeated_token = repetition.group(2)
        if "forcing EOS token" in clean:
            self.token_repetition = bool(re.search(
                r"token_repetition\s*=\s*True", clean, re.IGNORECASE
            ))
            self.long_tail = bool(re.search(
                r"long_tail\s*=\s*True", clean, re.IGNORECASE
            ))
            self.alignment_repetition = bool(re.search(
                r"alignment_repetition\s*=\s*True", clean, re.IGNORECASE
            ))
        if "EOS token detected" in clean:
            self.eos_detected = True
            step = self._EOS_STEP_RE.search(clean)
            if step:
                self.eos_step = int(step.group(1))

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_repetition": self.token_repetition,
            "repetition_count": self.repetition_count,
            "repeated_token": self.repeated_token,
            "long_tail": self.long_tail,
            "alignment_repetition": self.alignment_repetition,
            "eos_detected": self.eos_detected,
            "eos_step": self.eos_step,
            "messages": self.messages,
        }


@contextmanager
def _capture_chatterbox_generation_logs():
    """Captura logging/loguru durante uma geração serializada pelo worker."""
    capture = _ChatterboxGenerationLogCapture()
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.addHandler(capture)
    if previous_level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    loguru_logger = None
    loguru_sink_id = None
    try:
        try:
            from loguru import logger as imported_loguru_logger

            loguru_logger = imported_loguru_logger
            loguru_sink_id = loguru_logger.add(
                lambda message: capture.feed(str(message)),
                level="INFO",
                format="{message}",
            )
        except Exception:
            loguru_logger = None
            loguru_sink_id = None
        yield capture
    finally:
        root_logger.removeHandler(capture)
        root_logger.setLevel(previous_level)
        if loguru_logger is not None and loguru_sink_id is not None:
            try:
                loguru_logger.remove(loguru_sink_id)
            except Exception:
                pass


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
    global _MODEL, _LOADED_MODEL_VERSION
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                loader_signature = inspect.signature(
                    ChatterboxMultilingualTTS.from_pretrained
                )
                loader_kwargs: dict[str, Any] = {"device": DEVICE}
                if "t3_model" in loader_signature.parameters:
                    loader_kwargs["t3_model"] = MODEL_VERSION
                    _LOADED_MODEL_VERSION = MODEL_VERSION
                else:
                    _LOADED_MODEL_VERSION = "default-compatible"
                print(
                    "[CTEC] Carregando Chatterbox Multilingual "
                    f"({_LOADED_MODEL_VERSION}) em {DEVICE}...",
                    flush=True,
                )
                _MODEL = ChatterboxMultilingualTTS.from_pretrained(
                    **loader_kwargs,
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


def punctuate_legal_speech_structure(text: str) -> str:
    """Acrescenta somente pausas de fala; não muda nenhuma palavra da lei."""
    # Cabeçalhos que vieram em uma linha própria do PDF.
    text = re.sub(
        r"(?m)^\s*([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇ\s]{4,})\s*$",
        lambda match: match.group(1).strip() + ".",
        text,
    )

    # "Capítulo um DISPOSIÇÕES..." vira "Capítulo um. DISPOSIÇÕES...".
    heading_pattern = rf"(?:Título|Capítulo|Seção|Subseção|Livro|Parte)\s+{_LEGAL_NUMBER_EXPRESSION}"
    text = re.sub(
        rf"\b({heading_pattern})"
        rf"(?=\s+(?!(?:e\s+)?(?:{_LEGAL_NUMBER_WORDS})\b)"
        rf"[A-Za-zÀ-ÖØ-öø-ÿ])",
        r"\1. ",
        text,
        flags=re.IGNORECASE,
    )

    # Se o título em caixa alta estiver colado ao artigo, cria a fronteira de fala.
    text = re.sub(
        r"([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇ\s]{4,})"
        r"(?=\s+Artigo\b)",
        lambda match: match.group(1).rstrip() + ".",
        text,
    )

    # Só trata Artigo/Parágrafo/Inciso como cabeçalho quando aparecem no início
    # ou depois de uma fronteira forte. Referências como "no Artigo 165" ficam intactas.
    provision_pattern = rf"(?:Artigo|Parágrafo|Inciso)\s+{_LEGAL_NUMBER_EXPRESSION}"
    text = re.sub(
        rf"(^|[.!?:]\s+)({provision_pattern})"
        rf"(?=\s+(?!(?:e\s+)?(?:{_LEGAL_NUMBER_WORDS})\b)"
        rf"[A-Za-zÀ-ÖØ-öø-ÿ])",
        lambda match: match.group(1) + match.group(2).rstrip() + ". ",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(r"\.{2,}", ".", text)
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
        r"§\s*(\d+)(?:\s*[º°])?",
        lambda match: f"Parágrafo {number_words(int(match.group(1)), True)}",
        text,
    )
    text = re.sub(
        r"\bArts?\.\s*(\d+)(?:\s*[º°])?",
        lambda match: (
            f"Artigo {number_words(int(match.group(1)), True)}"
            if int(match.group(1)) <= 9
            else f"Artigo {number_words(int(match.group(1)))}"
        ),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bArtigo\s+(\d+)(?:\s*[º°])?",
        lambda match: (
            f"Artigo {number_words(int(match.group(1)), True)}"
            if int(match.group(1)) <= 9
            else f"Artigo {number_words(int(match.group(1)))}"
        ),
        text,
        flags=re.IGNORECASE,
    )

    # Algarismos romanos de cabeçalhos precisam ser escritos por extenso na
    # cópia narrada. Caso contrário, "CAPÍTULO I" tende a ser pronunciado como
    # a letra "i" e o ASR pode registrar apenas a conjunção "e".
    text = re.sub(
        r"\b(Título|Capítulo|Seção|Subseção|Livro|Parte)\s+"
        r"([IVXLCDM]{1,12})\b",
        lambda match: (
            f"{match.group(1)} "
            f"{number_words(roman_to_int(match.group(2)))}"
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
        r"\bLei\s+Complementar\s+número\s+(\d{1,6})\b",
        lambda match: "Lei Complementar número " + number_words(int(match.group(1))),
        text,
        flags=re.IGNORECASE,
    )

    # Referência legal brasileira com separador de milhar, por exemplo:
    # "Lei número 15.121" -> "Lei número quinze mil cento e vinte e um".
    # A transformação existe apenas na cópia de fala; o texto jurídico original
    # recebido pelo serviço permanece intacto.
    text = re.sub(
        r"\bLei\s+número\s+(\d{1,3}(?:\.\d{3})+)\b",
        lambda match: (
            "Lei número "
            + number_words(int(match.group(1).replace(".", "")))
        ),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b(\d{{1,2}})\s+de\s+({_LEGAL_MONTHS})\s+de\s+(\d{{4}})\b",
        lambda match: (
            f"{number_words(int(match.group(1)))} de {match.group(2)} de "
            f"{number_words(int(match.group(3)))}"
        ),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(para|exercício\s+de|ano\s+de)\s+((?:19|20)\d{2})\b",
        lambda match: f"{match.group(1)} {number_words(int(match.group(2)))}",
        text,
        flags=re.IGNORECASE,
    )

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

    # Mantém a oração inteira dentro do mesmo contexto acústico. Antes, cada
    # dois-pontos/ponto e vírgula virava nova linha e reiniciava a interpretação.
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s*:\s*", ": ", text)
    text = re.sub(r"\s*[—–]\s*", " — ", text)
    text = re.sub(r" +([,.;:])", r"\1", text)
    text = punctuate_legal_speech_structure(text)
    return collapse_soft_line_breaks(text)


def collapse_soft_line_breaks(text: str) -> str:
    """Une linhas visuais de PDF e conserva somente parágrafos verdadeiros."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]*\n[ \t]*\n(?:[ \t]*\n)*[ \t]*", "\n\n", text)
    text = re.sub(r"(?<!\n)[ \t]*\n[ \t]*(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



def prepare_text(text: str, mode: str, custom_dictionary=None) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if mode == "law":
        text = normalize_law_text(text, custom_dictionary)
    else:
        text = collapse_soft_line_breaks(text)
    return text.strip()


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
    target: int | None = None,
) -> list[tuple[str, bool]]:
    """Agrupa várias frases relacionadas em uma única chamada ao TTS."""
    # Mantido no contrato por compatibilidade. Sobreposição real repetiria
    # palavras no áudio e, portanto, não é aplicada.
    _ = context_margin_words
    effective_limit = max(120, int(limit))
    target_limit = int(target or min(300, effective_limit))
    target_limit = int(clamp(target_limit, 120, effective_limit))

    source = collapse_soft_line_breaks(text)
    if not split_by_legal_structure:
        source = re.sub(r"\s*\n{2,}\s*", " ", source)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", source) if p.strip()]

    units: list[tuple[str, bool]] = []
    for paragraph in paragraphs:
        sentences = (
            re.split(r"(?<=[.!?])\s+", paragraph)
            if preserve_complete_sentences
            else [paragraph]
        )
        sentences = [item.strip() for item in sentences if item.strip()]
        expanded: list[str] = []
        for sentence in sentences:
            if len(sentence) > effective_limit:
                expanded.extend(split_long_unit(sentence, effective_limit))
            else:
                expanded.append(sentence)
        for index, sentence in enumerate(expanded):
            units.append((sentence, index == len(expanded) - 1))

    chunks: list[tuple[str, bool]] = []
    current = ""
    current_end = False
    current_units = 0

    def flush() -> None:
        nonlocal current, current_end, current_units
        if current:
            chunks.append((current.strip(), current_end))
        current = ""
        current_end = False
        current_units = 0

    for unit, paragraph_end in units:
        candidate = f"{current} {unit}".strip()
        exceeds_limit = bool(current) and len(candidate) > effective_limit
        reached_target = (
            bool(current)
            and current_units >= 2
            and len(current) >= target_limit
            and len(candidate) > target_limit
        )
        if exceeds_limit or reached_target:
            flush()
            candidate = unit
        current = candidate
        current_end = paragraph_end
        current_units += 1

    flush()
    return chunks


def legal_chunk_complexity(value: str) -> dict[str, Any]:
    """Classifica dificuldade jurídica sem usar só a quantidade de caracteres."""
    text = str(value or "")
    plain = "".join(
        char for char in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(char) != "Mn"
    )
    legal_references = len(re.findall(
        r"\b(?:artigo|paragrafo|inciso|alinea|capitulo|titulo|secao|subsecao|"
        r"lei complementar|constituicao|codigo)\b",
        plain,
    ))
    numeric_references = len(re.findall(
        r"\b(?:artigo|paragrafo|inciso|lei(?:\s+complementar)?(?:\s+numero)?)\s+"
        r"(?:\d+|[ivxlcdm]+|zero|um|uma|dois|duas|tres|quatro|cinco|seis|sete|"
        r"oito|nove|dez|primeiro|segundo|terceiro|cento|mil)\b",
        plain,
    ))
    dates = len(re.findall(
        r"\b(?:\d{1,2}|um|dois|tres|quatro|cinco|seis|sete|oito|nove|dez|"
        r"onze|doze|treze|catorze|quinze|dezesseis|dezessete|dezoito|"
        r"dezenove|vinte|trinta)\s+de\s+(?:" + _LEGAL_MONTHS.replace("ç", "c") + r")\b",
        plain,
    ))
    uppercase_headings = len(re.findall(
        r"(?:^|\s)(?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{2,}(?:\s+|$)){2,}",
        text,
    ))
    sentences = [item.strip() for item in re.split(r"[.!?]+", text) if item.strip()]
    longest_sentence = max((len(item) for item in sentences), default=len(text))

    score = min(8, legal_references)
    score += min(6, numeric_references * 2)
    score += min(4, dates * 2)
    score += min(4, uppercase_headings * 2)
    if longest_sentence > 180:
        score += 2
    if longest_sentence > 260:
        score += 2
    if legal_references >= 4:
        score += 2

    level = "alta" if score >= 10 else "média" if score >= 5 else "baixa"
    return {
        "level": level,
        "score": score,
        "characters": len(text),
        "words": len(text.split()),
        "legal_references": legal_references,
        "numeric_references": numeric_references,
        "dates": dates,
        "uppercase_headings": uppercase_headings,
        "longest_sentence": longest_sentence,
    }


def _protected_legal_reference_spans(value: str) -> list[tuple[int, int]]:
    """Localiza referências que nunca podem ser cortadas no meio."""
    number = rf"(?:\d+[º°]?|[IVXLCDM]+|{_LEGAL_NUMBER_EXPRESSION})"
    patterns = [
        rf"\bArtigo\s+{number}(?:\s*,?\s*Parágrafo\s+{number})?",
        rf"\bParágrafo\s+{number}",
        rf"\bInciso\s+{number}",
        rf"\bLei\s+Complementar\s+(?:número\s+)?{number}"
        rf"(?:\s*,?\s*de\s+{number}\s+de\s+(?:{_LEGAL_MONTHS})\s+de\s+{number})?",
        rf"\bLei\s+(?:número\s+)?(?:\d{1,3}(?:\.\d{3})+|{number})"
        rf"(?:\s*,?\s*de\s+{number}\s+de\s+(?:{_LEGAL_MONTHS})\s+de\s+{number})?",
        rf"\b{number}\s+de\s+(?:{_LEGAL_MONTHS})\s+de\s+{number}\b",
        r"\b(?:Art|Arts)\.\s*\d+[º°]?",
        r"§\s*\d+[º°]?",
    ]
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        spans.extend(
            (match.start(), match.end())
            for match in re.finditer(pattern, value, flags=re.IGNORECASE)
        )
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _boundary_cuts_protected_reference(
    boundary: int,
    spans: list[tuple[int, int]],
) -> bool:
    return any(start < boundary < end for start, end in spans)


def _split_oversized_legal_unit(value: str, limit: int) -> list[str]:
    """Usa ponto/semicolon/vírgula/palavra nessa ordem, protegendo referências."""
    source = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(source) <= limit:
        return [source] if source else []

    spans = _protected_legal_reference_spans(source)
    output: list[str] = []
    start = 0
    minimum = max(35, min(int(MIN_TTS_SUBCHUNK_CHARS), limit // 2))

    while len(source) - start > limit:
        upper = min(len(source), start + limit)
        candidates: list[tuple[int, int]] = []
        for match in re.finditer(r"[.!?;:,]\s+", source[start:upper + 1]):
            boundary = start + match.end()
            if boundary - start < minimum:
                continue
            if _boundary_cuts_protected_reference(boundary, spans):
                continue
            punctuation = source[boundary - 2:boundary].strip()[-1:]
            priority = 0 if punctuation in ".!?" else 1 if punctuation in ";:" else 2
            candidates.append((priority, boundary))

        boundary = 0
        if candidates:
            # Mantém o ponto mais próximo do limite dentro da melhor prioridade.
            best_priority = min(priority for priority, _ in candidates)
            boundary = max(
                point for priority, point in candidates
                if priority == best_priority
            )
        else:
            whitespace = [
                start + match.start()
                for match in re.finditer(r"\s+", source[start:upper + 1])
                if start + match.start() - start >= minimum
                and not _boundary_cuts_protected_reference(
                    start + match.start(),
                    spans,
                )
            ]
            if whitespace:
                boundary = max(whitespace)

        if not boundary:
            crossing = [
                (span_start, span_end) for span_start, span_end in spans
                if span_start < upper < span_end
            ]
            if crossing:
                span_start, span_end = crossing[0]
                if span_start - start >= minimum:
                    boundary = span_start
                else:
                    boundary = span_end
            else:
                # Nunca corta uma palavra no meio apenas para obedecer ao limite
                # de caracteres. O limite é um alvo de segurança para o TTS, não
                # autorização para alterar o texto jurídico.
                boundary = upper
                cuts_word = (
                    boundary > start
                    and boundary < len(source)
                    and source[boundary - 1].isalnum()
                    and source[boundary].isalnum()
                )
                if cuts_word:
                    # Primeiro tenta recuar para a última fronteira segura dentro
                    # do limite, desde que ela não produza um fragmento minúsculo.
                    backward = [
                        start + match.start()
                        for match in re.finditer(r"\s+", source[start:upper + 1])
                        if start + match.start() - start >= minimum
                        and not _boundary_cuts_protected_reference(
                            start + match.start(),
                            spans,
                        )
                    ]
                    if backward:
                        boundary = max(backward)
                    else:
                        # Se não houver fronteira segura antes do limite, avança
                        # até o fim da palavra atual. É preferível exceder poucos
                        # caracteres a mutilar um token como FUNDAMENTAIS.
                        forward_match = re.search(r"\s+", source[upper:])
                        if forward_match:
                            boundary = upper + forward_match.start()
                        else:
                            boundary = len(source)

        piece = source[start:boundary].strip()
        if not piece or boundary <= start:
            break
        output.append(piece)
        start = boundary
        while start < len(source) and source[start].isspace():
            start += 1

    tail = source[start:].strip()
    if tail:
        output.append(tail)
    return output


def _leading_legal_structure(value: str) -> str:
    plain = "".join(
        char for char in unicodedata.normalize("NFD", str(value or "").lower())
        if unicodedata.category(char) != "Mn"
    ).lstrip()
    for name in ("titulo", "capitulo", "secao", "subsecao", "livro", "parte"):
        if plain.startswith(name + " "):
            return "heading"
    if plain.startswith("artigo "):
        return "article"
    if plain.startswith("paragrafo "):
        return "paragraph"
    if plain.startswith("inciso "):
        return "inciso"
    if plain.startswith("alinea "):
        return "alinea"
    return "body"


def split_legal_semantic_chunk(value: str, limit: int) -> list[str]:
    """Divide em limites jurídicos e preserva a ordem/todos os tokens."""
    source = re.sub(r"\s+", " ", str(value or "")).strip()
    if not source:
        return []
    # O chunking preventivo continua usando seus limites normais. No fallback
    # recursivo, porém, precisamos permitir microchunks menores quando o próprio
    # Chatterbox força EOS por token_repetition.
    limit = max(36, int(limit))

    primary = [
        item.strip()
        for item in re.split(r"(?<=[.!?;:])\s+", source)
        if item.strip()
    ]
    units: list[str] = []
    for item in primary:
        units.extend(_split_oversized_legal_unit(item, limit))

    chunks: list[str] = []
    current = ""
    current_kind = "body"
    for unit in units:
        unit_kind = _leading_legal_structure(unit)
        force_boundary = bool(current) and unit_kind in {
            "heading", "article", "paragraph", "inciso", "alinea",
        }
        candidate = f"{current} {unit}".strip()
        if force_boundary or (current and len(candidate) > limit):
            chunks.append(current)
            current = unit
            current_kind = unit_kind
            continue
        current = candidate
        if current_kind == "body":
            current_kind = unit_kind
    if current:
        chunks.append(current)

    return [item for item in chunks if is_valid_generation_chunk(item)]


def adapt_chunks_for_legal_complexity(
    chunks: list[tuple[str, bool]],
    maximum_chars: int,
) -> list[tuple[str, bool]]:
    """Reduz preventivamente somente blocos jurídicos médios/complexos."""
    output: list[tuple[str, bool]] = []
    for chunk, paragraph_end in chunks:
        complexity = legal_chunk_complexity(chunk)
        if complexity["level"] == "alta":
            adaptive_limit = min(int(maximum_chars), 165)
        elif complexity["level"] == "média":
            adaptive_limit = min(int(maximum_chars), 205)
        else:
            adaptive_limit = int(maximum_chars)

        if len(chunk) <= adaptive_limit:
            output.append((chunk, paragraph_end))
            continue

        parts = split_legal_semantic_chunk(chunk, adaptive_limit)
        if len(parts) <= 1:
            output.append((chunk, paragraph_end))
            continue
        validate_chunk_integrity(chunk, [(part, False) for part in parts])
        for index, part in enumerate(parts):
            output.append((
                part,
                paragraph_end if index == len(parts) - 1 else False,
            ))
        print(
            "[CTEC] Chunking jurídico adaptativo: "
            f"complexidade={complexity['level']} | score={complexity['score']} | "
            f"caracteres={len(chunk)} | limite={adaptive_limit} | "
            f"subchunks={len(parts)}",
            flush=True,
        )
    return output


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


def _reference_window_score(segment: torch.Tensor, sample_rate: int) -> float:
    """Pontua uma janela priorizando fala contínua, nível saudável e ausência de clipping."""
    if segment.ndim > 1:
        segment = segment.mean(dim=0)
    if segment.numel() == 0 or sample_rate <= 0:
        return -1e9

    rms = float(torch.sqrt(torch.mean(segment ** 2) + 1e-9))
    clipping_ratio = float((torch.abs(segment) >= 0.995).float().mean())

    frame = max(1, int(sample_rate * 0.03))
    usable = segment[: (segment.numel() // frame) * frame]
    if usable.numel() == 0:
        return -1e9

    energies = torch.sqrt(
        torch.mean(usable.reshape(-1, frame) ** 2, dim=1) + 1e-9
    )
    threshold = max(0.003, float(torch.median(energies)) * 0.22)
    silence_ratio = float((energies < threshold).float().mean())

    # Penaliza silêncio, clipping e volume muito baixo/alto.
    score = 100.0
    score -= min(70.0, silence_ratio * 85.0)
    score -= min(80.0, clipping_ratio * 12000.0)

    if rms < 0.008:
        score -= 45.0
    elif rms < 0.015:
        score -= 20.0
    elif rms > 0.35:
        score -= 25.0

    return score


def _select_reference_segment(
    decoded: Path,
    root: Path,
    *,
    target_seconds: float = 22.0,
    max_seconds: float = 30.0,
) -> tuple[Path, dict[str, Any]]:
    """
    Para referências longas, escolhe automaticamente uma janela de fala útil.
    O arquivo original/decodificado nunca é sobrescrito.
    """
    waveform, sample_rate = torchaudio.load(str(decoded))
    if waveform.numel() == 0 or sample_rate <= 0:
        raise ValueError("A amostra de voz está vazia ou corrompida.")

    mono = waveform.mean(dim=0)
    total_seconds = float(mono.numel() / sample_rate)

    if total_seconds < 3.0:
        raise ValueError(
            "A amostra precisa ter pelo menos 3 segundos; prefira de 15 a 30 segundos de fala limpa."
        )

    # Garante uma configuração coerente sem desfazer o conditioning curto do
    # voice_clone_fidelity_mode (10 s de alvo / 12 s de máximo).
    max_seconds = max(3.0, float(max_seconds))
    target_seconds = float(clamp(float(target_seconds), 3.0, max_seconds))
    minimum_segment_seconds = min(
        max_seconds,
        max(3.0, target_seconds * 0.90),
    )

    # Referências já curtas não precisam ser recortadas.
    if total_seconds <= max_seconds:
        print(
            "[CTEC] reference_target_seconds="
            f"{target_seconds:.2f} | reference_max_seconds={max_seconds:.2f} | "
            "reference_minimum_segment_seconds="
            f"{minimum_segment_seconds:.2f} | reference_windows_total=1 | "
            "reference_windows_scored=0 | reference_windows_rejected_duration=0 | "
            "best_reference_score=not_scored_short_reference | "
            "best_reference_start_sec=0.00",
            flush=True,
        )
        return decoded, {
            "reference_original_duration_sec": round(total_seconds, 2),
            "reference_selected_duration_sec": round(total_seconds, 2),
            "reference_segment_start_sec": 0.0,
            "reference_was_trimmed": False,
        }

    target_samples = max(1, int(target_seconds * sample_rate))
    minimum_segment_samples = max(1, int(minimum_segment_seconds * sample_rate))
    edge_guard_seconds = 5.0
    first_start = int(min(edge_guard_seconds, max(0.0, total_seconds * 0.05)) * sample_rate)
    last_start = max(first_start, mono.numel() - target_samples - int(edge_guard_seconds * sample_rate))

    # Avalia janelas a cada 5 s. Isso evita assumir que o melhor trecho está
    # no começo e reduz a chance de pegar vinheta, silêncio ou encerramento.
    step_samples = max(1, int(sample_rate * 5.0))
    starts = list(range(first_start, last_start + 1, step_samples))
    if last_start not in starts:
        starts.append(last_start)

    best_start = None
    best_score = -1e9
    windows_scored = 0
    windows_rejected_duration = 0

    for start in starts:
        end = min(mono.numel(), start + target_samples)
        segment = mono[start:end]
        if segment.numel() < minimum_segment_samples:
            windows_rejected_duration += 1
            continue
        score = _reference_window_score(segment, sample_rate)
        windows_scored += 1
        if score > best_score:
            best_score = score
            best_start = start

    best_score_log = f"{best_score:.2f}" if best_start is not None else "none"
    best_start_log = (
        f"{(best_start / sample_rate):.2f}" if best_start is not None else "none"
    )
    print(
        "[CTEC] reference_target_seconds="
        f"{target_seconds:.2f} | reference_max_seconds={max_seconds:.2f} | "
        "reference_minimum_segment_seconds="
        f"{minimum_segment_seconds:.2f} | reference_windows_total={len(starts)} | "
        f"reference_windows_scored={windows_scored} | "
        f"reference_windows_rejected_duration={windows_rejected_duration} | "
        f"best_reference_score={best_score_log} | "
        f"best_reference_start_sec={best_start_log}",
        flush=True,
    )

    if best_start is None:
        if windows_scored == 0:
            raise ValueError(
                "Nenhuma janela da referência atingiu a duração mínima necessária "
                f"para análise ({minimum_segment_seconds:.2f} s). "
                f"Janelas avaliadas: {len(starts)}; rejeitadas por duração: "
                f"{windows_rejected_duration}."
            )
        raise ValueError(
            "Nenhuma janela de referência pôde ser selecionada após a análise acústica."
        )

    selected = root / "reference_selected.wav"
    end = min(mono.numel(), best_start + target_samples)
    selected_waveform = mono[best_start:end].unsqueeze(0)
    torchaudio.save(str(selected), selected_waveform, sample_rate)

    selected_seconds = float(selected_waveform.shape[1] / sample_rate)
    if selected_seconds < 3.0:
        raise ValueError(
            "A janela selecionada da referência ficou abaixo de 3 segundos após o recorte."
        )

    return selected, {
        "reference_original_duration_sec": round(total_seconds, 2),
        "reference_selected_duration_sec": round(selected_seconds, 2),
        "reference_segment_start_sec": round(best_start / sample_rate, 2),
        "reference_selection_score": round(best_score, 2),
        "reference_was_trimmed": True,
    }


def prepare_reference_audio(
    source: Path,
    root: Path,
    settings: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """
    Decodifica e seleciona uma amostra curta sem alterar o arquivo original.

    No modo de clonagem fiel, preserva o espectro e a dinâmica da voz:
    converte apenas para mono/24 kHz/PCM, sem loudnorm, highpass ou lowpass.
    """
    decoded = root / "reference_decoded.wav"
    prepared = root / "reference_prepared.wav"

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
                "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(decoded),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "O áudio de referência está corrompido ou não pôde ser decodificado."
        ) from exc

    original_metrics = analyze_reference_audio(decoded)
    original_duration = float(original_metrics.get("durationSeconds") or 0)

    if original_duration < 3:
        raise ValueError(
            "A amostra precisa ter pelo menos 3 segundos; prefira de 15 a 30 segundos de fala limpa."
        )

    fidelity_mode = bool((settings or {}).get("voice_clone_fidelity_mode", True))

    # Chatterbox trabalha melhor quando o prompt de voz é curto e limpo.
    # O worker antigo extraía 22 s de uma referência de vários minutos; isso
    # criava um conditioning excessivamente longo e, nos logs CTEC, coincidia
    # com colapso de tokens até em frases de 4 palavras. Mantemos a voz, mas
    # condicionamos com uma janela curta e estável.
    selected_source, selection_metrics = _select_reference_segment(
        decoded,
        root,
        target_seconds=10.0 if fidelity_mode else 20.0,
        max_seconds=12.0 if fidelity_mode else 30.0,
    )

    try:
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(selected_source),
        ]

        if not fidelity_mode:
            command += [
                "-af",
                "highpass=f=60,lowpass=f=11500,"
                "loudnorm=I=-20:TP=-3:LRA=7",
            ]

        command += [
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(prepared),
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "A amostra selecionada não pôde ser preparada para clonagem."
        ) from exc

    prepared_metrics = analyze_reference_audio(prepared)

    # Mantém as métricas já usadas pelo restante do worker e acrescenta
    # os campos de auditoria solicitados.
    metrics = dict(prepared_metrics)
    metrics.update(selection_metrics)
    metrics["reference_original_path"] = str(source)
    metrics["reference_prepared_path"] = str(prepared)
    metrics["originalDurationSeconds"] = round(original_duration, 2)
    metrics["selectedDurationSeconds"] = selection_metrics[
        "reference_selected_duration_sec"
    ]
    metrics["wasTrimmed"] = selection_metrics["reference_was_trimmed"]
    metrics["voiceCloneFidelityMode"] = fidelity_mode
    metrics["referenceProcessing"] = (
        "format_only" if fidelity_mode else "filtered_and_normalized"
    )

    print(
        "[CTEC] reference_original_duration_sec="
        f"{metrics['reference_original_duration_sec']} | "
        "reference_selected_duration_sec="
        f"{metrics['reference_selected_duration_sec']} | "
        f"reference_original_path={metrics['reference_original_path']} | "
        f"reference_prepared_path={metrics['reference_prepared_path']} | "
        f"reference_was_trimmed={str(metrics['reference_was_trimmed']).lower()} | "
        f"voice_clone_fidelity_mode={str(fidelity_mode).lower()} | "
        f"reference_processing={metrics['referenceProcessing']}",
        flush=True,
    )

    return prepared, metrics


def resolve_settings(data: dict[str, Any]) -> dict[str, Any]:
    profile_name = str(data.get("profile") or "law_natural").strip().lower()
    base = dict(PROFILES.get(profile_name, PROFILES["law_natural"]))

    # A calibração aprovada chega do Flutter em um objeto aninhado. Antes,
    # resolve_settings() simplesmente ignorava esse objeto e a geração acabava
    # usando somente o preset. Agora o handler é a autoridade: ele aplica a
    # calibração somente quando ela foi realmente aprovada e depois passa todos
    # os valores pelo envelope de segurança abaixo. Assim não existe mais a
    # combinação "calibração diz uma coisa / handler usa outra".
    calibration = data.get("calibration")
    if isinstance(calibration, dict) and to_bool(
        calibration.get("calibrationApproved"), False
    ):
        calibration_map = {
            "stability": "stability",
            "voiceFidelity": "voice_fidelity",
            "temperature": "temperature",
            "cfgWeight": "cfg_weight",
            "commaPauseMs": "pause_comma_ms",
            "periodPauseMs": "pause_sentence_ms",
            "colonPauseMs": "pause_colon_ms",
            "paragraphPauseMs": "pause_paragraph_ms",
            "initialSilenceMs": "initial_silence_ms",
            "finalSilenceMs": "final_silence_ms",
            "maxChunkCharacters": "chunk_limit",
            "chunkOverlapWords": "chunk_overlap_words",
            "preserveCompleteSentences": "preserve_complete_sentences",
            "splitByLegalStructure": "split_by_legal_structure",
        }
        for source_key, target_key in calibration_map.items():
            if source_key in calibration:
                base[target_key] = calibration[source_key]

        print(
            f"[CTEC] approved_calibration=true | profile={profile_name} | "
            f"temperature={base.get('temperature')} | "
            f"cfg={base.get('cfg_weight')} | "
            f"stability={base.get('stability')} | "
            f"voice_fidelity={base.get('voice_fidelity')}",
            flush=True,
        )
    else:
        print(
            f"[CTEC] approved_calibration=false | profile={profile_name} | "
            "using_profile_defaults=true",
            flush=True,
        )

    base.setdefault("stability", 0.72)
    base.setdefault("voice_fidelity", 0.78)
    base.setdefault("pause_comma_ms", 250)
    base.setdefault("pause_colon_ms", 380)
    base.setdefault("pause_continuation_ms", 110)
    base.setdefault("initial_silence_ms", 180)
    base.setdefault("final_silence_ms", 260)
    base.setdefault("chunk_overlap_words", 0)
    base.setdefault("chunk_target", 260)
    base.setdefault("edge_silence_keep_ms", 45)
    base.setdefault("seam_fade_ms", 12)
    base.setdefault("preserve_complete_sentences", True)
    base.setdefault("split_by_legal_structure", True)

    # Consistência e integridade do áudio.
    base.setdefault("voice_consistency_mode", True)
    base.setdefault("voice_clone_fidelity_mode", True)
    base.setdefault("verify_each_chunk", True)
    base.setdefault("chunk_verify_threshold", 0.90)
    base.setdefault("chunk_verify_attempts", 3)
    base.setdefault("voice_seed", 1701)

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
        "pause_continuation_ms": (0, 600),
        "initial_silence_ms": (0, 2000),
        "final_silence_ms": (0, 3000),
        "chunk_overlap_words": (0, 20),
        "chunk_target": (120, 500),
        "edge_silence_keep_ms": (20, 120),
        "seam_fade_ms": (2, 30),
        "stability": (0.0, 1.0),
        "voice_fidelity": (0.0, 1.0),
        "chunk_verify_threshold": (0.70, 0.99),
        "chunk_verify_attempts": (1, 5),
        "voice_seed": (0, 2147483647),
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
        "voice_consistency_mode",
        "voice_clone_fidelity_mode",
        "verify_each_chunk",
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
        "pause_continuation_ms",
        "initial_silence_ms",
        "final_silence_ms",
        "chunk_overlap_words",
        "chunk_target",
        "edge_silence_keep_ms",
        "seam_fade_ms",
        "chunk_verify_attempts",
        "voice_seed",
    ):
        base[key] = int(base[key])
    base["chunk_target"] = min(base["chunk_target"], base["chunk_limit"])

    # IMPORTANTE: fidelidade da voz e sampling do T3 são coisas diferentes.
    # A versão anterior transformava "fidelidade" em temperature muito baixa
    # e CFG alto (<=0.56 / >=0.62). Isso afastava o motor do envelope padrão do
    # Chatterbox Multilingual e, nos logs CTEC, o modelo colapsava em repetição
    # de token. A fidelidade continua sendo preservada pelo áudio de referência;
    # não esmagamos mais os parâmetros de geração.
    if base.get("voice_clone_fidelity_mode"):
        base["stability"] = max(float(base.get("stability", 0.72)), 0.80)
        base["voice_fidelity"] = max(float(base.get("voice_fidelity", 0.78)), 0.88)

    elif base.get("voice_consistency_mode"):
        base["stability"] = max(float(base.get("stability", 0.72)), 0.78)
        base["voice_fidelity"] = max(float(base.get("voice_fidelity", 0.78)), 0.86)

    # Envelope seguro do Chatterbox Multilingual. O perfil law_natural já usa
    # os defaults próximos aos recomendados pelo projeto (temp~0.8, cfg~0.5).
    # Impedimos apenas extremos capazes de tornar o sampling rígido demais.
    if profile_name in {"law_natural", "law_formal"}:
        base["temperature"] = clamp(float(base.get("temperature", 0.78)), 0.70, 1.00)
        base["cfg_weight"] = clamp(float(base.get("cfg_weight", 0.50)), 0.30, 0.58)
        base["exaggeration"] = clamp(float(base.get("exaggeration", 0.48)), 0.30, 0.60)
        base["repetition_penalty"] = clamp(float(base.get("repetition_penalty", 1.20)), 1.15, 1.30)

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
        "continuationPauseMs": settings["pause_continuation_ms"],
        "initialSilenceMs": settings["initial_silence_ms"],
        "finalSilenceMs": settings["final_silence_ms"],
        "chunk_limit": settings["chunk_limit"],
        "chunkTarget": settings["chunk_target"],
        "chunkOverlapWords": settings["chunk_overlap_words"],
        "edgeSilenceKeepMs": settings["edge_silence_keep_ms"],
        "seamFadeMs": settings["seam_fade_ms"],
        "preserveCompleteSentences": settings["preserve_complete_sentences"],
        "splitByLegalStructure": settings["split_by_legal_structure"],
        "text_mode": settings["text_mode"],
        "normalize": settings["normalize"],
        "trim_silence": settings["trim_silence"],
        "pitch_semitones": settings["pitch_semitones"],
        "gain_db": settings["gain_db"],
        "voiceConsistencyMode": settings["voice_consistency_mode"],
        "voiceCloneFidelityMode": settings["voice_clone_fidelity_mode"],
        "verifyEachChunk": settings["verify_each_chunk"],
        "chunkVerifyThreshold": settings["chunk_verify_threshold"],
        "chunkVerifyAttempts": settings["chunk_verify_attempts"],
        "voiceSeed": settings["voice_seed"],
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


def probe_audio_duration(path: Path) -> float:
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


_VERIFICATION_NUMBER_WORDS = {
    "zero": "0",
    "um": "1", "uma": "1", "primeiro": "1", "primeira": "1",
    "dois": "2", "duas": "2", "segundo": "2", "segunda": "2",
    "três": "3", "tres": "3", "terceiro": "3", "terceira": "3",
    "quatro": "4", "quarto": "4", "quarta": "4",
    "cinco": "5", "quinto": "5", "quinta": "5",
    "seis": "6", "sexto": "6", "sexta": "6",
    "sete": "7", "sétimo": "7", "setimo": "7", "sétima": "7", "setima": "7",
    "oito": "8", "oitavo": "8", "oitava": "8",
    "nove": "9", "nono": "9", "nona": "9",
    "dez": "10", "décimo": "10", "decimo": "10", "décima": "10", "decima": "10",
}


def normalize_compare_text(value: str) -> str:
    value = str(value or "").lower()

    # O Whisper frequentemente transcreve "um" como "1", "quinto" como "5",
    # etc. Para verificação de conteúdo, essas formas são semanticamente iguais.
    value = re.sub(r"(\d+)\s*[º°ª]", r"\1", value)
    value = re.sub(r"[^\w\sáàâãéêíóôõúüç]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()

    tokens = []
    for token in value.split():
        canonical = _VERIFICATION_NUMBER_WORDS.get(token, token)
        tokens.append(canonical)

    return " ".join(tokens)


def transcription_similarity(expected: str, actual: str) -> float:
    a = normalize_compare_text(expected)
    b = normalize_compare_text(actual)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _roman_token_to_number(token: str) -> str:
    raw = str(token or "").strip().upper()
    if not raw or not re.fullmatch(
        r"M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})",
        raw,
    ):
        return str(token or "")
    value = roman_to_int(raw)
    return str(value) if value > 0 else str(token or "")


_LEGAL_HEAD_CANONICAL = {
    "art": "artigo",
    "artigo": "artigo",
    "paragrafo": "paragrafo",
    "inciso": "inciso",
    "alinea": "alinea",
    "item": "item",
    "titulo": "titulo",
    "capitulo": "capitulo",
    "secao": "secao",
    "subsecao": "subsecao",
    "livro": "livro",
    "parte": "parte",
}

_LEGAL_NUMBER_COMPONENTS = {
    "zero": 0,
    "um": 1, "uma": 1, "primeiro": 1, "primeira": 1,
    "dois": 2, "duas": 2, "segundo": 2, "segunda": 2,
    "tres": 3, "terceiro": 3, "terceira": 3,
    "quatro": 4, "quarto": 4, "quarta": 4,
    "cinco": 5, "quinto": 5, "quinta": 5,
    "seis": 6, "sexto": 6, "sexta": 6,
    "sete": 7, "setimo": 7, "setima": 7,
    "oito": 8, "oitavo": 8, "oitava": 8,
    "nove": 9, "nono": 9, "nona": 9,
    "dez": 10, "decimo": 10, "decima": 10,
    "onze": 11, "decimo-primeiro": 11, "decima-primeira": 11,
    "doze": 12, "decimo-segundo": 12, "decima-segunda": 12,
    "treze": 13, "decimo-terceiro": 13, "decima-terceira": 13,
    "quatorze": 14, "catorze": 14,
    "quinze": 15,
    "dezesseis": 16, "dezasseis": 16,
    "dezessete": 17, "dezassete": 17,
    "dezoito": 18,
    "dezenove": 19,
    "vinte": 20, "vigesimo": 20, "vigesima": 20,
    "trinta": 30, "trigesimo": 30, "trigesima": 30,
    "quarenta": 40, "quadragesimo": 40, "quadragesima": 40,
    "cinquenta": 50, "quinquagesimo": 50, "quinquagesima": 50,
    "sessenta": 60, "sexagesimo": 60, "sexagesima": 60,
    "setenta": 70, "septuagesimo": 70, "septuagesima": 70,
    "oitenta": 80, "octogesimo": 80, "octogesima": 80,
    "noventa": 90, "nonagesimo": 90, "nonagesima": 90,
    "cem": 100, "cento": 100, "centesimo": 100, "centesima": 100,
    "duzentos": 200, "duzentas": 200, "ducentesimo": 200, "ducentesima": 200,
    "trezentos": 300, "trezentas": 300, "trecentesimo": 300, "trecentesima": 300,
    "quatrocentos": 400, "quatrocentas": 400, "quadringentesimo": 400,
    "quadringentesima": 400,
    "quinhentos": 500, "quinhentas": 500, "quingentesimo": 500,
    "quingentesima": 500,
    "seiscentos": 600, "seiscentas": 600, "sexcentesimo": 600,
    "sexcentesima": 600,
    "setecentos": 700, "setecentas": 700, "septingentesimo": 700,
    "septingentesima": 700,
    "oitocentos": 800, "oitocentas": 800, "octingentesimo": 800,
    "octingentesima": 800,
    "novecentos": 900, "novecentas": 900, "nongentesimo": 900,
    "nongentesima": 900,
}


def _verification_plain_token(value: str) -> str:
    raw = re.sub(
        r"(\d+)[º°ª]",
        r"\1",
        str(value or "").lower(),
    )
    normalized = unicodedata.normalize("NFKD", raw)
    normalized = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    return normalized.strip()


def _verification_display_tokens(value: str) -> list[tuple[str, str]]:
    # NFC preserva º/ª como símbolos. NFKC os transformaria em letras soltas
    # ("1º" -> "1o"), criando um token extra artificial.
    source = unicodedata.normalize("NFC", str(value or "")).lower()
    raw_tokens = re.findall(
        r"§|\d+[º°ª]?|[a-zA-ZÀ-ÖØ-öø-ÿ]+",
        source,
        flags=re.UNICODE,
    )
    output: list[tuple[str, str]] = []
    for raw in raw_tokens:
        plain = "paragrafo" if raw == "§" else _verification_plain_token(raw)
        if plain:
            output.append((raw, plain))
    return output


def _consume_legal_number(
    tokens: list[tuple[str, str]],
    start: int,
) -> tuple[str | None, int]:
    if start >= len(tokens):
        return None, start

    token = tokens[start][1]
    if token.isdigit():
        return str(int(token)), start + 1

    roman = _roman_token_to_number(token)
    if roman != token and roman.isdigit():
        return roman, start + 1

    total = 0
    consumed = 0
    index = start
    while index < len(tokens):
        current = tokens[index][1]
        if current == "e" and consumed:
            if (
                index + 1 < len(tokens)
                and tokens[index + 1][1] in (
                    set(_LEGAL_NUMBER_COMPONENTS) | {"mil"}
                )
            ):
                index += 1
                continue
            break
        if current == "mil":
            total = max(1, total) * 1000
            consumed += 1
            index += 1
            continue
        component = _LEGAL_NUMBER_COMPONENTS.get(current)
        if component is None:
            break
        total += component
        consumed += 1
        index += 1

    if not consumed:
        return None, start
    return str(total), index


def _normalize_legal_validation(value: str) -> dict[str, Any]:
    """Normaliza somente a cópia usada para validar a transcrição."""
    tokens = _verification_display_tokens(value)
    normalized: list[str] = []
    references: list[dict[str, str]] = []
    index = 0

    while index < len(tokens):
        raw, plain = tokens[index]
        head = _LEGAL_HEAD_CANONICAL.get(plain)
        if head is None:
            normalized.append(plain)
            index += 1
            continue

        start = index
        index += 1
        canonical_value: str | None = None

        if index < len(tokens) and tokens[index][1] in {"unico", "unica"}:
            canonical_value = "unico"
            index += 1
        elif head == "alinea" and index < len(tokens):
            possible_letter = tokens[index][1]
            if re.fullmatch(r"[a-z]", possible_letter):
                canonical_value = possible_letter
                index += 1
        else:
            canonical_value, consumed_until = _consume_legal_number(tokens, index)
            if canonical_value is not None:
                index = consumed_until

        normalized.append(head)
        if canonical_value is not None:
            normalized.append(canonical_value)
            original = " ".join(item[0] for item in tokens[start:index])
            references.append({
                "original": original,
                "canonical": f"{head} {canonical_value}",
            })
        elif plain != head:
            references.append({
                "original": raw,
                "canonical": head,
            })

    return {
        "normalized_text": " ".join(normalized),
        "tokens": normalized,
        "references": references,
    }


def normalize_legal_verification_text(value: str) -> str:
    """
    Normalização contextual usada apenas para comparar TTS com Whisper.
    O texto narrado, salvo e exibido nunca passa por esta função.
    """
    return str(_normalize_legal_validation(value)["normalized_text"])


def _asr_phonetic_span_equivalence(
    expected_tokens: list[str],
    actual_tokens: list[str],
    global_similarity: float,
) -> dict[str, Any] | None:
    """Reconcilia apenas segmentações fonéticas muito próximas feitas pelo ASR."""
    if global_similarity < 0.94:
        return None
    if not expected_tokens or not actual_tokens:
        return None
    if len(expected_tokens) > 2 or len(actual_tokens) > 2:
        return None
    if len(expected_tokens) == len(actual_tokens) == 1:
        return None
    if any(
        re.search(r"\d", token)
        for token in expected_tokens + actual_tokens
    ):
        return None

    expected_joined = "".join(expected_tokens)
    actual_joined = "".join(actual_tokens)
    if min(len(expected_joined), len(actual_joined)) < 8:
        return None
    if abs(len(expected_joined) - len(actual_joined)) > 3:
        return None

    phonetic_similarity = difflib.SequenceMatcher(
        None,
        expected_joined,
        actual_joined,
        autojunk=False,
    ).ratio()
    if phonetic_similarity < 0.92:
        return None

    return {
        "esperado": " ".join(expected_tokens),
        "reconhecido": " ".join(actual_tokens),
        "similaridade_fonetica": round(phonetic_similarity, 3),
        "resultado": "EQUIVALENTE_FONETICO_ASR",
    }


def _transcription_token_alignment(
    expected_tokens: list[str],
    actual_tokens: list[str],
    global_similarity: float,
) -> dict[str, Any]:
    matcher = difflib.SequenceMatcher(
        None,
        expected_tokens,
        actual_tokens,
        autojunk=False,
    )
    exact_matches = 0
    adjusted_matches = 0
    missing_tokens: list[str] = []
    extra_tokens: list[str] = []
    phonetic_equivalences: list[dict[str, Any]] = []

    for tag, expected_start, expected_end, actual_start, actual_end in (
        matcher.get_opcodes()
    ):
        expected_span = expected_tokens[expected_start:expected_end]
        actual_span = actual_tokens[actual_start:actual_end]
        if tag == "equal":
            exact_matches += len(expected_span)
            adjusted_matches += len(expected_span)
            continue
        if tag == "replace":
            equivalence = _asr_phonetic_span_equivalence(
                expected_span,
                actual_span,
                global_similarity,
            )
            if equivalence is not None:
                adjusted_matches += len(expected_span)
                phonetic_equivalences.append(equivalence)
                continue
        missing_tokens.extend(expected_span)
        extra_tokens.extend(actual_span)

    denominator = max(1, len(expected_tokens))
    return {
        "raw_recall": exact_matches / denominator,
        "recall": adjusted_matches / denominator,
        "missing_tokens": missing_tokens,
        "extra_tokens": extra_tokens,
        "phonetic_equivalences": phonetic_equivalences,
    }


def _normalized_legal_divergences(
    expected_references: list[dict[str, str]],
    actual_references: list[dict[str, str]],
) -> list[dict[str, str]]:
    divergences: list[dict[str, str]] = []
    used_actual: set[int] = set()
    for expected in expected_references:
        for actual_index, actual in enumerate(actual_references):
            if actual_index in used_actual:
                continue
            if expected["canonical"] != actual["canonical"]:
                continue
            used_actual.add(actual_index)
            expected_form = _verification_plain_token(expected["original"])
            actual_form = _verification_plain_token(actual["original"])
            if expected_form != actual_form:
                divergences.append({
                    "esperado": expected["original"],
                    "reconhecido": actual["original"],
                    "normalizado": expected["canonical"],
                    "resultado": "EQUIVALENTE",
                })
            break
    return divergences


def validate_legal_transcription(
    expected: str,
    actual: str,
    threshold: float,
) -> dict[str, Any]:
    expected_result = _normalize_legal_validation(expected)
    actual_result = _normalize_legal_validation(actual)
    expected_text = str(expected_result["normalized_text"])
    actual_text = str(actual_result["normalized_text"])
    expected_tokens = list(expected_result["tokens"])
    actual_tokens = list(actual_result["tokens"])

    if expected_text and actual_text:
        similarity = difflib.SequenceMatcher(
            None,
            expected_text,
            actual_text,
            autojunk=False,
        ).ratio()
    else:
        similarity = 0.0
    alignment = _transcription_token_alignment(
        expected_tokens,
        actual_tokens,
        similarity,
    )
    recall = float(alignment["recall"])
    raw_recall = float(alignment["raw_recall"])
    missing_tokens = list(alignment["missing_tokens"])
    extra_tokens = list(alignment["extra_tokens"])
    normalized_divergences = _normalized_legal_divergences(
        list(expected_result["references"]),
        list(actual_result["references"]),
    )
    normalized_divergences.extend(alignment["phonetic_equivalences"])
    similarity_threshold = max(0.78, float(threshold) - 0.08)
    critical_missing_tokens = [
        token for token in missing_tokens
        if token.isdigit() or len(token) >= 5
    ]
    approved = (
        recall >= float(threshold)
        and similarity >= similarity_threshold
        and not critical_missing_tokens
    )

    return {
        "approved": approved,
        "similarity": similarity,
        "recall": recall,
        "raw_recall": raw_recall,
        "similarity_threshold": similarity_threshold,
        "recall_threshold": float(threshold),
        "expected_normalized": expected_text,
        "recognized_normalized": actual_text,
        "missing_expected_tokens": missing_tokens,
        "critical_missing_tokens": critical_missing_tokens,
        "extra_recognized_tokens": extra_tokens,
        "normalized_divergences": normalized_divergences,
        "equivalent_representation": bool(normalized_divergences),
        "material_omission": bool(critical_missing_tokens) or (
            bool(missing_tokens) and recall < float(threshold)
        ),
    }


def legal_short_chunk_equivalent(expected: str, actual: str) -> bool:
    """
    Aceita diferenças apenas de representação numérica em estruturas jurídicas
    curtas, por exemplo:
      "Inciso um." == "Inciso 1."
      "Artigo quinto." == "Artigo 5."
      "Parágrafo primeiro." == "Parágrafo 1."
    """
    a = normalize_legal_verification_text(expected)
    b = normalize_legal_verification_text(actual)
    if not a or not b:
        return False

    if a == b:
        return True

    return a == b


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
    # A calibração já possui sua própria etapa de transcrição e pontuação
    # logo abaixo. Não deve usar a verificação rígida por chunk, pois isso
    # pode impedir a calibração de concluir antes de comparar os candidatos.
    calibration_settings = dict(settings)
    calibration_settings["verify_each_chunk"] = False

    audio = generate_chunk_with_retry(
        model,
        text,
        language_id="pt",
        reference_path=reference_path,
        settings=calibration_settings,
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

    # A calibração nunca mais testa combinações fora do envelope do perfil.
    # Antes A/B usavam temperature 0.46/0.54 e CFG 0.68/0.60; isso podia
    # provocar exatamente o comportamento de EOS precoce por repetição que
    # apareceu nos logs. Os candidatos abaixo ficam dentro da faixa segura e
    # variam apenas o suficiente para a calibração comparar naturalidade.
    candidates = [
        {
            "id": "candidate_a",
            "name": "A",
            "exaggeration": 0.36,
            "cfg_weight": 0.50,
            "temperature": 0.78,
            "speed": 0.96,
            "stability": 0.84,
        },
        {
            "id": "candidate_b",
            "name": "B",
            "exaggeration": 0.44,
            "cfg_weight": 0.44,
            "temperature": 0.86,
            "speed": 0.99,
            "stability": 0.80,
        },
    ]

    with _GENERATION_LOCK, tempfile.TemporaryDirectory(prefix="ctec_calibration_") as tmp:
        root = Path(tmp)
        reference_source = save_reference_audio(data, root)
        reference_path, reference_metrics = prepare_reference_audio(
            reference_source,
            root,
            base,
        )
        model = get_model()
        results = []

        for index, candidate in enumerate(candidates, start=1):
            runpod.serverless.progress_update(
                job, f"Calibração automática: teste {index} de {len(candidates)}"
            )
            try:
                settings = dict(base)
                settings.update(candidate)
                # Reaplica as mesmas regras do handler depois do A/B. O candidato
                # não pode escapar do envelope só porque foi criado pela própria
                # rotina de calibração.
                settings = resolve_settings({
                    "profile": target_profile,
                    **{
                        "speed": settings.get("speed"),
                        "exaggeration": settings.get("exaggeration"),
                        "cfg_weight": settings.get("cfg_weight"),
                        "temperature": settings.get("temperature"),
                        "stability": settings.get("stability"),
                        "repetition_penalty": settings.get("repetition_penalty"),
                    },
                })
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
                # Não deixa uma amostra com transcrição claramente incompleta
                # ser escolhida apenas porque teve ritmo parecido.
                if completeness < 0.80:
                    score -= (0.80 - completeness) * 100.0
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
            except Exception as candidate_error:
                print(
                    f"[CTEC] calibração candidato {candidate['name']} falhou: "
                    f"{candidate_error!r}; tentando o próximo candidato.",
                    flush=True,
                )

        if not results:
            raise RuntimeError(
                "A calibração automática não conseguiu gerar nenhum candidato "
                "dentro do envelope seguro do perfil."
            )

        best = max(results, key=lambda item: item["score"])
        best_settings = dict(best["settings"])
        best_settings.update({
            "stability": best["settings"].get("stability", 0.80),
            "voiceFidelity": round(clamp(
                reference_metrics["qualityScore"] / 100.0, 0.55, 0.95
            ), 3),
            "cfgWeight": best_settings["cfg_weight"],
            "commaPauseMs": 110 if target_profile != "law_formal" else 130,
            "periodPauseMs": 220 if target_profile != "law_formal" else 260,
            "paragraphPauseMs": 430 if target_profile != "law_formal" else 520,
            "continuationPauseMs": 100,
            "maxChunkCharacters": 300,
            "chunkTargetCharacters": 260,
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
            "calibrationPolicy": "safe_ab_v2",
            "candidates": results,
        }


def capabilities() -> dict[str, Any]:
    loader_signature = inspect.signature(
        ChatterboxMultilingualTTS.from_pretrained
    )
    supports_explicit_model = "t3_model" in loader_signature.parameters
    generate_signature = inspect.signature(ChatterboxMultilingualTTS.generate)
    generate_parameters = sorted(generate_signature.parameters)
    analyzer_guard_parameters = {
        "token_repetition_threshold",
        "alignment_repetition_threshold",
        "stop_on_eos",
        "stopping_criteria",
    }
    effective_model = (
        _LOADED_MODEL_VERSION
        or (MODEL_VERSION if supports_explicit_model else "default-compatible")
    )
    return {
        "status": "ok",
        "worker": "CTEC Estúdio de Voz",
        "version": "5.4.6",
        "contract_version": WORKER_CONTRACT_VERSION,
        "device": DEVICE,
        "model": f"Chatterbox Multilingual {effective_model}",
        "requested_model_version": MODEL_VERSION,
        "explicit_model_selection": supports_explicit_model,
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
        "semantic_chunks": True,
        "adaptive_pauses": True,
        "pcm_assembly": True,
        "single_final_encoding": True,
        "chunk_text_integrity": True,
        "per_chunk_whisper_verification": WhisperModel is not None,
        "voice_consistency_mode": True,
        "voice_clone_fidelity_mode": True,
        "reference_identity_preservation": True,
        "whisper_roman_heading_equivalence": True,
        "punctuation_prosody_engine": True,
        "ui_pause_controls_applied_inside_chunks": True,
        "calibration_uses_own_scoring": True,
        "whisper_numeric_equivalence": True,
        "contextual_legal_validation": True,
        "validation_token_diagnostics": True,
        "adaptive_incomplete_chunk_fallback": True,
        "recursive_semantic_fallback": True,
        "legal_complexity_chunking": True,
        "approved_subchunk_temporary_cache": True,
        "generation_diagnostics_v2": True,
        "chatterbox_public_generate_parameters": generate_parameters,
        "token_repetition_guard_configurable": bool(
            analyzer_guard_parameters.intersection(generate_parameters)
        ),
        "chatterbox_log_capture": True,
        "early_eos_duration_detection": True,
        "token_repetition_retry_strategy": True,
        "failure_reason_classification": True,
        "chatterbox_internal_guard_modified": False,
        "adaptive_fallback_max_depth": MAX_ADAPTIVE_FALLBACK_DEPTH,
        "adaptive_fallback_min_chars": int(clamp(
            MIN_TTS_SUBCHUNK_CHARS,
            60,
            110,
        )),
        "safe_tts_fallback_chars": int(clamp(
            SAFE_TTS_FALLBACK_CHARS,
            120,
            220,
        )),
        "spoken_roman_legal_headings": True,
        "guarded_asr_phonetic_alignment": True,
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
    return int(settings.get("pause_continuation_ms", 110))


def prepare_audio_for_join(
    audio: torch.Tensor,
    sample_rate: int,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apara apenas excesso de silêncio e protege os fonemas das bordas."""
    audio = audio.detach().cpu().float()
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise RuntimeError(f"Formato inesperado do áudio gerado: {tuple(audio.shape)}")
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = audio.clamp(-1.0, 1.0).contiguous()

    if audio.shape[1] < max(1, int(sample_rate * 0.08)):
        raise RuntimeError("O modelo devolveu um trecho curto demais para ser válido.")

    samples = audio[0].abs()
    peak = float(samples.max().item())
    if peak < 1e-4:
        raise RuntimeError("O modelo devolveu um trecho sem fala audível.")
    threshold = max(0.0015, peak * 0.012)
    active = torch.nonzero(samples > threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        raise RuntimeError("Não foi possível localizar fala no trecho gerado.")

    leading_samples = int(active[0].item())
    trailing_samples = int(samples.numel() - 1 - active[-1].item())
    keep_samples = max(
        1,
        int(sample_rate * int(settings.get("edge_silence_keep_ms", 45)) / 1000),
    )
    cut_start = max(0, leading_samples - keep_samples)
    cut_end = max(0, trailing_samples - keep_samples)
    end_index = audio.shape[1] - cut_end if cut_end else audio.shape[1]
    if end_index - cut_start >= int(sample_rate * 0.08):
        audio = audio[:, cut_start:end_index].contiguous()

    samples = audio[0].abs()
    active = torch.nonzero(samples > threshold, as_tuple=False).flatten()
    leading_samples = int(active[0].item())
    trailing_samples = int(samples.numel() - 1 - active[-1].item())

    configured_fade = max(
        1,
        int(sample_rate * int(settings.get("seam_fade_ms", 12)) / 1000),
    )
    emergency_fade = max(1, int(sample_rate * 0.003))
    fade_in = min(
        configured_fade if leading_samples >= configured_fade else emergency_fade,
        audio.shape[1] // 2,
    )
    fade_out = min(
        configured_fade if trailing_samples >= configured_fade else emergency_fade,
        audio.shape[1] // 2,
    )
    if fade_in > 0:
        audio[:, :fade_in] *= torch.linspace(0.0, 1.0, fade_in)
    if fade_out > 0:
        audio[:, -fade_out:] *= torch.linspace(1.0, 0.0, fade_out)

    return audio, {
        "leadingSilenceMs": leading_samples * 1000.0 / sample_rate,
        "trailingSilenceMs": trailing_samples * 1000.0 / sample_rate,
        "trimmedStartMs": cut_start * 1000.0 / sample_rate,
        "trimmedEndMs": cut_end * 1000.0 / sample_rate,
        "peak": peak,
    }


class ContinuousWaveAssembler:
    """Monta um único WAV/PCM progressivo, sem MP3 entre os blocos."""

    def __init__(
        self,
        path: Path,
        sample_rate: int,
        settings: dict[str, Any],
    ) -> None:
        self.path = path
        self.sample_rate = int(sample_rate)
        self.settings = settings
        self.total_samples = 0
        self.chunk_count = 0
        self.previous_chunk = ""
        self.previous_paragraph_end = False
        self.previous_trailing_ms = 0.0
        self._closed = False
        self._writer = wave.open(str(path), "wb")
        self._writer.setnchannels(1)
        self._writer.setsampwidth(2)
        self._writer.setframerate(self.sample_rate)

    def _write_silence(self, milliseconds: float) -> int:
        sample_count = max(0, int(round(self.sample_rate * milliseconds / 1000.0)))
        if sample_count:
            self._writer.writeframesraw(b"\x00\x00" * sample_count)
            self.total_samples += sample_count
        return sample_count

    def _write_audio(self, audio: torch.Tensor) -> int:
        pcm = (audio[0].clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)
        self._writer.writeframesraw(pcm.contiguous().numpy().tobytes())
        sample_count = int(pcm.numel())
        self.total_samples += sample_count
        return sample_count

    def add(
        self,
        audio: torch.Tensor,
        chunk: str,
        paragraph_end: bool,
    ) -> dict[str, float]:
        prepared, metrics = prepare_audio_for_join(
            audio,
            self.sample_rate,
            self.settings,
        )
        leading_ms = metrics["leadingSilenceMs"]
        if self.chunk_count == 0:
            desired_pause_ms = float(self.settings.get("initial_silence_ms", 0))
            existing_pause_ms = leading_ms
        else:
            desired_pause_ms = float(pause_for_chunk(
                self.previous_chunk,
                self.previous_paragraph_end,
                self.settings,
            ))
            existing_pause_ms = self.previous_trailing_ms + leading_ms

        inserted_pause_ms = max(0.0, desired_pause_ms - existing_pause_ms)
        self._write_silence(inserted_pause_ms)
        start_sample = self.total_samples
        audio_samples = self._write_audio(prepared)

        self.chunk_count += 1
        self.previous_chunk = chunk
        self.previous_paragraph_end = paragraph_end
        self.previous_trailing_ms = metrics["trailingSilenceMs"]
        metrics.update({
            "insertedPauseMs": inserted_pause_ms,
            "startSample": float(start_sample),
            "audioSamples": float(audio_samples),
        })
        return metrics

    def close(self, add_final_silence: bool = True) -> None:
        if self._closed:
            return
        if add_final_silence and self.chunk_count:
            desired_ms = float(self.settings.get("final_silence_ms", 0))
            self._write_silence(max(0.0, desired_ms - self.previous_trailing_ms))
        self._writer.close()
        self._closed = True

    def __enter__(self) -> "ContinuousWaveAssembler":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(add_final_silence=exc_type is None)



def split_chunk_for_prosody(value: str) -> list[tuple[str, str]]:
    """
    Divide um chunk somente em pontos naturais de prosódia.
    Retorna (texto_da_unidade, pontuacao_final).
    A pontuação permanece no texto enviado ao TTS; ela não é narrada.
    """
    value = str(value or "").strip()
    if not value:
        return []

    # Mantém o delimitador para sabermos qual pausa aplicar depois.
    parts = re.split(r"(?<=[,;:.!?])\s+", value)
    units: list[tuple[str, str]] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        punctuation = part[-1] if part[-1] in ",;:.!?" else ""
        units.append((part, punctuation))

    return units


def pause_for_punctuation(
    punctuation: str,
    *,
    paragraph_end: bool,
    settings: dict[str, Any],
) -> int:
    """
    Usa exatamente os controles já existentes na interface:
    vírgula, ponto, dois-pontos e parágrafo.
    """
    if paragraph_end:
        return int(settings["pause_paragraph_ms"])

    if punctuation == ",":
        return int(settings["pause_comma_ms"])

    if punctuation == ":":
        return int(settings["pause_colon_ms"])

    if punctuation == ";":
        # Não há slider específico para ponto e vírgula.
        # Usa uma pausa intermediária entre vírgula e dois-pontos.
        return int(round(
            (int(settings["pause_comma_ms"]) + int(settings["pause_colon_ms"])) / 2
        ))

    if punctuation in {".", "!", "?"}:
        return int(settings["pause_sentence_ms"])

    return 0


def prosody_units_for_chunk(
    chunk: str,
    paragraph_end: bool,
    settings: dict[str, Any],
) -> list[tuple[str, int]]:
    """
    Converte um chunk em unidades narráveis e associa a pausa configurada
    após cada unidade. A última unidade respeita a pausa de parágrafo.
    """
    units = split_chunk_for_prosody(chunk)
    if not units:
        return []

    output: list[tuple[str, int]] = []
    for index, (unit_text, punctuation) in enumerate(units):
        is_last = index == len(units) - 1
        pause_ms = pause_for_punctuation(
            punctuation,
            paragraph_end=paragraph_end and is_last,
            settings=settings,
        )
        output.append((unit_text, pause_ms))

    return output


def is_valid_generation_chunk(value: str) -> bool:
    cleaned = clean_generation_chunk(value)
    # Um único algarismo/letra pode ser conteúdo jurídico válido.
    return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", cleaned))


def merge_tiny_chunks(
    chunks: list[tuple[str, bool]],
    minimum_chars: int = 70,
    maximum_chars: int = 320,
) -> list[tuple[str, bool]]:
    """
    Junta trechos pequenos sem descartar conteúdo textual.
    Pontuação isolada pode ser ignorada; qualquer trecho com letra/número é preservado.
    """
    merged: list[tuple[str, bool]] = []
    pending = ""
    pending_end = False

    for raw_chunk, paragraph_end in chunks:
        chunk = clean_generation_chunk(raw_chunk)

        # Só ignora fragmentos realmente sem conteúdo lexical.
        if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", chunk):
            continue

        if len(chunk) < minimum_chars:
            if merged:
                previous, previous_end = merged[-1]
                candidate = f"{previous} {chunk}".strip()
                if len(candidate) <= maximum_chars:
                    merged[-1] = (candidate, paragraph_end or previous_end)
                    continue
            candidate = f"{pending} {chunk}".strip()
            if pending and len(candidate) > maximum_chars:
                merged.append((pending, pending_end))
                pending = ""
            pending = f"{pending} {chunk}".strip()
            pending_end = paragraph_end
            continue

        if pending:
            candidate = f"{pending} {chunk}".strip()
            if len(candidate) <= maximum_chars:
                chunk = candidate
                paragraph_end = paragraph_end or pending_end
            else:
                merged.append((pending, pending_end))
            pending = ""
            pending_end = False

        merged.append((chunk, paragraph_end))

    if pending:
        if merged:
            previous, paragraph_end = merged.pop()
            candidate = f"{previous} {pending}".strip()
            if len(candidate) <= maximum_chars:
                merged.append((candidate, paragraph_end or pending_end))
            else:
                merged.append((previous, paragraph_end))
                merged.append((pending, pending_end))
        else:
            merged.append((pending, pending_end))

    return merged


def _integrity_tokens(value: str) -> list[str]:
    normalized = normalize_compare_text(value)
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", normalized, flags=re.UNICODE)


def validate_chunk_integrity(
    prepared_text: str,
    chunks: list[tuple[str, bool]],
) -> dict[str, Any]:
    """
    Confere que a divisão em chunks não perdeu palavras.
    A comparação ignora apenas pontuação e espaços.
    """
    expected = _integrity_tokens(prepared_text)
    reconstructed = _integrity_tokens(
        " ".join(chunk for chunk, _ in chunks)
    )

    ok = expected == reconstructed
    missing_text = ""

    if not ok:
        matcher = difflib.SequenceMatcher(
            None,
            expected,
            reconstructed,
            autojunk=False,
        )
        first_bad = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                first_bad = i1
                break
        missing_text = " ".join(expected[first_bad:first_bad + 30])

    result = {
        "original_text_chars": len(prepared_text),
        "chunk_text_chars": sum(len(chunk) for chunk, _ in chunks),
        "prepared_token_count": len(expected),
        "chunk_token_count": len(reconstructed),
        "text_integrity_ok": ok,
        "missing_text": missing_text,
    }

    print(
        "[CTEC] Integridade de chunks: "
        f"text_integrity_ok={str(ok).lower()} | "
        f"prepared_tokens={len(expected)} | "
        f"chunk_tokens={len(reconstructed)} | "
        f"missing_text={missing_text[:180]!r}",
        flush=True,
    )

    if not ok:
        raise RuntimeError(
            "A divisão do texto em trechos perdeu ou alterou conteúdo. "
            f"Primeiro conteúdo divergente: {missing_text[:220]!r}"
        )

    return result


def transcription_word_recall(expected: str, actual: str) -> float:
    expected_words = normalize_compare_text(expected).split()
    actual_words = normalize_compare_text(actual).split()
    if not expected_words or not actual_words:
        return 0.0

    matcher = difflib.SequenceMatcher(
        None,
        expected_words,
        actual_words,
        autojunk=False,
    )
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / max(1, len(expected_words))


def reference_path_hash(path: Path) -> str:
    """Hash curto do arquivo de referência preparado usado em todos os chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _set_generation_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _split_incomplete_chunk_for_fallback(
    value: str,
    depth: int = 0,
) -> list[str]:
    """
    Subdivide recursivamente em limites jurídicos, sem perder palavras.

    Correção CTEC:
    - um chunk que falhou por EOS não deixa de ser subdividido só porque já é
      menor que SAFE_TTS_FALLBACK_CHARS;
    - cada profundidade reduz de fato o alvo;
    - referências legais e datas são preservadas;
    - citações legais entre parênteses recebem fronteiras naturais.
    """
    source = re.sub(r"\s+", " ", str(value or "")).strip()
    if not source:
        return []

    depth = max(0, int(depth))
    base_limit = int(clamp(SAFE_TTS_FALLBACK_CHARS, 100, 220))
    hard_floor = max(36, int(MIN_TTS_SUBCHUNK_CHARS) // 2)

    safe_limit = max(
        hard_floor,
        int(round(base_limit * (0.75 ** depth))),
    )

    # Caso real: 131 caracteres com fallback=140 retornava uma única parte.
    # Se o trecho chegou ao fallback, ele já falhou e precisa encolher.
    if len(source) <= safe_limit and len(source) > hard_floor:
        safe_limit = max(
            hard_floor,
            min(safe_limit, int(round(len(source) * 0.62))),
        )

    semantic_seed_parts: list[str] = []
    cursor = 0
    for match in re.finditer(
        r"\((\s*Lei\s+(?:Complementar\s+)?número\s+[^()]+)\)",
        source,
        flags=re.IGNORECASE,
    ):
        prefix = source[cursor:match.start()].strip(" ,;:")
        if prefix:
            semantic_seed_parts.append(prefix)

        inside = match.group(1).strip(" ,;:")
        legal_bits = [
            bit.strip(" ,;:")
            for bit in re.split(r",\s+(?=de\s+)", inside, maxsplit=1)
            if bit.strip(" ,;:")
        ]
        semantic_seed_parts.extend(legal_bits)
        cursor = match.end()

    tail = source[cursor:].strip(" ,;:")
    if tail:
        semantic_seed_parts.append(tail)

    if len(semantic_seed_parts) <= 1:
        semantic_seed_parts = [source]

    parts: list[str] = []
    for seed_part in semantic_seed_parts:
        if len(seed_part) <= safe_limit:
            parts.append(seed_part)
        else:
            parts.extend(split_legal_semantic_chunk(seed_part, safe_limit))

    parts = [
        clean_generation_chunk(part)
        for part in parts
        if is_valid_generation_chunk(part)
    ]

    # Se a divisão semântica ainda não reduziu, força progresso com limite menor,
    # sempre respeitando os spans de referências jurídicas.
    if len(parts) <= 1 and len(source) > hard_floor:
        forced_limit = max(
            hard_floor,
            min(len(source) - 1, int(round(len(source) * 0.55))),
        )
        forced = _split_oversized_legal_unit(source, forced_limit)
        parts = [
            clean_generation_chunk(part)
            for part in forced
            if is_valid_generation_chunk(part)
        ]

    if len(parts) <= 1:
        return []

    validate_chunk_integrity(source, [(part, False) for part in parts])
    return parts

def _fallback_join_pause_ms(
    next_text: str,
    settings: dict[str, Any],
) -> float:
    kind = _leading_legal_structure(next_text)
    if kind == "heading":
        return float(clamp(
            float(settings.get("pause_paragraph_ms", 430)),
            240,
            460,
        ))
    if kind == "article":
        return float(clamp(
            float(settings.get("pause_sentence_ms", 220)),
            170,
            300,
        ))
    if kind in {"paragraph", "inciso", "alinea"}:
        return float(clamp(
            float(settings.get("pause_colon_ms", 180)),
            110,
            230,
        ))
    return float(clamp(
        float(settings.get("pause_continuation_ms", 110)),
        70,
        160,
    ))


def _assemble_fallback_audio_parts(
    audio_parts: list[torch.Tensor],
    text_parts: list[str],
    sample_rate: int,
    settings: dict[str, Any],
) -> torch.Tensor:
    """Une subpartes validadas em PCM, com pausa curta e sem nova codificação."""
    if not audio_parts:
        raise RuntimeError("O fallback não produziu nenhuma parte de áudio.")

    output: list[torch.Tensor] = []
    previous_trailing_ms = 0.0

    for part_index, audio in enumerate(audio_parts):
        prepared, metrics = prepare_audio_for_join(
            audio,
            sample_rate,
            settings,
        )
        if part_index:
            desired_pause_ms = _fallback_join_pause_ms(
                text_parts[part_index],
                settings,
            )
            existing_pause_ms = (
                previous_trailing_ms + metrics["leadingSilenceMs"]
            )
            missing_pause_ms = max(0.0, desired_pause_ms - existing_pause_ms)
            silence_samples = int(round(
                sample_rate * missing_pause_ms / 1000.0
            ))
            if silence_samples:
                output.append(torch.zeros(
                    (1, silence_samples),
                    dtype=prepared.dtype,
                ))
        output.append(prepared)
        previous_trailing_ms = metrics["trailingSilenceMs"]

    return torch.cat(output, dim=1).contiguous()


def _validation_failure_reason(validation: dict[str, Any] | None) -> str:
    if not validation:
        return "falha_de_geracao_sem_transcricao_valida"
    if validation.get("material_omission"):
        return "omissao_material"
    if validation.get("critical_missing_tokens"):
        return "tokens_criticos_ausentes"
    if float(validation.get("recall", 0.0)) < float(
        validation.get("recall_threshold", 1.0)
    ):
        return "recall_abaixo_do_limite"
    if float(validation.get("similarity", 0.0)) < float(
        validation.get("similarity_threshold", 1.0)
    ):
        return "similaridade_abaixo_do_limite"
    return "validacao_reprovada"


def _audio_duration_seconds(audio: torch.Tensor, sample_rate: int) -> float:
    try:
        samples = int(audio.shape[-1])
        return max(0.0, samples / max(1, int(sample_rate)))
    except Exception:
        return 0.0


def _estimated_speech_duration(text: str) -> dict[str, float | int | bool]:
    words = len(re.findall(r"\b[\wÀ-ÿ]+\b", str(text or ""), re.UNICODE))
    wpm = float(clamp(EARLY_EOS_NARRATION_WPM, 120, 210))
    estimated = words * 60.0 / wpm
    minimum = max(
        1.2,
        estimated * float(clamp(EARLY_EOS_MIN_DURATION_RATIO, 0.30, 0.60)),
    )
    return {
        "words": words,
        "estimated_seconds": estimated,
        "minimum_reasonable_seconds": minimum,
        "duration_check_enabled": words >= int(clamp(
            EARLY_EOS_MIN_WORDS,
            8,
            20,
        )),
    }


def _retry_sampling_parameters(
    *,
    attempt_index: int,
    previous_failure_reason: str,
    temperature: float,
    exaggeration: float,
    cfg_weight: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
) -> dict[str, float]:
    """Variações pequenas para o retry; a primeira tentativa fica intacta."""
    if attempt_index <= 1:
        temperature_delta = 0.0
        cfg_delta = 0.0
        repetition_delta = 0.0
        min_p_delta = 0.0
    elif previous_failure_reason == "early_eos_token_repetition":
        # Saída de colapso do T3: aumenta moderadamente a entropia e reduz CFG.
        # Não muda timbre/referência; muda apenas o sampling da tentativa que
        # já falhou por repetição explícita de token.
        temperature_delta = 0.08 if attempt_index == 2 else 0.14
        cfg_delta = -0.10 if attempt_index == 2 else -0.18
        repetition_delta = 0.04 if attempt_index == 2 else 0.08
        min_p_delta = 0.01 if attempt_index == 2 else 0.02
    else:
        # Este hotfix não muda a calibração para falhas comuns de ASR/TTS.
        temperature_delta = 0.0
        cfg_delta = 0.0
        repetition_delta = 0.0
        min_p_delta = 0.0

    return {
        "temperature": float(clamp(
            temperature + temperature_delta,
            0.10,
            1.10,
        )),
        "exaggeration": float(clamp(exaggeration, 0.0, 1.0)),
        "cfg_weight": float(clamp(cfg_weight + cfg_delta, 0.20, 1.0)),
        "repetition_penalty": float(clamp(
            repetition_penalty + repetition_delta,
            1.0,
            1.40,
        )),
        "min_p": float(clamp(min_p + min_p_delta, 0.01, 0.10)),
        "top_p": float(clamp(top_p, 0.80, 1.0)),
    }


def _has_numeric_substitution(validation: dict[str, Any] | None) -> bool:
    if not validation:
        return False
    missing = [str(token) for token in validation.get(
        "critical_missing_tokens", []
    )]
    extras = [str(token) for token in validation.get(
        "extra_recognized_tokens", []
    )]
    missing_numbers = {token for token in missing if token.isdigit()}
    extra_numbers = {token for token in extras if token.isdigit()}
    return bool(missing_numbers and extra_numbers and missing_numbers != extra_numbers)


def _classify_generation_failure(
    validation: dict[str, Any] | None,
    *,
    token_repetition: bool = False,
    early_eos_suspected: bool = False,
    generation_exception: bool = False,
) -> str:
    # token_repetition só é marcado quando o próprio Chatterbox registra
    # "forcing EOS token ... token_repetition=True". Essa evidência interna
    # tem prioridade sobre a heurística secundária de duração.
    if token_repetition:
        return "early_eos_token_repetition"
    if generation_exception:
        return "generation_exception"
    if _has_numeric_substitution(validation):
        return "numeric_substitution"
    if validation and validation.get("material_omission"):
        return "real_content_omission"
    if validation and not validation.get("approved") and not validation.get(
        "critical_missing_tokens"
    ):
        return "asr_minor_divergence"
    return "unknown"


def generate_chunk_with_retry(
    model: ChatterboxMultilingualTTS,
    chunk: str,
    *,
    language_id: str,
    reference_path: Path,
    settings: dict[str, Any],
    chunk_index: int,
    total_chunks: int,
    verification_dir: Path | None = None,
    _allow_safe_fallback: bool = True,
    _fallback_label: str = "",
    _seed_offset: int = 0,
    _fallback_depth: int = 0,
    _subchunk_index: int = 0,
    _subchunk_total: int = 0,
    _approved_subchunk_cache: dict[str, torch.Tensor] | None = None,
    _reference_hash: str = "",
) -> torch.Tensor:
    """
    Gera um chunk com a mesma identidade vocal e, quando habilitado,
    só aceita o áudio após conferência do texto pelo Whisper.
    """
    candidate = clean_generation_chunk(chunk)
    if not is_valid_generation_chunk(candidate):
        raise ValueError(
            f"Trecho {chunk_index}/{total_chunks} ficou vazio ou inválido "
            "depois da preparação jurídica."
        )

    consistency_mode = bool(settings.get("voice_consistency_mode", True))
    fidelity_mode = bool(settings.get("voice_clone_fidelity_mode", True))
    verify_each_chunk = bool(settings.get("verify_each_chunk", True))
    verify_threshold = float(settings.get("chunk_verify_threshold", 0.90))
    max_attempts = int(settings.get("chunk_verify_attempts", 3))
    complexity = legal_chunk_complexity(candidate)
    mode = "normal" if _fallback_depth == 0 else "subchunk"
    subchunk_label = (
        f"{_subchunk_index}/{_subchunk_total}"
        if _subchunk_index and _subchunk_total
        else None
    )
    approved_subchunk_cache = (
        _approved_subchunk_cache
        if _approved_subchunk_cache is not None
        else {}
    )

    stability = float(settings.get("stability", 0.90))
    fidelity = float(settings.get("voice_fidelity", 0.93))

    # Sampling deve refletir uma única configuração efetiva. O worker antigo
    # aplicava novos clamps aqui, depois de resolve_settings(), criando duas
    # camadas de calibração e forçando exatamente o padrão visto nos logs
    # (temperature~0.55 + cfg~0.64). Agora esta função usa o envelope já
    # resolvido uma única vez.
    effective_temperature = float(settings["temperature"])
    effective_exaggeration = float(settings["exaggeration"])
    effective_cfg = float(settings["cfg_weight"])

    last_error: Exception | None = None
    last_transcript = ""
    last_similarity = 0.0
    last_recall = 0.0
    last_validation: dict[str, Any] | None = None
    last_failure_reason = "unknown"
    last_generation_diagnostics: dict[str, Any] = {}
    attempts_executed = 0

    for attempt_index in range(1, max_attempts + 1):
        attempts_executed = attempt_index
        chatterbox_logs: _ChatterboxGenerationLogCapture | None = None
        try:
            # A segunda tentativa reforça apenas a pontuação de fala. Nenhuma
            # palavra ou parâmetro da voz é alterado.
            attempt_candidate = (
                candidate
                if attempt_index == 1
                else punctuate_legal_speech_structure(candidate)
            )
            seed = (
                int(settings.get("voice_seed", 1701))
                + chunk_index * 1009
                + attempt_index * 37
                + int(_seed_offset)
            )
            _set_generation_seed(seed)

            attempt_parameters = _retry_sampling_parameters(
                attempt_index=attempt_index,
                previous_failure_reason=last_failure_reason,
                temperature=effective_temperature,
                exaggeration=effective_exaggeration,
                cfg_weight=effective_cfg,
                repetition_penalty=float(settings["repetition_penalty"]),
                min_p=float(settings["min_p"]),
                top_p=float(settings["top_p"]),
            )

            print(
                f"[CTEC] Trecho {chunk_index}/{total_chunks}, "
                f"modo={mode} | fallback={_fallback_label or 'none'} | "
                f"subchunk={subchunk_label or 'none'} | "
                f"complexidade={complexity['level']} | "
                f"tentativa {attempt_index}/{max_attempts}: {attempt_candidate[:180]!r} | "
                f"fidelity_mode={str(fidelity_mode).lower()} | "
                f"temperature={attempt_parameters['temperature']:.3f} | "
                f"exaggeration={attempt_parameters['exaggeration']:.3f} | "
                f"cfg={attempt_parameters['cfg_weight']:.3f} | "
                f"repetition_penalty={attempt_parameters['repetition_penalty']:.3f}",
                flush=True,
            )

            with _capture_chatterbox_generation_logs() as chatterbox_logs:
                audio = model.generate(
                    attempt_candidate,
                    language_id=language_id,
                    audio_prompt_path=str(reference_path),
                    exaggeration=attempt_parameters["exaggeration"],
                    cfg_weight=attempt_parameters["cfg_weight"],
                    temperature=attempt_parameters["temperature"],
                    repetition_penalty=attempt_parameters[
                        "repetition_penalty"
                    ],
                    min_p=attempt_parameters["min_p"],
                    top_p=attempt_parameters["top_p"],
                ).detach().cpu()

            if audio.numel() == 0:
                raise RuntimeError("O modelo devolveu um tensor de áudio vazio.")

            if audio.ndim == 1:
                audio = audio.unsqueeze(0)

            duration_seconds = _audio_duration_seconds(audio, model.sr)
            duration_estimate = _estimated_speech_duration(attempt_candidate)
            early_eos_suspected = bool(
                duration_estimate["duration_check_enabled"]
                and duration_seconds > 0.0
                and duration_seconds
                < float(duration_estimate["minimum_reasonable_seconds"])
            )
            analyzer_diagnostics = chatterbox_logs.as_dict()
            token_repetition_eos = bool(
                analyzer_diagnostics["token_repetition"]
            )
            last_generation_diagnostics = {
                "chunk_original": f"{chunk_index}/{total_chunks}",
                "modo": mode,
                "subchunk": subchunk_label,
                "profundidade_fallback": _fallback_depth,
                "texto_enviado": attempt_candidate[:1000],
                "caracteres": len(attempt_candidate),
                "palavras": duration_estimate["words"],
                "token_repetido": analyzer_diagnostics["repeated_token"],
                "repeticoes_detectadas": analyzer_diagnostics[
                    "repetition_count"
                ],
                "passo_eos": analyzer_diagnostics["eos_step"],
                "token_repetition": analyzer_diagnostics["token_repetition"],
                "long_tail": analyzer_diagnostics["long_tail"],
                "alignment_repetition": analyzer_diagnostics[
                    "alignment_repetition"
                ],
                "eos_detectado": analyzer_diagnostics["eos_detected"],
                "duracao_produzida_s": round(duration_seconds, 3),
                "duracao_estimada_s": round(float(
                    duration_estimate["estimated_seconds"]
                ), 3),
                "duracao_minima_razoavel_s": round(float(
                    duration_estimate["minimum_reasonable_seconds"]
                ), 3),
                "early_eos_suspected": early_eos_suspected,
                "temperature": attempt_parameters["temperature"],
                "exaggeration": attempt_parameters["exaggeration"],
                "cfg": attempt_parameters["cfg_weight"],
                "repetition_penalty": attempt_parameters[
                    "repetition_penalty"
                ],
                "min_p": attempt_parameters["min_p"],
                "top_p": attempt_parameters["top_p"],
                "referencia_voz_hash": _reference_hash or "unknown",
                "logs_analisador": analyzer_diagnostics["messages"],
            }
            print(
                "[CTEC] Diagnóstico da geração: "
                + json.dumps(last_generation_diagnostics, ensure_ascii=False),
                flush=True,
            )

            # Só pula o Whisper quando há evidência conjunta: o próprio
            # Chatterbox declarou token_repetition e a duração é inviável.
            # Isso nunca aprova áudio; apenas escolhe o retry/fallback correto.
            if token_repetition_eos:
                last_failure_reason = "early_eos_token_repetition"
                last_error = RuntimeError(
                    "EOS precoce provocado por repetição de token."
                )
                can_subdivide_after_retry = (
                    _allow_safe_fallback
                    and _fallback_depth < MAX_ADAPTIVE_FALLBACK_DEPTH
                    and len(candidate) > max(
                        36,
                        int(MIN_TTS_SUBCHUNK_CHARS) // 2,
                    )
                )
                if attempt_index >= min(2, max_attempts) and can_subdivide_after_retry:
                    break
                continue

            if verify_each_chunk and get_whisper() is not None:
                verify_root = verification_dir or reference_path.parent
                verify_root.mkdir(parents=True, exist_ok=True)
                verify_path = verify_root / (
                    f"verify_chunk_{chunk_index:05d}"
                    f"{'_' + _fallback_label if _fallback_label else ''}"
                    f"_attempt_{attempt_index}.wav"
                )
                torchaudio.save(str(verify_path), audio, model.sr)

                transcript = transcribe_audio(verify_path)

                validation = validate_legal_transcription(
                    candidate,
                    transcript,
                    verify_threshold,
                )
                similarity = float(validation["similarity"])
                recall = float(validation["recall"])

                last_transcript = transcript
                last_similarity = similarity
                last_recall = recall
                last_validation = validation

                # Para evitar informação pulada, recall pesa mais que similaridade geral.
                # Equivalências jurídicas são resolvidas na cópia normalizada antes
                # do cálculo; elas nunca ignoram os thresholds nem mascaram omissões.
                semantic_equivalent = bool(
                    validation["equivalent_representation"]
                )
                approved = bool(validation["approved"])
                classified_failure_reason = (
                    "unknown" if approved else _classify_generation_failure(
                        validation,
                        token_repetition=token_repetition_eos,
                        early_eos_suspected=early_eos_suspected,
                    )
                )
                if not approved:
                    last_failure_reason = classified_failure_reason

                print(
                    "[CTEC] Verificação Whisper: "
                    f"chunk={chunk_index}/{total_chunks} | "
                    f"attempt={attempt_index} | "
                    f"similarity={similarity:.3f} | "
                    f"word_recall={recall:.3f} | "
                    f"semantic_equivalent={str(semantic_equivalent).lower()} | "
                    f"approved={str(approved).lower()} | "
                    f"recognized={transcript[:180]!r}",
                    flush=True,
                )
                print(
                    "[CTEC] Diagnóstico da validação: "
                    + json.dumps({
                        "chunk_original": f"{chunk_index}/{total_chunks}",
                        "modo": mode,
                        "subchunk": subchunk_label,
                        "profundidade_fallback": _fallback_depth,
                        "caracteres": len(candidate),
                        "palavras": len(candidate.split()),
                        "complexidade": complexity["level"],
                        "pontuacao_complexidade": complexity["score"],
                        "tentativa": f"{attempt_index}/{max_attempts}",
                        "similaridade": round(similarity, 6),
                        "recall": round(recall, 6),
                        "omissao_material": validation["material_omission"],
                        "texto_esperado": candidate[:600],
                        "transcricao": transcript[:600],
                        "motivo_reprovacao": (
                            None if approved
                            else _validation_failure_reason(validation)
                        ),
                        "failure_reason": (
                            None if approved else classified_failure_reason
                        ),
                        "early_eos_suspected": early_eos_suspected,
                        "token_repetition": token_repetition_eos,
                        "token_repetido": analyzer_diagnostics[
                            "repeated_token"
                        ],
                        "passo_eos": analyzer_diagnostics["eos_step"],
                        "raw_recall": validation["raw_recall"],
                        "esperado_normalizado": validation["expected_normalized"],
                        "reconhecido_normalizado": validation["recognized_normalized"],
                        "tokens_esperados_nao_encontrados": validation[
                            "missing_expected_tokens"
                        ],
                        "tokens_criticos_nao_encontrados": validation[
                            "critical_missing_tokens"
                        ],
                        "tokens_extras_reconhecidos": validation[
                            "extra_recognized_tokens"
                        ],
                        "divergencias_normalizadas": validation[
                            "normalized_divergences"
                        ],
                    }, ensure_ascii=False),
                    flush=True,
                )

                try:
                    verify_path.unlink(missing_ok=True)
                except Exception:
                    pass

                if not approved:
                    last_error = RuntimeError(
                        "O trecho gerado não reproduziu todo o texto esperado."
                    )
                    can_subdivide = (
                        _allow_safe_fallback
                        and _fallback_depth < MAX_ADAPTIVE_FALLBACK_DEPTH
                        and len(candidate) > max(
                            36,
                            int(MIN_TTS_SUBCHUNK_CHARS) // 2,
                        )
                    )
                    # Duas ocorrências confirmadas de EOS por repetição vão para
                    # subdivisão. Omissão real complexa também não é repetida
                    # três vezes; subchunks continuam sujeitos ao validador.
                    if (
                        can_subdivide
                        and (
                            (
                                classified_failure_reason
                                == "early_eos_token_repetition"
                                and attempt_index >= min(2, max_attempts)
                            )
                            or (
                                bool(validation.get("material_omission"))
                                and (
                                    complexity["level"] == "alta"
                                    or attempt_index >= min(2, max_attempts)
                                )
                            )
                        )
                    ):
                        break
                    continue

            return audio

        except (IndexError, RuntimeError) as error:
            last_error = error
            captured_exception_logs = (
                chatterbox_logs.as_dict()
                if chatterbox_logs is not None
                else {}
            )
            exception_token_repetition = bool(
                captured_exception_logs.get("token_repetition")
            )
            last_failure_reason = _classify_generation_failure(
                last_validation,
                token_repetition=exception_token_repetition,
                early_eos_suspected=exception_token_repetition,
                generation_exception=not exception_token_repetition,
            )
            print(
                f"[CTEC] Falha no trecho {chunk_index}/{total_chunks}, "
                f"fallback={_fallback_label or 'none'}, "
                f"tentativa {attempt_index}: {type(error).__name__}: {error} | "
                f"failure_reason={last_failure_reason}",
                flush=True,
            )

    can_fallback = (
        _allow_safe_fallback
        and _fallback_depth < MAX_ADAPTIVE_FALLBACK_DEPTH
        and len(candidate) > max(36, int(MIN_TTS_SUBCHUNK_CHARS) // 2)
        and (
            last_validation is None
            or not bool(last_validation.get("approved"))
        )
    )
    if can_fallback:
        fallback_parts = _split_incomplete_chunk_for_fallback(
            candidate,
            depth=_fallback_depth,
        )
        if fallback_parts:
            print(
                "[CTEC] Fallback adaptativo ativado: "
                f"chunk={chunk_index}/{total_chunks} | "
                f"depth={_fallback_depth + 1}/{MAX_ADAPTIVE_FALLBACK_DEPTH} | "
                f"reason={last_failure_reason} | "
                f"parts={len(fallback_parts)} | "
                f"complexity={complexity['level']}",
                flush=True,
            )
            fallback_audio: list[torch.Tensor] = []
            try:
                for part_index, fallback_part in enumerate(
                    fallback_parts,
                    start=1,
                ):
                    generation_part = fallback_part
                    cache_key = hashlib.sha256(
                        (
                            f"{chunk_index}|{_fallback_depth + 1}|"
                            f"{part_index}|{generation_part}"
                        ).encode("utf-8")
                    ).hexdigest()
                    if cache_key in approved_subchunk_cache:
                        part_audio = approved_subchunk_cache[cache_key]
                        print(
                            "[CTEC] Subchunk aprovado reutilizado do cache "
                            f"temporário: {part_index}/{len(fallback_parts)}",
                            flush=True,
                        )
                    else:
                        part_audio = generate_chunk_with_retry(
                            model,
                            generation_part,
                            language_id=language_id,
                            reference_path=reference_path,
                            settings=settings,
                            chunk_index=chunk_index,
                            total_chunks=total_chunks,
                            verification_dir=verification_dir,
                            _allow_safe_fallback=True,
                            _fallback_label=(
                                f"part_{part_index}_of_{len(fallback_parts)}"
                            ),
                            _seed_offset=part_index * 100003,
                            _fallback_depth=_fallback_depth + 1,
                            _subchunk_index=part_index,
                            _subchunk_total=len(fallback_parts),
                            _approved_subchunk_cache=approved_subchunk_cache,
                            _reference_hash=_reference_hash,
                        )
                        approved_subchunk_cache[cache_key] = part_audio
                    fallback_audio.append(part_audio)
                joined = _assemble_fallback_audio_parts(
                    fallback_audio,
                    fallback_parts,
                    model.sr,
                    settings,
                )
                print(
                    "[CTEC] Fallback adaptativo aprovado: "
                    f"chunk={chunk_index}/{total_chunks} | "
                    f"depth={_fallback_depth + 1} | "
                    f"parts={len(fallback_parts)}",
                    flush=True,
                )
                return joined
            except RuntimeError as fallback_error:
                last_error = RuntimeError(
                    "O fallback adaptativo também falhou: "
                    f"{fallback_error}"
                )

    details = ""
    if last_transcript:
        validation_details = {}
        if last_validation is not None:
            validation_details = {
                "tokens_esperados_nao_encontrados": last_validation[
                    "missing_expected_tokens"
                ],
                "tokens_extras_reconhecidos": last_validation[
                    "extra_recognized_tokens"
                ],
                "tokens_criticos_nao_encontrados": last_validation[
                    "critical_missing_tokens"
                ],
                "divergencias_normalizadas": last_validation[
                    "normalized_divergences"
                ],
                "omissao_material": last_validation["material_omission"],
                "failure_reason": last_failure_reason,
                "diagnostico_geracao": last_generation_diagnostics,
            }
        details = (
            f" Similaridade final: {last_similarity:.3f}; "
            f"recall final: {last_recall:.3f}; "
            f"reconhecido: {last_transcript[:600]!r}; "
            "diagnóstico: "
            + json.dumps(validation_details, ensure_ascii=False)
            + "."
        )

    raise RuntimeError(
        f"O Chatterbox não conseguiu gerar corretamente o trecho "
        f"{chunk_index}/{total_chunks} após {attempts_executed} tentativas. "
        f"Modo: {mode}; subchunk: {subchunk_label or 'não'}; "
        f"caracteres: {len(candidate)}; palavras: {len(candidate.split())}; "
        f"complexidade: {complexity['level']}; "
        f"failure_reason: {last_failure_reason}; "
        f"Trecho esperado: {candidate[:220]!r}. "
        f"Erro final: {last_error}.{details}"
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

    raw_chunks = split_text(
        prepared,
        int(settings["chunk_limit"]),
        preserve_complete_sentences=settings["preserve_complete_sentences"],
        split_by_legal_structure=settings["split_by_legal_structure"],
        context_margin_words=settings["chunk_overlap_words"],
        target=int(settings["chunk_target"]),
    )
    chunks_before_merge = len(raw_chunks)
    chunks = merge_tiny_chunks(
        raw_chunks,
        maximum_chars=int(settings["chunk_limit"]),
    )
    chunks_before_adaptive = len(chunks)
    if settings.get("text_mode") == "law":
        chunks = adapt_chunks_for_legal_complexity(
            chunks,
            int(settings["chunk_limit"]),
        )
    if not chunks:
        raise ValueError("Nenhum trecho válido foi produzido.")
    integrity = validate_chunk_integrity(prepared, chunks)
    integrity["chunks_before_merge"] = chunks_before_merge
    integrity["chunks_after_merge"] = chunks_before_adaptive
    integrity["chunks_before_adaptive"] = chunks_before_adaptive
    integrity["chunks_after_adaptive"] = len(chunks)

    mp3_bitrate = str(data.get("mp3_bitrate") or "160k").strip().lower()
    if mp3_bitrate not in {"96k", "128k", "160k", "192k", "256k", "320k"}:
        mp3_bitrate = "160k"
    model = get_model()
    started = time.time()

    with _GENERATION_LOCK, tempfile.TemporaryDirectory(
        prefix="ctec_long_voice_"
    ) as tmp:
        root = Path(tmp)
        reference_source = save_reference_audio(data, root)
        reference_path, reference_metrics = prepare_reference_audio(
            reference_source,
            root,
            settings,
        )
        ref_hash = reference_path_hash(reference_path)
        print(
            f"[CTEC] reference_used=true | reference_path_hash={ref_hash} | "
            f"voice_consistency_mode={str(settings['voice_consistency_mode']).lower()}",
            flush=True,
        )
        raw_wav = root / "ctec-audio-continuo.wav"
        final_path = root / "ctec-audio-longo.mp3"

        marker_list: list[dict[str, Any]] = []
        total = len(chunks)
        speed = float(settings["speed"])

        # Uma chamada ao modelo por bloco semântico. Não divide mais cada frase,
        # vírgula ou inciso em nova interpretação.
        with ContinuousWaveAssembler(raw_wav, model.sr, settings) as assembler:
            for index, (chunk, paragraph_end) in enumerate(chunks, start=1):
                percent = 8 + int((index - 1) / max(1, total) * 82)
                runpod.serverless.progress_update(
                    job,
                    json.dumps({
                        "stage": "generating",
                        "stageLabel": f"Gerando bloco fluido {index} de {total}",
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
                    verification_dir=root / "verification",
                    _reference_hash=ref_hash,
                )
                seam = assembler.add(audio, chunk, paragraph_end)
                marker_list.append({
                    "index": index,
                    "startSeconds": round(
                        seam["startSample"] / model.sr / speed,
                        2,
                    ),
                    "durationSeconds": round(
                        seam["audioSamples"] / model.sr / speed,
                        2,
                    ),
                    "text": chunk[:500],
                    "join": {
                        "insertedPauseMs": round(seam["insertedPauseMs"], 1),
                        "leadingSilenceMs": round(seam["leadingSilenceMs"], 1),
                        "trailingSilenceMs": round(seam["trailingSilenceMs"], 1),
                    },
                })

        runpod.serverless.progress_update(
            job,
            json.dumps({
                "stage": "joining",
                "stageLabel": "Finalizando áudio contínuo",
                "progress": 0.93,
                "currentChunk": total,
                "totalChunks": total,
                "elapsedSeconds": int(time.time() - started),
            }),
        )

        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(raw_wav),
        ]
        final_filter = build_ffmpeg_filter(
            settings,
            include_edge_silence=False,
        )
        if final_filter:
            command += ["-filter:a", final_filter]
        command += [
            "-ac", "1", "-ar", str(model.sr),
            "-codec:a", "libmp3lame", "-b:a", mp3_bitrate,
            str(final_path),
        ]
        subprocess.run(command, check=True)
        final_duration = probe_audio_duration(final_path)

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
            "duration_seconds": round(final_duration, 2),
            "chunks": total,
            "markers": marker_list,
            "assembly": "single_pcm_then_single_mp3",
            "settings": public_settings(settings),
            "integrity": integrity,
            "reference_metrics": reference_metrics,
            "reference_used": True,
            "reference_path_hash": ref_hash,
            "voice_consistency_mode": settings["voice_consistency_mode"],
            "punctuation_prosody": {
                "comma_ms": settings["pause_comma_ms"],
                "sentence_ms": settings["pause_sentence_ms"],
                "colon_ms": settings["pause_colon_ms"],
                "paragraph_ms": settings["pause_paragraph_ms"],
                "internal_punctuation": "model_managed",
            },
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

    raw_chunks = split_text(
        text,
        int(settings["chunk_limit"]),
        preserve_complete_sentences=settings["preserve_complete_sentences"],
        split_by_legal_structure=settings["split_by_legal_structure"],
        context_margin_words=settings["chunk_overlap_words"],
        target=int(settings["chunk_target"]),
    )
    chunks_before_merge = len(raw_chunks)
    chunks = merge_tiny_chunks(
        raw_chunks,
        maximum_chars=int(settings["chunk_limit"]),
    )
    chunks_before_adaptive = len(chunks)
    if settings.get("text_mode") == "law":
        chunks = adapt_chunks_for_legal_complexity(
            chunks,
            int(settings["chunk_limit"]),
        )
    if not chunks:
        raise ValueError(
            "O texto não gerou nenhum trecho válido depois da preparação jurídica."
        )
    integrity = validate_chunk_integrity(text, chunks)
    integrity["chunks_before_merge"] = chunks_before_merge
    integrity["chunks_after_merge"] = chunks_before_adaptive
    integrity["chunks_before_adaptive"] = chunks_before_adaptive
    integrity["chunks_after_adaptive"] = len(chunks)

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
            settings,
        )
        ref_hash = reference_path_hash(reference_path)
        print(
            f"[CTEC] reference_used=true | reference_path_hash={ref_hash} | "
            f"voice_consistency_mode={str(settings['voice_consistency_mode']).lower()}",
            flush=True,
        )
        raw_wav = root / "raw.wav"
        final_path = root / f"ctec-voz-neural.{output_format}"

        total = len(chunks)

        with ContinuousWaveAssembler(raw_wav, model.sr, settings) as assembler:
            for index, (chunk, paragraph_end) in enumerate(chunks, start=1):
                runpod.serverless.progress_update(
                    job,
                    f"Gerando bloco fluido {index} de {total}",
                )
                print(f"[CTEC] Gerando bloco fluido {index}/{total}", flush=True)

                audio = generate_chunk_with_retry(
                    model,
                    chunk,
                    language_id=language_id,
                    reference_path=reference_path,
                    settings=settings,
                    chunk_index=index,
                    total_chunks=total,
                    verification_dir=root / "verification",
                    _reference_hash=ref_hash,
                )
                assembler.add(audio, chunk, paragraph_end)

        command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_wav)]
        audio_filter = build_ffmpeg_filter(
            settings,
            include_edge_silence=False,
        )
        if audio_filter:
            command += ["-filter:a", audio_filter]

        if output_format == "mp3":
            command += ["-codec:a", "libmp3lame", "-b:a", mp3_bitrate, str(final_path)]
            mime_type = "audio/mpeg"
        else:
            command += ["-codec:a", "pcm_s16le", str(final_path)]
            mime_type = "audio/wav"

        subprocess.run(command, check=True)
        duration_seconds = round(probe_audio_duration(final_path), 2)
        encoded = encode_output(final_path)
        recognized_text = ""
        if to_bool(data.get("verify_transcript"), True):
            recognized_text = transcribe_audio(final_path)

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
            "model": _LOADED_MODEL_VERSION or MODEL_VERSION,
            "chunks": total,
            "assembly": "single_pcm_then_single_encoding",
            "voice_id": str(data.get("voice_id") or "") or None,
            "settings": public_settings(settings),
            "prepared_text": text,
            "reference_metrics": reference_metrics,
            "integrity": integrity,
            "reference_used": True,
            "reference_path_hash": ref_hash,
            "voice_consistency_mode": settings["voice_consistency_mode"],
            "punctuation_prosody": {
                "comma_ms": settings["pause_comma_ms"],
                "sentence_ms": settings["pause_sentence_ms"],
                "colon_ms": settings["pause_colon_ms"],
                "paragraph_ms": settings["pause_paragraph_ms"],
                "internal_punctuation": "model_managed",
            },
            "recognized_text": recognized_text,
            "transcription_similarity": round(
                transcription_similarity(text, recognized_text) * 100,
                1,
            ) if recognized_text else None,
        }


if __name__ == "__main__":
    print("[CTEC] Iniciando CTEC Estúdio de Voz Worker 5.4.6...", flush=True)
    print(f"[CTEC] Device: {DEVICE} | Modelo solicitado: {MODEL_VERSION}", flush=True)
    runpod.serverless.start({"handler": generate})
