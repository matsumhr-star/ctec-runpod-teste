import base64
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any

import runpod
import torch
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_VERSION = os.getenv("CTEC_CHATTERBOX_MODEL", "v3").strip().lower() or "v3"
MAX_TEXT_CHARS = int(os.getenv("CTEC_MAX_TEXT_CHARS", "120000"))
MAX_REFERENCE_BYTES = int(os.getenv("CTEC_MAX_REFERENCE_BYTES", str(30 * 1024 * 1024)))
MAX_RESULT_BASE64_BYTES = int(os.getenv("CTEC_MAX_RESULT_BASE64_BYTES", str(14 * 1024 * 1024)))

_MODEL: ChatterboxMultilingualTTS | None = None
_MODEL_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()


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
                    t3_model=MODEL_VERSION,
                )
                print("[CTEC] Modelo carregado com sucesso.", flush=True)
    return _MODEL


def normalize_law_text(text: str) -> str:
    text = re.sub(r"\bArt\.\s*(\d+)", r"Artigo \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bArts\.\s*", "Artigos ", text, flags=re.IGNORECASE)
    text = re.sub(r"§\s*(\d+)º?", r"Parágrafo \1", text)
    text = re.sub(r"§\s*único", "Parágrafo único", text, flags=re.IGNORECASE)
    text = re.sub(r"\binc\.\s*", "inciso ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bal\.\s*", "alínea ", text, flags=re.IGNORECASE)
    return text


def prepare_text(text: str, mode: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if mode == "law":
        text = normalize_law_text(text)
    return text


def split_text(text: str, limit: int) -> list[tuple[str, bool]]:
    """Retorna (trecho, fim_de_paragrafo)."""
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks: list[tuple[str, bool]] = []

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?;:])\s+", paragraph)
        current = ""
        local: list[str] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > limit:
                if current:
                    local.append(current)
                    current = ""
                words = sentence.split()
                piece = ""
                for word in words:
                    candidate = f"{piece} {word}".strip()
                    if piece and len(candidate) > limit:
                        local.append(piece)
                        piece = word
                    else:
                        piece = candidate
                if piece:
                    local.append(piece)
                continue

            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > limit:
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


def resolve_settings(data: dict[str, Any]) -> dict[str, Any]:
    profile_name = str(data.get("profile") or "law_natural").strip().lower()
    base = dict(PROFILES.get(profile_name, PROFILES["law_natural"]))

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
        "pitch_semitones": (-6.0, 6.0),
        "gain_db": (-12.0, 12.0),
        "chunk_limit": (100, 600),
    }

    for key, (minimum, maximum) in numeric_limits.items():
        if key in data:
            base[key] = clamp(float(data[key]), minimum, maximum)

    for key in ("normalize", "trim_silence"):
        if key in data:
            base[key] = to_bool(data[key], bool(base.get(key, False)))

    if "text_mode" in data:
        base["text_mode"] = str(data["text_mode"]).strip().lower()

    base["profile"] = profile_name
    base["chunk_limit"] = int(base.get("chunk_limit", 320))
    base["pause_sentence_ms"] = int(base["pause_sentence_ms"])
    base["pause_paragraph_ms"] = int(base["pause_paragraph_ms"])
    return base


def build_ffmpeg_filter(settings: dict[str, Any]) -> str:
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
            "silenceremove=start_periods=1:start_duration=0.05:start_threshold=-50dB:"
            "stop_periods=-1:stop_duration=0.25:stop_threshold=-50dB"
        )

    if settings.get("normalize"):
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    return ",".join(filters)


def encode_output(path: Path) -> str:
    raw = path.read_bytes()
    if len(raw) > MAX_RESULT_BASE64_BYTES:
        raise ValueError(
            "O áudio final ficou grande demais para retorno em Base64. "
            "Divida o texto ou integre o worker ao Firebase Storage."
        )
    return base64.b64encode(raw).decode("ascii")


def capabilities() -> dict[str, Any]:
    return {
        "status": "ok",
        "worker": "CTEC Voz Neural",
        "version": "2.0.0",
        "device": DEVICE,
        "model": f"Chatterbox Multilingual {MODEL_VERSION}",
        "profiles": PROFILES,
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        "reference_inputs": ["voice_id", "reference_audio_url", "reference_audio_base64"],
        "output_formats": ["mp3", "wav"],
    }


def generate(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("input") or {}
    action = str(data.get("action") or "generate").strip().lower()

    if action in {"capabilities", "config", "health"}:
        return capabilities()

    text = str(data.get("text") or "").strip()
    if len(text) < 3:
        raise ValueError("O texto precisa ter pelo menos 3 caracteres.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"O texto excede o limite de {MAX_TEXT_CHARS} caracteres.")

    language_id = str(data.get("language_id") or "pt").strip().lower()
    if language_id not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Idioma não suportado: {language_id}")

    settings = resolve_settings(data)
    text = prepare_text(text, settings["text_mode"])

    if to_bool(data.get("preview"), False):
        preview_chars = int(clamp(float(data.get("preview_chars", 450)), 80, 1500))
        text = text[:preview_chars].rsplit(" ", 1)[0].strip() or text[:preview_chars]

    chunks = split_text(text, int(settings["chunk_limit"]))
    if not chunks:
        raise ValueError("Não foi possível dividir o texto para geração.")

    output_format = str(data.get("output_format") or "mp3").strip().lower()
    if output_format not in {"mp3", "wav"}:
        raise ValueError("output_format deve ser mp3 ou wav.")
    mp3_bitrate = str(data.get("mp3_bitrate") or "192k").strip().lower()
    if mp3_bitrate not in {"96k", "128k", "160k", "192k", "256k", "320k"}:
        mp3_bitrate = "192k"

    model = get_model()

    with _GENERATION_LOCK, tempfile.TemporaryDirectory(prefix="ctec_voice_") as tmp:
        root = Path(tmp)
        reference_path = save_reference_audio(data, root)
        raw_wav = root / "raw.wav"
        final_path = root / f"ctec-voz-neural.{output_format}"

        sentence_silence = torch.zeros(
            1, int(model.sr * settings["pause_sentence_ms"] / 1000.0), dtype=torch.float32
        )
        paragraph_silence = torch.zeros(
            1, int(model.sr * settings["pause_paragraph_ms"] / 1000.0), dtype=torch.float32
        )

        segments: list[torch.Tensor] = []
        total = len(chunks)

        for index, (chunk, paragraph_end) in enumerate(chunks, start=1):
            runpod.serverless.progress_update(job, f"Gerando trecho {index} de {total}")
            print(f"[CTEC] Gerando trecho {index}/{total}", flush=True)

            audio = model.generate(
                chunk,
                language_id=language_id,
                audio_prompt_path=str(reference_path),
                exaggeration=float(settings["exaggeration"]),
                cfg_weight=float(settings["cfg_weight"]),
                temperature=float(settings["temperature"]),
                repetition_penalty=float(settings["repetition_penalty"]),
                min_p=float(settings["min_p"]),
                top_p=float(settings["top_p"]),
            ).detach().cpu()

            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            segments.append(audio)

            if index < total:
                segments.append(paragraph_silence if paragraph_end else sentence_silence)

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

        duration_seconds = round(combined.shape[1] / model.sr / float(settings["speed"]), 2)

        return {
            "status": "ok",
            "audio_base64": encoded,
            "mime_type": mime_type,
            "file_name": final_path.name,
            "sample_rate": model.sr,
            "duration_seconds_estimate": duration_seconds,
            "device": DEVICE,
            "model": MODEL_VERSION,
            "chunks": total,
            "voice_id": str(data.get("voice_id") or "") or None,
            "settings": settings,
        }


if __name__ == "__main__":
    print("[CTEC] Iniciando CTEC Voz Neural Worker 2.0...", flush=True)
    print(f"[CTEC] Device: {DEVICE} | Modelo: {MODEL_VERSION}", flush=True)
    runpod.serverless.start({"handler": generate})
