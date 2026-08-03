import base64
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

_MODEL: ChatterboxMultilingualTTS | None = None
_MODEL_LOCK = threading.Lock()


def get_model() -> ChatterboxMultilingualTTS:
    """Carrega o modelo somente na primeira geração."""
    global _MODEL

    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                print(f"[CTEC] Carregando Chatterbox Multilingual em {DEVICE}...", flush=True)
                _MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
                print("[CTEC] Modelo carregado com sucesso.", flush=True)

    return _MODEL


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def split_text(text: str, limit: int = 320) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
    chunks: list[str] = []

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?;:])\s+", paragraph)
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > limit:
                if current:
                    chunks.append(current)
                    current = ""

                for start in range(0, len(sentence), limit):
                    piece = sentence[start:start + limit].strip()
                    if piece:
                        chunks.append(piece)
                continue

            candidate = f"{current} {sentence}".strip()

            if current and len(candidate) > limit:
                chunks.append(current)
                current = sentence
            else:
                current = candidate

        if current:
            chunks.append(current)

    return chunks


def reference_suffix(data: dict[str, Any], base64_value: str) -> str:
    explicit = str(data.get("reference_audio_format") or "").strip().lower()
    if explicit in {"wav", "mp3", "m4a", "ogg", "flac"}:
        return f".{explicit}"

    match = re.match(r"^data:audio/([^;]+);base64,", base64_value)
    if match:
        mime_part = match.group(1).lower()
        mapping = {
            "wav": ".wav",
            "x-wav": ".wav",
            "mpeg": ".mp3",
            "mp3": ".mp3",
            "mp4": ".m4a",
            "m4a": ".m4a",
            "ogg": ".ogg",
            "flac": ".flac",
        }
        return mapping.get(mime_part, ".wav")

    return ".wav"


def save_reference_audio(data: dict[str, Any], destination_root: Path) -> Path:
    base64_value = str(data.get("reference_audio_base64") or "").strip()
    url_value = str(data.get("reference_audio_url") or "").strip()

    if base64_value:
        suffix = reference_suffix(data, base64_value)
        destination = destination_root / f"reference{suffix}"
        cleaned = re.sub(r"^data:audio/[^;]+;base64,", "", base64_value)
        destination.write_bytes(base64.b64decode(cleaned, validate=True))
        return destination

    if url_value:
        suffix = Path(url_value.split("?", 1)[0]).suffix.lower()
        if suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}:
            suffix = ".wav"

        destination = destination_root / f"reference{suffix}"
        request = urllib.request.Request(
            url_value,
            headers={"User-Agent": "CTEC-Voice-Worker/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            destination.write_bytes(response.read())
        return destination

    raise ValueError("Envie reference_audio_base64 ou reference_audio_url.")


def generate(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("input") or {}

    text = str(data.get("text") or "").strip()
    language_id = str(data.get("language_id") or "pt").strip().lower()
    speed = clamp(float(data.get("speed", 1.0)), 0.75, 1.30)
    expression = clamp(float(data.get("expression", 45)), 0, 100)
    cfg_weight = clamp(float(data.get("cfg_weight", 0.5)), 0.0, 1.0)
    pause_ms = int(clamp(float(data.get("pause_ms", 280)), 0, 2000))
    chunk_limit = int(clamp(float(data.get("chunk_limit", 320)), 100, 600))

    if len(text) < 3:
        raise ValueError("O texto precisa ter pelo menos 3 caracteres.")

    exaggeration = 0.28 + (expression / 100.0) * 0.62
    chunks = split_text(text, limit=chunk_limit)

    if not chunks:
        raise ValueError("Não foi possível preparar o texto para geração.")

    model = get_model()

    with tempfile.TemporaryDirectory(prefix="ctec_voice_") as temporary:
        root = Path(temporary)
        reference_path = save_reference_audio(data, root)
        wav_path = root / "generated.wav"
        mp3_path = root / "generated.mp3"

        segments: list[torch.Tensor] = []
        silence = torch.zeros(
            1,
            int(model.sr * (pause_ms / 1000.0)),
            dtype=torch.float32,
        )

        total = len(chunks)

        for index, chunk in enumerate(chunks, start=1):
            print(f"[CTEC] Gerando trecho {index}/{total}...", flush=True)
            runpod.serverless.progress_update(job, f"Gerando trecho {index} de {total}")

            audio = model.generate(
                chunk,
                language_id=language_id,
                audio_prompt_path=str(reference_path),
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            ).detach().cpu()

            if audio.ndim == 1:
                audio = audio.unsqueeze(0)

            segments.append(audio)

            if index < total and pause_ms > 0:
                segments.append(silence)

        combined = torch.cat(segments, dim=1)
        torchaudio.save(str(wav_path), combined, model.sr)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-filter:a",
                f"atempo={speed:.2f}",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(mp3_path),
            ],
            check=True,
        )

        encoded = base64.b64encode(mp3_path.read_bytes()).decode("ascii")

        return {
            "status": "ok",
            "audio_base64": encoded,
            "mime_type": "audio/mpeg",
            "file_name": "ctec-voz-neural.mp3",
            "sample_rate": model.sr,
            "device": DEVICE,
            "chunks": total,
            "settings": {
                "language_id": language_id,
                "speed": speed,
                "expression": expression,
                "cfg_weight": cfg_weight,
                "pause_ms": pause_ms,
            },
        }


if __name__ == "__main__":
    print("[CTEC] Iniciando worker RunPod Serverless...", flush=True)
    runpod.serverless.start({"handler": generate})
