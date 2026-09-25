import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np
import runpod
import soundfile as sf
import torch
from num2words import num2words
from qwen_tts import Qwen3TTSModel

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

BUILD = "CTEC-QWEN3-PTBR-ICL-V4-NO-PA-2026-09-25"
MODEL_ID = os.getenv("CTEC_QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_TEXT_CHARS = int(os.getenv("CTEC_MAX_TEXT_CHARS", "120000"))
MAX_REFERENCE_BYTES = int(os.getenv("CTEC_MAX_REFERENCE_BYTES", str(30 * 1024 * 1024)))
MAX_RESULT_BASE64_BYTES = int(os.getenv("CTEC_MAX_RESULT_BASE64_BYTES", str(14 * 1024 * 1024)))
CHUNK_LIMIT = int(os.getenv("CTEC_QWEN_CHUNK_CHARS", "650"))
WORKER_CONTRACT_VERSION = 3

_MODEL = None
_MODEL_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()

_WHISPER = None
_WHISPER_LOCK = threading.Lock()
_REFERENCE_TEXT_CACHE: dict[str, str] = {}
_REFERENCE_TEXT_CACHE_LOCK = threading.Lock()
WHISPER_MODEL_SIZE = os.getenv("CTEC_WHISPER_MODEL", "small")

LANGUAGES = {
    "pt": "Portuguese", "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "ru": "Russian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese",
}

def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "sim", "on"}

def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(float(value), minimum), maximum)

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_voice_library() -> dict[str, str]:
    raw = os.getenv("CTEC_VOICE_LIBRARY_JSON", "{}").strip()
    try:
        parsed = json.loads(raw or "{}")
        return {str(k): str(v) for k, v in parsed.items() if str(v).strip()}
    except Exception:
        return {}

def get_model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                if DEVICE == "cpu":
                    raise RuntimeError("Qwen3-TTS 1.7B requer worker RunPod com GPU CUDA.")
                print(f"[CTEC-QWEN] build={BUILD} carregando {MODEL_ID} em {DEVICE}", flush=True)
                _MODEL = Qwen3TTSModel.from_pretrained(
                    MODEL_ID,
                    device_map=DEVICE,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                print(f"[CTEC-QWEN] modelo carregado | build={BUILD}", flush=True)
    return _MODEL

def download(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "CTEC-Qwen3/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        total = 0
        with path.open("wb") as f:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_REFERENCE_BYTES:
                    raise ValueError("Áudio de referência excede o limite permitido.")
                f.write(block)

def save_reference_audio(data: dict[str, Any], root: Path) -> Path:
    # Aceita o contrato antigo do CTEC e aliases para facilitar a migração.
    b64 = (
        data.get("reference_audio_base64")
        or data.get("referenceAudioBase64")
        or data.get("audio_prompt_base64")
    )
    url = (
        data.get("reference_audio_url")
        or data.get("referenceAudioUrl")
        or data.get("audio_prompt_url")
    )
    voice_id = str(data.get("voice_id") or data.get("voiceId") or "").strip()

    voice = data.get("voice")
    if isinstance(voice, dict):
        voice_id = str(voice.get("voiceId") or voice.get("voice_id") or voice_id).strip()
        b64 = voice.get("referenceAudioBase64") or voice.get("reference_audio_base64") or b64
        url = voice.get("referenceAudioUrl") or voice.get("reference_audio_url") or url

    if not url and voice_id:
        url = load_voice_library().get(voice_id)

    source = root / "reference_input"
    if b64:
        payload = str(b64)
        if "," in payload and "base64" in payload[:100].lower():
            payload = payload.split(",", 1)[1]
        raw = base64.b64decode(payload, validate=False)
        if len(raw) > MAX_REFERENCE_BYTES:
            raise ValueError("Áudio de referência excede o limite permitido.")
        source.write_bytes(raw)
        return source

    if url:
        download(str(url), source)
        return source

    raise ValueError(
        "Nenhuma referência de voz chegou ao worker. "
        "Envie voice_id configurado na biblioteca, reference_audio_url ou reference_audio_base64."
    )

def prepare_reference(source: Path, root: Path) -> Path:
    # V4: mantém a referência inteira dentro do teto de 12 s, mas reserva
    # 500 ms finais de silêncio real. No ICL isso impede que o contexto termine
    # em um fonema da voz de referência, reduzindo o vazamento "pã" no início
    # da geração seguinte sem alterar timbre, sotaque ou modo de clonagem.
    target = root / "reference.wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
        "-ac", "1", "-ar", "24000", "-t", "11.5",
        "-af", "apad=pad_dur=0.5", "-t", "12",
        "-c:a", "pcm_s16le", str(target),
    ]
    subprocess.run(cmd, check=True)
    if not target.exists() or target.stat().st_size < 1024:
        raise RuntimeError("Falha ao preparar o áudio de referência.")
    return target

def normalize_law_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"§\s*único", "Parágrafo único", text, flags=re.I)
    text = re.sub(
        r"§\s*(\d+)(?:\s*[º°])?",
        lambda m: f"Parágrafo {num2words(int(m.group(1)), lang='pt_BR', to='ordinal')}",
        text,
    )
    text = re.sub(
        r"\bArts?\.\s*(\d+)(?:\s*[º°])?",
        lambda m: f"Artigo {num2words(int(m.group(1)), lang='pt_BR', to='ordinal' if int(m.group(1)) <= 9 else 'cardinal')}",
        text,
        flags=re.I,
    )
    roman = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
    def roman_int(s: str) -> int:
        total, prev = 0, 0
        for ch in reversed(s.upper()):
            cur = roman.get(ch, 0)
            if cur < prev: total -= cur
            else: total += cur; prev = cur
        return total
    text = re.sub(
        r"\b(Título|Capítulo|Seção|Subseção|Livro|Parte)\s+([IVXLCDM]{1,12})\b",
        lambda m: f"{m.group(1)} {num2words(roman_int(m.group(2)), lang='pt_BR')}",
        text, flags=re.I,
    )
    text = re.sub(r"(?m)^\s*([IVXLCDM]{1,12})\s*[—–-]\s*",
                  lambda m: f"Inciso {num2words(roman_int(m.group(1)), lang='pt_BR')}. ", text)
    # Cabeçalhos em caixa alta viram fala natural sem alterar o texto original recebido.
    lines = []
    for line in text.splitlines():
        s = line.strip()
        letters = [c for c in s if c.isalpha()]
        if letters and len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) >= .85:
            s = s.lower()
            s = s[:1].upper() + s[1:]
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()

def split_text(text: str, limit: int = CHUNK_LIMIT) -> list[tuple[str, bool]]:
    limit = max(220, int(limit))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[tuple[str, bool]] = []
    for paragraph in paragraphs:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", paragraph) if s.strip()]
        current = ""
        for sentence in sentences:
            if len(sentence) > limit:
                words = sentence.split()
                for word in words:
                    candidate = f"{current} {word}".strip()
                    if current and len(candidate) > limit:
                        out.append((current, False))
                        current = word
                    else:
                        current = candidate
                continue
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > limit:
                out.append((current, False))
                current = sentence
            else:
                current = candidate
        if current:
            out.append((current, True))
            current = ""
    return out or [(text.strip(), True)]

def numpy_to_tensor(wav: np.ndarray) -> torch.Tensor:
    audio = np.asarray(wav, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return torch.from_numpy(audio).float().unsqueeze(0)


def _speech_token(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^\wÀ-ÿ]+", "", value, flags=re.UNICODE)
    return value.strip("_")

def clean_icl_leading_artifact(
    audio: torch.Tensor,
    sample_rate: int,
    expected_text: str,
    root: Path,
    chunk_index: int,
) -> tuple[torch.Tensor, float]:
    """
    Mitiga vazamento/artefato no início de blocos gerados em ICL.

    Estratégia conservadora:
    - transcreve apenas o áudio recém-gerado com timestamps por palavra;
    - procura o começo real do texto solicitado;
    - só corta quando há fala reconhecida ANTES desse começo;
    - preserva 60 ms antes da primeira palavra esperada para não comer fonema;
    - aplica fade-in curtíssimo para evitar clique de corte.
    """
    if audio.numel() == 0 or sample_rate <= 0:
        return audio, 0.0

    probe = root / f"boundary_probe_{chunk_index:04d}.wav"
    pcm = audio[0].detach().cpu().numpy().astype(np.float32)
    sf.write(str(probe), pcm, sample_rate)

    expected_tokens = [
        _speech_token(t)
        for t in re.findall(r"\S+", expected_text)
        if _speech_token(t)
    ]
    if not expected_tokens:
        return audio, 0.0

    try:
        model = get_whisper()
        segments, _ = model.transcribe(
            str(probe),
            language="pt",
            beam_size=3,
            best_of=3,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=True,
        )

        words = []
        for segment in segments:
            for word in (getattr(segment, "words", None) or []):
                token = _speech_token(getattr(word, "word", ""))
                if token:
                    words.append(
                        (
                            token,
                            float(getattr(word, "start", 0.0) or 0.0),
                            float(getattr(word, "end", 0.0) or 0.0),
                        )
                    )

        if not words:
            return audio, 0.0

        # Procura a primeira palavra do texto-alvo e confirma, quando possível,
        # com a segunda palavra para não cortar por uma coincidência.
        first = expected_tokens[0]
        second = expected_tokens[1] if len(expected_tokens) > 1 else None
        match_index = None

        for i, (token, _, _) in enumerate(words):
            if token != first:
                continue
            if second is None:
                match_index = i
                break
            if i + 1 < len(words) and words[i + 1][0] == second:
                match_index = i
                break

        # Se o ASR não localizou o começo do texto, não arriscamos cortar.
        if match_index is None:
            return audio, 0.0

        # V4: o "pã" pode ser curto demais para virar uma palavra no Whisper.
        # Nesse caso a primeira palavra esperada aparece como words[0], porém
        # começa alguns milissegundos depois do artefato. Antes o V3 devolvia o
        # áudio intacto quando match_index == 0. Agora usamos o timestamp da
        # própria primeira palavra como guarda: só removemos o que estiver antes
        # dela, preservando 60 ms de ataque.
        start_sec = max(0.0, words[match_index][1] - 0.060)
        if match_index == 0 and start_sec < 0.080:
            return audio, 0.0
        cut_samples = int(round(start_sec * sample_rate))
        if cut_samples <= 0 or cut_samples >= audio.shape[-1]:
            return audio, 0.0

        cleaned = audio[:, cut_samples:].clone()

        # Fade-in de 8 ms somente na borda criada pelo corte.
        fade_samples = min(
            cleaned.shape[-1],
            max(1, int(round(sample_rate * 0.008))),
        )
        if fade_samples > 1:
            ramp = torch.linspace(
                0.0,
                1.0,
                fade_samples,
                dtype=cleaned.dtype,
                device=cleaned.device,
            )
            cleaned[:, :fade_samples] *= ramp

        prefix = " ".join(w[0] for w in words[:match_index])
        print(
            f"[CTEC-QWEN] boundary_clean bloco={chunk_index} "
            f"corte_ms={start_sec * 1000:.0f} prefixo_removido={prefix!r}",
            flush=True,
        )
        return cleaned, start_sec

    except Exception as exc:
        # Limpeza é proteção de qualidade; jamais derruba a geração principal.
        print(
            f"[CTEC-QWEN] boundary_clean bloco={chunk_index} "
            f"ignorado_por={type(exc).__name__}: {exc}",
            flush=True,
        )
        return audio, 0.0
    finally:
        try:
            probe.unlink(missing_ok=True)
        except Exception:
            pass

def write_pcm16(writer: wave.Wave_write, audio: torch.Tensor) -> int:
    pcm = (audio[0].clamp(-1, 1) * 32767).round().to(torch.int16).cpu().numpy()
    writer.writeframesraw(pcm.tobytes())
    return int(pcm.size)

def silence(writer: wave.Wave_write, sample_rate: int, ms: int) -> None:
    count = max(0, int(sample_rate * ms / 1000))
    if count:
        writer.writeframesraw(b"\x00\x00" * count)

def atempo_filter(speed: float) -> str | None:
    speed = clamp(speed, 0.75, 1.25)
    if abs(speed - 1.0) < 0.01:
        return None
    return f"atempo={speed:.4f}"

def encode_output(path: Path) -> str:
    raw = path.read_bytes()
    encoded = base64.b64encode(raw)
    if len(encoded) > MAX_RESULT_BASE64_BYTES:
        raise ValueError(
            "O áudio final excede o limite de retorno Base64. "
            "Use a rota de projeto longo do CTEC."
        )
    return encoded.decode("ascii")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def get_whisper():
    global _WHISPER
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper não está disponível. "
            "Ele é necessário para transcrever a referência no modo ICL pt-BR."
        )
    if _WHISPER is None:
        with _WHISPER_LOCK:
            if _WHISPER is None:
                compute_type = "float16" if DEVICE.startswith("cuda") else "int8"
                whisper_device = "cuda" if DEVICE.startswith("cuda") else "cpu"
                print(
                    f"[CTEC-QWEN] carregando Whisper={WHISPER_MODEL_SIZE} "
                    f"device={whisper_device} compute_type={compute_type}",
                    flush=True,
                )
                _WHISPER = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device=whisper_device,
                    compute_type=compute_type,
                )
    return _WHISPER

def normalize_reference_transcript(text: str) -> str:
    # Mantém o conteúdo reconhecido; apenas remove espaços quebrados.
    return re.sub(r"\s+", " ", str(text or "")).strip()

def transcribe_reference_ptbr(reference: Path) -> tuple[str, str]:
    """
    Transcreve a MESMA referência curta enviada ao Qwen.
    O texto reconhecido é usado como ref_text no ICL completo.
    Cache por SHA-256 evita retranscrever a mesma voz em cada geração.
    """
    key = file_sha256(reference)

    with _REFERENCE_TEXT_CACHE_LOCK:
        cached = _REFERENCE_TEXT_CACHE.get(key)
    if cached:
        print(
            f"[CTEC-QWEN] referência ICL: transcrição em cache "
            f"hash={key[:12]} chars={len(cached)}",
            flush=True,
        )
        return cached, key

    model = get_whisper()
    segments, info = model.transcribe(
        str(reference),
        language="pt",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    transcript = normalize_reference_transcript(
        " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    )

    if len(transcript) < 3:
        raise RuntimeError(
            "Não foi possível obter uma transcrição utilizável do áudio de referência "
            "para a clonagem ICL."
        )

    with _REFERENCE_TEXT_CACHE_LOCK:
        _REFERENCE_TEXT_CACHE[key] = transcript

    detected = getattr(info, "language", None) or "pt"
    probability = getattr(info, "language_probability", None)
    print(
        f"[CTEC-QWEN] referência ICL transcrita | hash={key[:12]} "
        f"idioma={detected} prob={probability} chars={len(transcript)} "
        f"texto={transcript!r}",
        flush=True,
    )
    return transcript, key

def capabilities() -> dict[str, Any]:
    return {
        "status": "ok",
        "engine": "qwen3_tts",
        "build": BUILD,
        "contract_version": WORKER_CONTRACT_VERSION,
        "model": MODEL_ID,
        "device": DEVICE,
        "voice_clone": True,
        "reference_max_seconds": 12,
        "voice_clone_mode": "icl_full_ref_audio_plus_ref_text",
        "boundary_cleanup": "asr_expected_text_guard_v2_plus_ref_tail_silence",
        "reference_transcription": "faster_whisper_pt",
        "target_locale": "pt-BR",
        "languages": list(LANGUAGES.keys()),
        "chatterbox": False,
    }

def generate(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("input") or {}
    action = str(data.get("action") or "generate").strip().lower()
    if action in {"health", "config", "capabilities"}:
        return capabilities()

    original_text = str(data.get("text") or "").strip()
    if len(original_text) < 3:
        raise ValueError("O texto precisa ter pelo menos 3 caracteres.")
    if len(original_text) > MAX_TEXT_CHARS:
        raise ValueError(f"O texto excede {MAX_TEXT_CHARS} caracteres.")

    language_id = str(data.get("language_id") or data.get("languageId") or "pt").lower()
    language = LANGUAGES.get(language_id, "Portuguese")
    profile = str(data.get("profile") or "law_natural")
    speed = clamp(data.get("speed") or 1.0, 0.75, 1.25)
    text_mode = "law" if profile.startswith("law_") or to_bool(data.get("normalize_legal_text"), False) else "general"
    prepared_text = normalize_law_text(original_text) if text_mode == "law" else original_text
    chunks = split_text(prepared_text)

    request_id = str(data.get("requestId") or data.get("request_id") or "")
    text_hash = str(data.get("textHash") or data.get("text_hash") or sha256_text(original_text))

    model = get_model()
    with _GENERATION_LOCK, tempfile.TemporaryDirectory(prefix="ctec_qwen3_") as tmp:
        root = Path(tmp)
        source = save_reference_audio(data, root)
        reference = prepare_reference(source, root)

        # PT-BR V2:
        # usa ICL completo (áudio + transcrição da própria referência).
        # Isso fornece ao Qwen, além do timbre, o padrão fonético/prosódico
        # presente na fala brasileira usada como referência.
        reference_text, reference_hash = transcribe_reference_ptbr(reference)
        clone_prompt = model.create_voice_clone_prompt(
            ref_audio=str(reference),
            ref_text=reference_text,
            x_vector_only_mode=False,
        )
        print(
            f"[CTEC-QWEN] clone ICL completo | locale=pt-BR "
            f"ref_hash={reference_hash[:12]} ref_chars={len(reference_text)}",
            flush=True,
        )

        raw_wav = root / "raw.wav"
        sample_rate = None
        with wave.open(str(raw_wav), "wb") as writer:
            for index, (chunk, paragraph_end) in enumerate(chunks, 1):
                runpod.serverless.progress_update(job, f"Qwen3-TTS: bloco {index} de {len(chunks)}")
                print(
                    f"[CTEC-QWEN] build={BUILD} bloco={index}/{len(chunks)} "
                    f"chars={len(chunk)} profile={profile}",
                    flush=True,
                )
                wavs, sr = model.generate_voice_clone(
                    text=chunk,
                    language=language,
                    voice_clone_prompt=clone_prompt,
                    max_new_tokens=4096,
                )
                if not wavs:
                    raise RuntimeError(f"Qwen3-TTS não devolveu áudio no bloco {index}.")
                if sample_rate is None:
                    sample_rate = int(sr)
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(sample_rate)
                elif int(sr) != sample_rate:
                    raise RuntimeError("Qwen3-TTS alterou o sample rate entre blocos.")
                tensor = numpy_to_tensor(wavs[0])

                # Qwen ICL pode vazar um pequeno fragmento antes do texto-alvo
                # em cada nova geração. Remove apenas prefixo confirmado por ASR.
                tensor, boundary_trim_seconds = clean_icl_leading_artifact(
                    tensor,
                    sample_rate,
                    chunk,
                    root,
                    index,
                )

                write_pcm16(writer, tensor)
                silence(writer, sample_rate, 520 if paragraph_end else 260)

        if sample_rate is None:
            raise RuntimeError("Nenhum áudio foi gerado.")

        output_format = str(data.get("output_format") or "mp3").lower()
        if output_format not in {"mp3", "wav"}:
            output_format = "mp3"
        final = root / f"ctec-voz-qwen3.{output_format}"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_wav)]
        filt = atempo_filter(speed)
        if filt:
            cmd += ["-filter:a", filt]
        if output_format == "mp3":
            cmd += ["-codec:a", "libmp3lame", "-b:a", "192k", str(final)]
            mime = "audio/mpeg"
        else:
            cmd += ["-codec:a", "pcm_s16le", str(final)]
            mime = "audio/wav"
        subprocess.run(cmd, check=True)

        with sf.SoundFile(str(final)) as af:
            duration = round(len(af) / af.samplerate, 2)

        return {
            "status": "ok",
            "action": "generate",
            "engine": "qwen3_tts",
            "build": BUILD,
            "contract_version": WORKER_CONTRACT_VERSION,
            "request_id": request_id,
            "text_hash": text_hash,
            "audio_base64": encode_output(final),
            "mime_type": mime,
            "file_name": final.name,
            "sample_rate": sample_rate,
            "duration_seconds_estimate": duration,
            "device": DEVICE,
            "model": MODEL_ID,
            "chunks": len(chunks),
            "prepared_text": prepared_text,
            "reference_used": True,
            "clone_mode": "icl_full",
            "target_locale": "pt-BR",
            "reference_transcript": reference_text,
            "reference_hash": reference_hash,
            "settings": {
                "profile": profile,
                "speed": speed,
                "language": language,
                "locale": "pt-BR",
                "reference_language": "pt",
                "text_mode": text_mode,
                "engine": "qwen3_tts",
            },
        }

if __name__ == "__main__":
    print(f"[CTEC-QWEN] iniciando | build={BUILD} | model={MODEL_ID} | device={DEVICE}", flush=True)
    runpod.serverless.start({"handler": generate})
