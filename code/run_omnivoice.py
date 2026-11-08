import torch
import scipy.io.wavfile as wavfile
import numpy as np
from tqdm import tqdm

from common import (
    LANG_DIR, load_all_texts, run_phonemization,
    build_result_row, finish_asr, save_results,
)

MODEL_NAME = "OmniVoice"
OMNIVOICE_MODEL_ID = "k2-fsa/OmniVoice"
OMNIVOICE_NUM_STEP = 32
OMNIVOICE_SAMPLE_RATE = 24000

REF_AUDIO_REL = "references/reference_01.wav"
REF_TEXT_REL = "references/reference_01.txt"


def main():
    model_dir = LANG_DIR / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {MODEL_NAME} ===")

    ref_audio = model_dir / REF_AUDIO_REL
    ref_text_path = model_dir / REF_TEXT_REL
    if not ref_audio.exists():
        raise FileNotFoundError(
            f"Нет референсного аудио: {ref_audio}. "
            f"Положите WAV и (опционально) {REF_TEXT_REL}"
        )
    ref_text = ref_text_path.read_text(encoding="utf-8").strip() \
        if ref_text_path.exists() else None
    print(f"  референс: {ref_audio}  (текст: {'есть' if ref_text else 'нет'})")

    texts_df = load_all_texts(model_dir)
    if texts_df.empty:
        print("[SKIP] нет текстов")
        return
    print(f"  загружено текстов: {len(texts_df)}")

    run_phonemization(model_dir, texts_df)

    print("  загрузка OmniVoice...")
    from omnivoice import OmniVoice

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = OmniVoice.from_pretrained(
        OMNIVOICE_MODEL_ID, device_map=device, dtype=dtype
    )

    audio_dir = model_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    rows = []
    for _, row in tqdm(texts_df.iterrows(), total=len(texts_df), desc="  синтез"):
        text_id = str(row["id"])
        text = str(row["text"])
        audio_path = audio_dir / f"{text_id}.wav"
        status, error = "ok", ""
        duration = None
        try:
            audio = model.generate(
                text=text,
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                num_step=OMNIVOICE_NUM_STEP,
            )
            # OmniVoice возвращает list[np.ndarray], shape (T,) @ 24kHz [citation:1]
            if isinstance(audio, list):
                wav = audio[0]
            else:
                # fallback на случай tensor
                wav = audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
            if wav.ndim > 1:
                wav = wav.squeeze()
            wavfile.write(str(audio_path), rate=OMNIVOICE_SAMPLE_RATE, data=wav)
            duration = len(wav) / OMNIVOICE_SAMPLE_RATE
        except Exception as e:
            status, error = "error", str(e)[:200]
            print(f"    [SYNTH error] id={text_id}: {e}")

        rows.append(build_result_row(
            text_id, audio_path, MODEL_NAME, text, status, error,
            references=REF_AUDIO_REL, duration=duration,
            sample_rate=OMNIVOICE_SAMPLE_RATE,
        ))

    finish_asr(rows)
    save_results(model_dir, rows, MODEL_NAME)


if __name__ == "__main__":
    main()