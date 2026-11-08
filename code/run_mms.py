import torch
import scipy.io.wavfile as wavfile
from tqdm import tqdm
from transformers import VitsModel, AutoTokenizer

from common import (
    LANG_DIR, load_all_texts, run_phonemization,
    build_result_row, finish_asr, save_results,
)

MODEL_NAME = "MMS-TTS fra"
MMS_MODEL_ID = str(LANG_DIR / "MMS-TTS fra" / "model" / "models" / "facebook--mms-tts-fra" / "snapshots" / "master")
MMS_SEED = 42


def main():
    model_dir = LANG_DIR / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {MODEL_NAME} ===")

    texts_df = load_all_texts(model_dir)
    if texts_df.empty:
        print("[SKIP] нет текстов")
        return
    print(f"  загружено текстов: {len(texts_df)}")

    run_phonemization(model_dir, texts_df)

    print("  загрузка MMS-TTS...")
    model = VitsModel.from_pretrained(MMS_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MMS_MODEL_ID)
    sr = model.config.sampling_rate
    print(f"  sample_rate = {sr}")

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
            torch.manual_seed(MMS_SEED)
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform
            audio = output.squeeze().cpu().numpy()
            wavfile.write(str(audio_path), rate=sr, data=audio)
            duration = len(audio) / sr
        except Exception as e:
            status, error = "error", str(e)[:200]
            print(f"    [SYNTH error] id={text_id}: {e}")

        rows.append(build_result_row(
            text_id, audio_path, MODEL_NAME, text, status, error,
            duration=duration, sample_rate=sr,
        ))

    finish_asr(rows)
    save_results(model_dir, rows, MODEL_NAME)


if __name__ == "__main__":
    main()