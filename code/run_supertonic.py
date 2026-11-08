from tqdm import tqdm
from supertonic import TTS

from common import (
    LANG_DIR, load_all_texts, run_phonemization,
    build_result_row, finish_asr, save_results,
)

MODEL_NAME = "Supertonic 3"
SUPERTONIC_VOICE = "M1"
SUPERTONIC_LANG = "fr"


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

    print("  загрузка Supertonic...")
    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name=SUPERTONIC_VOICE)

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
            wav, dur = tts.synthesize(text, voice_style=style, lang=SUPERTONIC_LANG)
            tts.save_audio(wav, str(audio_path))
            duration = dur
        except Exception as e:
            status, error = "error", str(e)[:200]
            print(f"    [SYNTH error] id={text_id}: {e}")

        rows.append(build_result_row(
            text_id, audio_path, MODEL_NAME, text, status, error,
            duration=duration,
        ))

    finish_asr(rows)
    save_results(model_dir, rows, MODEL_NAME)


if __name__ == "__main__":
    main()