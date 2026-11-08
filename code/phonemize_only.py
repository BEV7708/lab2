from common import LANG_DIR, load_all_texts, run_phonemization

MODELS = ["MMS-TTS fra", "Supertonic 3", "OmniVoice"]

if __name__ == "__main__":
    for m in MODELS:
        d = LANG_DIR / m
        if not d.exists():
            print(f"[SKIP] {d}")
            continue
        print(f"=== G2P: {m} ===")
        df = load_all_texts(d)
        if df.empty:
            print("  нет текстов")
            continue
        run_phonemization(d, df)