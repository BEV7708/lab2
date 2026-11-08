import os
import re
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ============================================================
# Базовые пути и загрузка текстов
# ============================================================
CODE_DIR = Path(__file__).resolve().parent
LANG_DIR = CODE_DIR.parent / "French"   # <-- при необходимости поменяйте имя папки


TEXT_CATEGORIES = [
    "01_tongue_twisters",
    "02_syllables",
    "03_repetitions",
    "04_complex_words",
    "05_numbers",
    "06_regular_texts",
]


def load_all_texts(model_dir: Path) -> pd.DataFrame:
    """Загружает все CSV-файлы категорий из model_dir.

    Ожидает колонки text и (опционально) source, id, category.
    Если id нет — генерирует его как '<NN>_<NNN>' на основе имени файла категории.
    """
    frames = []
    for cat in TEXT_CATEGORIES:
        path = model_dir / f"{cat}.csv"
        if not path.exists():
            matches = sorted(model_dir.glob(f"{cat.split('_')[0]}_*.csv"))
            if matches:
                path = matches[0]
            else:
                print(f"  [WARN] нет файла категории: {path.name}")
                continue

        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]

        if "text" not in df.columns:
            raise ValueError(
                f"{path.name}: нет колонки 'text'. "
                f"Реальные колонки: {df.columns.tolist()}"
            )

        # id: если нет — генерируем
        if "id" not in df.columns:
            prefix = cat.split("_")[0]  # '01', '02', ...
            df["id"] = [f"{prefix}_{i+1:03d}" for i in range(len(df))]
        df["id"] = df["id"].astype(str)

        if "category" not in df.columns:
            df["category"] = cat

        if "source" not in df.columns:
            df["source"] = ""

        frames.append(df[["id", "text", "category", "source"]])

    if not frames:
        return pd.DataFrame(columns=["id", "text", "category", "source"])

    full = pd.concat(frames, ignore_index=True)
    full["text"] = full["text"].astype(str)
    return full


# ============================================================
# G2P через Python API phonemizer (без CLI, обход проблемы с DLL на Windows)
# ============================================================
G2P_LANG_CODE = "fr-fr"
G2P_BACKEND = "espeak"
ESPEAK_DLL = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"


def run_phonemization(model_dir: Path, texts_df: pd.DataFrame):
    if texts_df.empty:
        return

    # Явно указываем phonemizer путь к DLL eSpeak NG
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(ESPEAK_DLL)
        print(f"  [G2P] EspeakWrapper.set_library({ESPEAK_DLL})")
    except Exception as e:
        print(f"  [WARN] EspeakWrapper.set_library failed: {e}")

    from phonemizer import phonemize as phonemize_fn

    texts = [str(t).strip() for t in texts_df["text"].tolist()]
    print(f"  [G2P] фонемизация {len(texts)} текстов через phonemizer API...")

    try:
        phonemes_list = phonemize_fn(
            texts,
            language=G2P_LANG_CODE,
            backend=G2P_BACKEND,
            strip=True,
            preserve_punctuation=False,
        )
    except Exception as e:
        raise RuntimeError(f"phonemizer API error: {e}")

    if isinstance(phonemes_list, str):
        phonemes_list = [phonemes_list]

    if len(phonemes_list) != len(texts_df):
        raise RuntimeError(
            f"phonemizer вернул {len(phonemes_list)} строк, ожидалось {len(texts_df)}"
        )

    rows = []
    for (_, text_row), phon_line in zip(texts_df.iterrows(), phonemes_list):
        rows.append({"id": text_row["id"], "phonemes": str(phon_line).strip()})

    ph_df = pd.DataFrame(rows)
    ph_df.to_csv(model_dir / "phonemes.csv", index=False, encoding="utf-8")

    empty = (ph_df["phonemes"] == "").sum()
    print(f"  [G2P] phonemes.csv сохранён ({len(ph_df)} строк, пустых: {empty})")
    if empty == len(ph_df):
        raise RuntimeError("Все фонемы пустые. Проверьте eSpeak NG.")


# ============================================================
# ASR и метрики
# ============================================================
_ASR_PIPE = None


def get_asr_pipeline():
    global _ASR_PIPE
    if _ASR_PIPE is None:
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1
        print("  [ASR] загрузка aihpi/FrWhisper...")
        _ASR_PIPE = pipeline(
            "automatic-speech-recognition",
            model="aihpi/FrWhisper",
            device=device,
            chunk_length_s=30,
        )
        print("  [ASR] модель загружена")
    return _ASR_PIPE


def _normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\sàâäéèêëîïôöùûüÿçœæ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compute_metrics(reference: str, hypothesis: str):
    import jiwer

    ref_norm = _normalize_text(reference)
    hyp_norm = _normalize_text(hypothesis)

    if not ref_norm:
        return 0.0, 0.0, ref_norm, hyp_norm

    try:
        cer = jiwer.cer(ref_norm, hyp_norm)
    except Exception:
        cer = 1.0
    try:
        wer = jiwer.wer(ref_norm, hyp_norm)
    except Exception:
        wer = 1.0

    return cer, wer, ref_norm, hyp_norm


def finish_asr(rows: list, audio_key: str = "audio_path"):
    asr = get_asr_pipeline()
    asr_name = "aihpi/FrWhisper"

    for row in tqdm(rows, desc="  ASR"):
        audio_path = row.get(audio_key)
        if row.get("status") == "error" or not audio_path:
            row["recognition"] = ""
            row["ref_normalized"] = _normalize_text(row.get("text", ""))
            row["hyp_normalized"] = ""
            row["cer"] = 1.0
            row["wer"] = 1.0
            row["asr_name"] = asr_name
            continue

        p = Path(str(audio_path))
        if not p.exists():
            row["status"] = "error"
            row["error"] = "audio file not found for ASR"
            row["recognition"] = ""
            row["ref_normalized"] = _normalize_text(row.get("text", ""))
            row["hyp_normalized"] = ""
            row["cer"] = 1.0
            row["wer"] = 1.0
            row["asr_name"] = asr_name
            continue

        try:
            result = asr(str(p))
            hyp = result["text"] if isinstance(result, dict) else result
            row["recognition"] = hyp
            cer, wer, ref_n, hyp_n = _compute_metrics(row.get("text", ""), hyp)
            row["ref_normalized"] = ref_n
            row["hyp_normalized"] = hyp_n
            row["cer"] = round(float(cer), 4)
            row["wer"] = round(float(wer), 4)
        except Exception as e:
            row["status"] = "error"
            row["error"] = f"ASR error: {str(e)[:180]}"
            row["recognition"] = ""
            row["ref_normalized"] = _normalize_text(row.get("text", ""))
            row["hyp_normalized"] = ""
            row["cer"] = 1.0
            row["wer"] = 1.0

        row["asr_name"] = asr_name

    return rows


# ============================================================
# Построение строки результата и сохранение
# ============================================================
def build_result_row(
    text_id, audio_path, model_name, text, status, error,
    references=None, duration=None, sample_rate=None,
):
    row = {
        "id": str(text_id),
        "audio_path": str(audio_path),
        "tts_name": model_name,
        "text": str(text),
        "status": status,
        "error": error,
        "references": references or "",
    }
    if duration is not None:
        row["duration_sec"] = round(float(duration), 3)
    elif sample_rate and Path(str(audio_path)).exists():
        try:
            import scipy.io.wavfile as wavfile
            sr, data = wavfile.read(str(audio_path))
            row["duration_sec"] = round(len(data) / sr, 3)
        except Exception:
            row["duration_sec"] = ""
    else:
        row["duration_sec"] = ""

    return row


def save_results(model_dir: Path, rows: list, model_name: str):
    df = pd.DataFrame(rows)
    out_path = model_dir / "results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  [SAVE] results.csv: {out_path} ({len(df)} строк)")