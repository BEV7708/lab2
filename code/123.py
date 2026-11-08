from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_Unlimited_7B_v2")
audio_files = ["/path/to/eng_audio1.flac", "/path/to/deu_audio2.wav"]
lang = ["eng_Latn", "deu_Latn"]
transcriptions = pipeline.transcribe(audio_files, lang=lang, batch_size=2)