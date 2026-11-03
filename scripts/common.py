"""Ortak yardimcilar: TranslateGemma istem/mesaj kurulumu. Egitim ve cikarim AYNI fonksiyonu kullanir."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_4B = ROOT / "models" / "translategemma-4b-it-8bit"


def user_msg(text: str, src: str, tgt: str) -> dict:
    return {"role": "user", "content": [
        {"type": "text", "source_lang_code": src, "target_lang_code": tgt, "text": text}]}


def train_record(src_text: str, tgt_text: str, src: str, tgt: str) -> dict:
    """mlx-lm ChatDataset bicimi. --mask-prompt ile kayip yalniz asistan kisminda."""
    return {"messages": [user_msg(src_text, src, tgt), {"role": "assistant", "content": tgt_text}]}


def inference_prompt(tok, text: str, src: str, tgt: str) -> str:
    return tok.apply_chat_template([user_msg(text, src, tgt)], tokenize=False,
                                   add_generation_prompt=True)
