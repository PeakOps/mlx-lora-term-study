"""Kalite kapisi (protokol 2.2.3): egitim dizgesi ile cikarim istemi bayt bayt ayni mi? 5 ornek."""
import sys

from transformers import AutoTokenizer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import MODEL_4B, inference_prompt, train_record  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL_4B)
# Orijinal sablonla cikarim istemi (yamadan once) - yama cikarim tarafini degistirmemeli
orig_tpl = (MODEL_4B / "chat_template.jinja.orig")
orig_tpl = orig_tpl.read_text() if orig_tpl.exists() else None
samples = [
    ("Slurm kuyruğu yine doldu, head node'a bakar mısın?", "Slurm queue is full again, can you check the head node?", "tr", "en"),
    ("GPU'ları compute node'lara dağıttık.", "We distributed the GPUs to the compute nodes.", "tr", "en"),
    ("The AMI build failed on the head node.", "AMI build'i head node'da başarısız oldu.", "en", "tr"),
    ("Please check the budget before launching the cluster.", "Cluster'ı başlatmadan önce bütçeyi kontrol eder misin?", "en", "tr"),
    ("  Baştaki ve sondaki boşluk testi.  ", "Leading and trailing whitespace test.", "tr", "en"),
]
all_ok = True
for i, (s, t, a, b) in enumerate(samples):
    rec = train_record(s, t, a, b)
    full_str = tok.apply_chat_template(rec["messages"], tokenize=False)
    infer_str = inference_prompt(tok, s, a, b)
    # mlx-lm ChatDataset maskesi: messages[:-1] + add_generation_prompt=True
    mask_str = tok.apply_chat_template(rec["messages"][:-1], tokenize=False, add_generation_prompt=True)
    full_ids = tok.apply_chat_template(rec["messages"])
    pre_ids = tok.apply_chat_template(rec["messages"][:-1], add_generation_prompt=True)
    ok_bytes = full_str.encode().startswith(infer_str.encode()) and mask_str == infer_str
    if orig_tpl is not None:  # yamali ve orijinal sablonun cikarim istemi bayt bayt ayni mi
        infer_orig = tok.apply_chat_template([rec["messages"][0]], tokenize=False,
                                             add_generation_prompt=True, chat_template=orig_tpl)
        ok_bytes &= infer_orig.encode() == infer_str.encode()
    ok_tokens = full_ids[: len(pre_ids)] == pre_ids
    tail = full_str[len(infer_str):]
    ok_tail = tail == t.strip() + "<end_of_turn>\n"
    ok = ok_bytes and ok_tokens and ok_tail
    all_ok &= ok
    print(f"[{i}] {a}->{b} bayt_onek={ok_bytes} token_onek={ok_tokens} kuyruk={ok_tail!s:5} "
          f"istem_tok={len(pre_ids)} toplam_tok={len(full_ids)}  kuyruk={tail!r}")
print("\nORNEK CIKARIM ISTEMI (0):\n" + inference_prompt(tok, *samples[0][:1], "tr", "en"))
print("KAPI:", "GECTI" if all_ok else "KALDI")
sys.exit(0 if all_ok else 1)
