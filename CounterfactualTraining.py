# ======================================================
# Colab: SGCT with GPT-2 + HPM only (recommended values)
# ======================================================
!pip install -q transformers datasets scikit-learn

import os, json, random, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline,
    Trainer, TrainingArguments
)
from torch.utils.data import Dataset

# -------------------- CONFIG --------------------
from google.colab import drive
drive.mount('/content/drive')

# Your trained HPM (critic)
HPM_DIR   = "/content/drive/MyDrive/Colab Notebooks/NewBestModel/hallucination_detector_final"

# Source prompts for SGCT mining (needs: question, evidence)
# You can use your training split or any pool you want to augment.
SRC_JSON  = "/content/drive/MyDrive/Colab Notebooks/train_dataset.json"

# Generator model
GEN_MODEL = "gpt2"

# Sampling / thresholds
K_SAMPLES = 8
MAX_GEN   = 64       # max new tokens for answers
TEMP      = 0.8
TOP_P     = 0.9

TAU_POS   = 0.80     # keep factual if p >= 0.80
TAU_NEG   = 0.20     # keep hallucinated if p <= 0.20 (for unlikelihood)
USE_NEGATIVE_CONTRAST = True

# Training hyperparams (RECOMMENDED)
MAX_LEN_TOK = 512
BATCH_SIZE  = 8
EPOCHS      = 3
LR          = 2e-5
WARMUP_RATIO= 0.1

# Repro
random.seed(42); np.random.seed(42); torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if torch.cuda.is_available() else -1
print("Device:", DEVICE)

# -------------------- Load models --------------------
tok = AutoTokenizer.from_pretrained(GEN_MODEL)
gen = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(DEVICE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
gen.resize_token_embeddings(len(tok))

hpm = pipeline(
    "text-classification",
    model=HPM_DIR, tokenizer=HPM_DIR,
    device=DEVICE_ID
)

# -------------------- Load source data --------------------
with open(SRC_JSON, "r", encoding="utf-8") as f:
    pool = json.load(f)

# Keep only what we need; you can downsample if needed
src = []
for r in pool:
    q = str(r.get("question","")).strip()
    e = str(r.get("evidence","")).strip()
    if q:
        src.append({"question": q, "evidence": e})
print(f"Loaded {len(src)} SGCT prompts.")

# -------------------- HPM scoring (paired) --------------------

def hpm_p_factual(q, e, ans):
    left = f"Q: {q}  EVIDENCE: {e}" if e else q
    out = hpm({"text": left, "text_pair": ans})
    out = out[0] if isinstance(out, list) else out
    is_fact_label = out["label"] in {"FACTUAL","LABEL_1","factual"}
    p = out["score"] if is_fact_label else 1.0 - out["score"]
    return float(p)

# -------------------- Generate K candidates per prompt --------------------
def generate_candidates(q, e, k=K_SAMPLES):
    prompt = f"Question: {q}\nEvidence: {e}\nAnswer:"
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outs = gen.generate(
            **inputs,
            max_new_tokens=MAX_GEN,
            do_sample=True,
            temperature=TEMP,
            top_p=TOP_P,
            num_return_sequences=k,
            pad_token_id=tok.eos_token_id
        )
    texts = [tok.decode(o, skip_special_tokens=True) for o in outs]
    answers = [t.split("Answer:")[-1].strip() for t in texts]
    return prompt, answers

# -------------------- Mine positives/negatives --------------------
cf_rows = []  # each row: {"prompt", "target_pos", "target_neg"(opt)}
kept_pos, kept_neg = 0, 0

for item in tqdm(src, desc="SGCT mining"):
    q, e = item["question"], item["evidence"]
    prompt, cand_answers = generate_candidates(q, e)

    # score with HPM
    scored = [(a, hpm_p_factual(q, e, a)) for a in cand_answers]
    scored.sort(key=lambda x: x[1], reverse=True)  # high p first

    # positive (supervision)
    pos = next((a for a,p in scored if p >= TAU_POS), None)

    # negative (unlikelihood)
    neg = None
    if USE_NEGATIVE_CONTRAST:
        low_sorted = sorted(scored, key=lambda x: x[1])  # low p first
        neg = next((a for a,p in low_sorted if p <= TAU_NEG), None)

    if pos is not None:
        row = {"prompt": prompt, "target_pos": pos}
        kept_pos += 1
        if USE_NEGATIVE_CONTRAST and neg is not None and neg != pos:
            row["target_neg"] = neg
            kept_neg += 1
        cf_rows.append(row)

print(f"Kept {kept_pos} positives; {kept_neg} negatives; from {len(src)} prompts.")

# -------------------- Training dataset (mask prompt in labels) --------------------
class CFDS(Dataset):
    def __init__(self, rows, tokenizer, max_len=MAX_LEN_TOK):
        self.rows = rows; self.tok = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        prompt, target = r["prompt"], r["target_pos"]
        full = (prompt + " " + target)

        enc = self.tok(full, truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attn      = enc["attention_mask"].squeeze(0)

        # compute prompt length on tokens, then ignore them in loss
        prompt_ids = self.tok(prompt, truncation=True, max_length=self.max_len,
                              padding="max_length", return_tensors="pt")["input_ids"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = int((prompt_ids != self.tok.pad_token_id).sum().item())
        labels[:prompt_len] = -100  # mask prompt

        item = {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

        # optional negative for unlikelihood
        if "target_neg" in r:
            neg_full = (prompt + " " + r["target_neg"])
            neg_ids = self.tok(neg_full, truncation=True, max_length=self.max_len,
                               padding="max_length", return_tensors="pt")["input_ids"].squeeze(0)
            item["neg_ids"] = neg_ids
            item["prompt_len"] = prompt_len
        return item

train_ds = CFDS(cf_rows, tok)

# -------------------- Custom Trainer with unlikelihood --------------------
class ULTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=0):
        neg_ids = inputs.pop("neg_ids", None)
        prompt_len = inputs.pop("prompt_len", None)
        outputs = model(**inputs)
        loss = outputs.loss

        # optional unlikelihood on negative
        if USE_NEGATIVE_CONTRAST and neg_ids is not None and prompt_len is not None:
            with torch.no_grad():
                neg_attn = (neg_ids != tok.pad_token_id).long()
            logits = model(input_ids=neg_ids, attention_mask=neg_attn).logits
            # discourage realized negative tokens after prompt
            vocab_probs = torch.log_softmax(logits, dim=-1).exp()     # [B,T,V]
            neg_token_probs = vocab_probs.gather(-1, neg_ids.unsqueeze(-1)).squeeze(-1)  # [B,T]
            steps = neg_ids.size(1)
            mask = (torch.arange(steps, device=neg_ids.device) >= prompt_len).unsqueeze(0)
            ul_token_probs = torch.clamp(neg_token_probs, 1e-6, 1-1e-6)
            ul_loss = -torch.log(1.0 - ul_token_probs)
            ul_loss = (ul_loss * mask).sum() / (mask.sum() + 1e-6)

            loss = loss + 0.5 * ul_loss  # weight the penalty

        return (loss, outputs) if return_outputs else loss

# ----------------------------- Train -----------------------------
save_dir = "/content/drive/MyDrive/Colab Notebooks/sgct_gpt2_only"
args = TrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=20,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=1,   # set >1 if you OOM
)

trainer = ULTrainer(model=gen, args=args, train_dataset=train_ds, tokenizer=tok)
print("🚀 Training GPT-2 with SGCT (no corrector)…")
trainer.train()
print("✅ Done. Model dir:", save_dir)