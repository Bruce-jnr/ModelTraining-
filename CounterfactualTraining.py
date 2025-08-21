
# HPM-Guided acting as a critic for SGCT with GPT-2 

!pip install -q transformers datasets scikit-learn

import os, json, random, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Dict, Tuple
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, pipeline,
    Trainer, TrainingArguments
)
from torch.utils.data import Dataset

# -------------------- CONFIG --------------------
from google.colab import drive
drive.mount('/content/drive')

# 1) Your trained HPM (critic) directory (HF format: config.json + model + tokenizer)
HPM_DIR   = "/content/drive/MyDrive/Colab Notebooks/hallucination_detector_final"

# 2) Seed prompts (needs at least 'question', optional 'evidence')
#    Accepts .json, .jsonl, or .csv; each row should have columns: question, evidence (answer ignored)
SRC_PATH  = "/content/drive/MyDrive/Colab Notebooks/merged_data_new/train_70.json"

# 3) Save locations
MINED_JSONL = "/content/drive/MyDrive/Colab Notebooks/sgct_mined.jsonl"
SAVE_DIR    = "/content/drive/MyDrive/Colab Notebooks/sgct_gpt2_only"

# 4) Generator model
GEN_MODEL = "gpt2"

# 5) Mining & thresholds
K_SAMPLES = 8                   # candidates per prompt
MAX_NEW_TOKENS = 64             # generation length for answers
TEMP, TOP_P = 0.8, 0.9

TAU_POS = 0.80                  # accept as factual if p >= 0.80
TAU_NEG = 0.20                  # accept as hallucinated if p <= 0.20
USE_NEGATIVE_CONTRAST = True    # enable unlikelihood on confident negatives

# 6) Training (thesis defaults)
MAX_LEN_TOKENS = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.10
UNLIKELIHOOD_WEIGHT = 0.5       # reduce to 0.25 if updates feel too strong

# 7) Reproducibility
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

hpm = pipeline("text-classification", model=HPM_DIR, tokenizer=HPM_DIR, device=DEVICE_ID)

# -------------------- Utilities --------------------
def load_seed_prompts(path: str, max_rows: int = None) -> List[Dict]:
    path = Path(path)
    rows = []
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows: break
                obj = json.loads(line)
                q = str(obj.get("question","")).strip()
                e = str(obj.get("evidence","")).strip() if obj.get("evidence") is not None else ""
                if q:
                    rows.append({"question": q, "evidence": e})
    elif path.suffix.lower() == ".json":
        data = json.load(open(path, "r", encoding="utf-8"))
        for i, obj in enumerate(data):
            if max_rows and i >= max_rows: break
            q = str(obj.get("question","")).strip()
            e = str(obj.get("evidence","")).strip() if obj.get("evidence") is not None else ""
            if q:
                rows.append({"question": q, "evidence": e})
    elif path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        if max_rows: df = df.head(max_rows)
        df["question"] = df["question"].astype(str)
        if "evidence" in df.columns:
            df["evidence"] = df["evidence"].fillna("").astype(str)
        else:
            df["evidence"] = ""
        rows = df[["question","evidence"]].to_dict("records")
    else:
        raise ValueError("SRC_PATH must be .json, .jsonl, or .csv")
    print(f"Loaded {len(rows)} seed prompts from {path.name}")
    return rows

def prompt_text(q: str, e: str) -> str:
    return f"Question: {q}\nEvidence: {e}\nAnswer:" if e else f"Question: {q}\nAnswer:"

def hpm_left(q: str, e: str) -> str:
    return f"Q: {q}  EVIDENCE: {e}" if e else q

def sample_k_answers(prompt: str, k: int = K_SAMPLES) -> List[str]:
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outs = gen.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMP,
            top_p=TOP_P,
            num_return_sequences=k,
            pad_token_id=tok.eos_token_id
        )
    texts = [tok.decode(o, skip_special_tokens=True) for o in outs]
    # keep only answer portion after 'Answer:'
    answers = [t.split("Answer:")[-1].strip() for t in texts]
    return answers

def factual_prob_from_hpm_output(out: Dict) -> float:
    lab = out["label"]
    is_fact = lab in {"FACTUAL", "LABEL_1", "factual"}
    p = out["score"] if is_fact else 1.0 - out["score"]
    return float(p)

# -------------------- Mining (batched HPM) --------------------
def mine_sgct_rows(src_prompts: List[Dict]) -> List[Dict]:
    prompts = []
    lefts = []
    answers_all = []
    meta = []  # (idx_prompt, answer)

    # 1) generate K candidates per prompt
    print("Generating candidates...")
    for idx, item in enumerate(tqdm(src_prompts)):
        q, e = item["question"], item["evidence"]
        ptxt = prompt_text(q, e)
        prompts.append(ptxt)
        ltxt = hpm_left(q, e)
        lefts.append(ltxt)
        ans_list = sample_k_answers(ptxt, K_SAMPLES)
        for a in ans_list:
            answers_all.append({"text": ltxt, "text_pair": a})
            meta.append((idx, a))

    # 2) HPM scoring in batches
    print("Scoring candidates with HPM (batched)...")
    BATCH = 64  # try 128 if VRAM allows
    pfacts = []
    for i in tqdm(range(0, len(answers_all), BATCH)):
        batch = answers_all[i:i+BATCH]
        outs = hpm(batch, batch_size=BATCH, truncation=True)
        pfacts.extend([factual_prob_from_hpm_output(o) for o in outs])

    # 3) regroup by prompt idx
    from collections import defaultdict
    by_idx = defaultdict(list)
    for (idx, ans), p in zip(meta, pfacts):
        by_idx[idx].append((ans, p))

    # 4) select positives/negatives
    rows = []
    kept_pos = kept_neg = 0
    for idx in range(len(prompts)):
        scored = by_idx[idx]
        if not scored: continue
        # sort desc by p
        scored.sort(key=lambda x: x[1], reverse=True)
        pos = next((a for a,p in scored if p >= TAU_POS), None)

        neg = None
        if USE_NEGATIVE_CONTRAST:
            low_sorted = sorted(scored, key=lambda x: x[1])  # asc
            neg = next((a for a,p in low_sorted if p <= TAU_NEG), None)

        if pos is not None:
            row = {"prompt": prompts[idx], "target_pos": pos}
            kept_pos += 1
            if USE_NEGATIVE_CONTRAST and neg is not None and neg != pos:
                row["target_neg"] = neg
                kept_neg += 1
            rows.append(row)

    print(f"Selected {kept_pos} positives; {kept_neg} negatives from {len(prompts)} prompts.")
    if kept_pos == 0:
        print("WARNING: No positives kept. Consider increasing K_SAMPLES or lowering TAU_POS slightly.")
    return rows

# -------------------- Dataset (masked prompt loss) --------------------
class CFDS(Dataset):
    def __init__(self, rows: List[Dict], tokenizer: AutoTokenizer, max_len: int = MAX_LEN_TOKENS):
        self.rows, self.tok, self.max_len = rows, tokenizer, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        prompt = r["prompt"]
        pos = r["target_pos"][0] if isinstance(r["target_pos"], tuple) else r["target_pos"]
        full = prompt + " " + pos

        enc = self.tok(full, truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attn      = enc["attention_mask"].squeeze(0)

        # compute prompt length then mask prompt tokens in labels
        p_ids = self.tok(prompt, truncation=True, max_length=self.max_len,
                         padding="max_length", return_tensors="pt")["input_ids"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = int((p_ids != self.tok.pad_token_id).sum().item())
        labels[:prompt_len] = -100

        item = {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

        if "target_neg" in r:
            neg = r["target_neg"][0] if isinstance(r["target_neg"], tuple) else r["target_neg"]
            neg_full = prompt + " " + neg
            neg_ids = self.tok(neg_full, truncation=True, max_length=self.max_len,
                               padding="max_length", return_tensors="pt")["input_ids"].squeeze(0)
            item["neg_ids"] = neg_ids
            item["prompt_len"] = prompt_len
        return item

# -------------------- Custom Trainer with Unlikelihood --------------------
class ULTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        neg_ids = inputs.pop("neg_ids", None)
        prompt_len = inputs.pop("prompt_len", None)
        outputs = model(**inputs)
        loss = outputs.loss

        # optional unlikelihood on negatives
        if USE_NEGATIVE_CONTRAST and neg_ids is not None and prompt_len is not None:
            with torch.no_grad():
                neg_attn = (neg_ids != tok.pad_token_id).long().to(model.device)
            logits = model(input_ids=neg_ids.to(model.device),
                           attention_mask=neg_attn).logits
            # discourage realized negative tokens after prompt
            # compute probs of realized tokens
            probs = torch.log_softmax(logits, dim=-1).exp()        # [B,T,V]
            tok_probs = probs.gather(-1, neg_ids.to(model.device).unsqueeze(-1)).squeeze(-1)  # [B,T]
            T = neg_ids.size(1)
            mask = (torch.arange(T, device=logits.device) >= prompt_len).unsqueeze(0)  # answer region
            tok_probs = torch.clamp(tok_probs, 1e-6, 1-1e-6)
            ul_loss = -torch.log(1.0 - tok_probs)
            ul_loss = (ul_loss * mask).sum() / (mask.sum() + 1e-6)
            loss = loss + UNLIKELIHOOD_WEIGHT * ul_loss

        return (loss, outputs) if return_outputs else loss

# -------------------- Run SGCT --------------------
# Load seeds
src_prompts = load_seed_prompts(SRC_PATH)

# Mine positives/negatives
rows = mine_sgct_rows(src_prompts)

# Save mined data for your appendix / reproducibility
Path(MINED_JSONL).parent.mkdir(parents=True, exist_ok=True)
with open(MINED_JSONL, "w", encoding="utf-8") as f:
    for r in rows:
        out = {
            "prompt": r["prompt"],
            "target_pos": r["target_pos"][0] if isinstance(r["target_pos"], tuple) else r["target_pos"],
        }
        if "target_neg" in r:
            out["target_neg"] = r["target_neg"][0] if isinstance(r["target_neg"], tuple) else r["target_neg"]
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
print(f"Saved mined SGCT dataset to: {MINED_JSONL}")

# Build dataset
train_ds = CFDS(rows, tok)

# Train
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=20,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=1,   # if OOM, set to 2 or 4
)

trainer = ULTrainer(model=gen, args=args, train_dataset=train_ds, tokenizer=tok)
print("🚀 Training GPT-2 with SGCT (HPM-guided)…")
trainer.train()
print("Done. Model saved in:", SAVE_DIR)
