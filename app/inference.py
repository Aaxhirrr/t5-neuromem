from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_MODEL_NAME = "t5-small"
_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _load():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
        _model.to(_device)
        _model.eval()

def build_prompt(query: str, chunks: List[Dict], max_chars_per_chunk: int = 600, max_chunks: int = 5) -> str:
    use = chunks[:max_chunks]
    lines = []
    for c in use:
        txt = (c.get("text") or "")[:max_chars_per_chunk]
        cid = c.get("chunk_id") or ""
        lines.append(f"[{cid}] {txt}")
    ctx = "\n".join(lines)
    prompt = (
        "You are a helpful assistant. Use ONLY the context to answer.\n"
        "Write 2-4 concise sentences. Include bracketed citations like [chunk_id] after facts.\n"
        f"Question: {query}\n\nContext:\n{ctx}\n\nAnswer:"
    )
    return prompt

@torch.inference_mode()
def generate_answer(query: str, chunks: List[Dict], max_input_tokens: int = 512, max_new_tokens: int = 128) -> str:
    _load()
    prompt = build_prompt(query, chunks)
    enc = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=min(max_input_tokens, 512)).to(_device)
    try:
        ids = _model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
            do_sample=False,
            early_stopping=True,
        )
    except Exception:
        ids = _model.generate(**enc, max_new_tokens=min(64, max_new_tokens))
    return _tokenizer.decode(ids[0], skip_special_tokens=True)

def count_tokens(text: str) -> int:
    _load()
    return len(_tokenizer.encode(text))
