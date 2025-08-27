import pprint
from app import memory_retrieve

def normalize_retrieve(res):
    # Return (chunks:list, meta:dict)
    if isinstance(res, tuple) or isinstance(res, list):
        # Two-element (chunks, meta)
        if len(res) == 2 and isinstance(res[1], dict):
            return res[0] or [], res[1] or {}
        # More than two elements -> assume first is chunks, second maybe meta
        if len(res) > 2:
            chunks = res[0] or []
            meta = res[1] if isinstance(res[1], dict) else {}
            return chunks, meta
        # Single-element sequence -> either a list of chunks or one chunk
        if len(res) == 1:
            if isinstance(res[0], list):
                return res[0], {}
            if isinstance(res[0], dict) and 'chunk_id' in res[0]:
                return [res[0]], {}
            # fallback
            return list(res), {}
    # dict: could be a single chunk or meta
    if isinstance(res, dict):
        if 'chunk_id' in res and 'text' in res:
            return [res], {}
        # assume it's meta
        return [], res
    # otherwise, unknown single object
    return [], {}

def pretty_print_chunk(c):
    if isinstance(c, dict):
        cid = c.get('chunk_id')
        print(f"CHUNK: {cid} | cosine={c.get('cosine')} | pagerank={c.get('pagerank')} | blend={c.get('blend')}")
        print("text:", (c.get('text') or "")[:400])
    elif isinstance(c, str):
        print("CHUNK (string):", c[:400])
    else:
        print("CHUNK (other):", repr(c)[:300])

# Run retrieval
res = memory_retrieve.retrieve_with_alpha('what is T5 and LoRA?', k=5)

chunks, meta = normalize_retrieve(res)

print('\\n===== META =====')
pprint.pprint(meta)
print('\\n===== CHUNKS (%d) =====' % (len(chunks)))
for i, c in enumerate(chunks):
    print(f'--- #{i+1} ---')
    pretty_print_chunk(c)
    print()
