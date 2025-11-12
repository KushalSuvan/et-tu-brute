import json
import os
import faiss
import pickle
import numpy as np 
from sentence_transformers import SentenceTransformer

INDEX_FILE = "pipeline/cache/vector.index"
META_FILE = "pipeline/cache/metadata.pkl"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
ACTS_PATH = "pipeline/data/julius_chunks.jsonl"


def get_faiss():
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        print("✅ Loading existing FAISS index and metadata...")
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    print("🧩 FAISS index not found. Building from documents.")

    model = SentenceTransformer(MODEL_NAME)
    acts = get_acts()
    embeddings = model.encode([act['text'] for act in acts], normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    id_to_meta = {i: acts[i] for i in range(len(acts))}

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(id_to_meta, f)

    print("💾 Index and metadata saved.")
    return index, id_to_meta


def get_acts():
    acts = []
    with open(ACTS_PATH, "r", encoding="utf-8") as f:
        for act in f:
            acts.append(json.loads(act))

    return acts
