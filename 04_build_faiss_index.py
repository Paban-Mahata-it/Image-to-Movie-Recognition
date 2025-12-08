import os
import json
import numpy as np
import faiss
from pathlib import Path

EMBEDDINGS_PATH = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Embeddings/clip_embeddings.npy"
INDEX_MAP_PATH = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Embeddings/clip_index_map.json"
FAISS_OUT = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Embeddings/faiss.index"
META_OUT = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Embeddings/meta_index.json"

# Choose index mode here: "flat" for small datasets, "ivf" for large.
# - flat: IndexFlatIP (fast exact inner-product after normalized embeddings)
# - ivf:  IndexIVFFlat + IndexFlatIP as quantizer (approximate but scalable)
INDEX_MODE = "ivf"  # choose "flat" or "ivf"

# IVF parameter (only used if INDEX_MODE == "ivf")
N_LIST = 1024  # number of Voronoi cells (tune: sqrt(num_vectors) is a good starting point)

def load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}") # trigger an exception
    return np.load(path).astype("float32")

def build_index(embeddings, index_mode="flat", n_list=1024):
    d = embeddings.shape[1]

    if index_mode == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        meta = {"index_mode":"flat", "d":d, "n_total":int(embeddings.shape[0])}
        return index, meta
    elif index_mode == "ivf":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_list, faiss.METRIC_INNER_PRODUCT)
        # training required for ivf
        print("Training IVF index (this may take a while)...")
        index.train(embeddings)
        index.add(embeddings)
        meta = {
            "index_mode": "ivf",
            "d": d,
            "n_list": n_list,
            "n_total": int(embeddings.shape[0])
        }
        return index, meta
    else:
        return ValueError("Unknown index_mode. Choose 'flat' or 'ivf'")

def save_index(index, faiss_out, meta_out, meta):
    Path(os.path.dirname(faiss_out)).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, faiss_out)
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved FIASS index to: {faiss_out}")
    print(f"Savd index metadata to: {meta_out}")

def main():
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # if embeddings are normalized, normalize them here
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print(f"Normalizing embeddings with shape: {embeddings.shape}")
        embeddings = embeddings / (norms[:, None] + 1e-10)

    index, meta = build_index(embeddings, index_mode=INDEX_MODE, n_list=N_LIST)
    save_index(index, FAISS_OUT, META_OUT, meta)

if __name__ == "__main__":
    main()