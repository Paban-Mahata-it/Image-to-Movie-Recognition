import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import clip

METADATA_CSV = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Metadata.csv"
EMBEDDING_OUT = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Embeddings/clip_embeddings.npy"
INDEX_MAP_OUT = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Embeddings/clip_index_map.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model():
    print("Loading CLIP model (ViT-B.32)...")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    return model, preprocess

def generate_embeddings():
    os.makedirs(os.path.dirname(EMBEDDING_OUT), exist_ok=True)

    df = pd.read_csv(METADATA_CSV)
    print(f"Loaded {len(df)} frames from metadata.")

    model, preprocess = load_clip_model()
    model.eval()

    embeddings = []
    index_map = []

    for i,row in tqdm(df.iterrows(), total=len(df), desc="Embedding frames"):
        frame_path = row["frame_path"]

        try:
            image = Image.open(frame_path).convert("RGB")
        except:
            print(f"Failed to load image: {frame_path}")
            continue

        image_input = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embeddings.append(image_features.cpu().numpy()[0])

        index_map.append({
            "movie_name": row["movie_name"],
            "timestamp": row["timestamp"],
            "frame_path": row["frame_path"]
        })
        
    embeddings = np.array(embeddings)

    np.save(EMBEDDING_OUT, embeddings)

    with open(INDEX_MAP_OUT, "w") as f:
        json.dump(index_map, f, indent=4)

    print(f"Saved embeddings to: {EMBEDDING_OUT}")
    print(f"Saved index map to: {INDEX_MAP_OUT}")

if __name__ == "__main__":
    generate_embeddings()