import os
import json
from io import BytesIO
from pathlib import Path

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
import clip
import faiss
import pandas as pd

TEST_SIZE = 20

FAISS_INDEX_PATH = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Embeddings/faiss.index"
INDEX_MAP_PATH = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Embeddings/clip_index_map.json"
META_INDEX_PATH = "C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Embeddings/meta_index.json"

TOP_K = 10
VOTE_TOP_N = 5
MULTI_CROP = True
CROP_BOXES = [
    (0.0,0.0,1.0,1.0), # full
    (0.1,0.1,0.9,0.9), # center crop
    (0.0,0.1,0.5,0.9), # laft half
    (0.5,0.1,1.0,0.9) # right half
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = Flask(__name__)

clip_model = None
preprocess = None
faiss_index = None
index_map = None
meta_index = None
test_dataset = None

def load_resources():
    global clip_model, preprocess, faiss_index, index_map, meta_index, test_dataset
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    print("Loading FAISS index...")
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at: {FAISS_INDEX_PATH}")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    print("Loading index map...")
    with open(INDEX_MAP_PATH, "r") as f:
        index_map = json.load(f)

    if os.path.exists(META_INDEX_PATH):
        with open(META_INDEX_PATH, "r") as f:
            meta_index = json.load(f)
    else:
        meta_index = {}

    print("Loading Dataset for Testing...")
    dataset = pd.read_csv("C:/Users/HP/Desktop/VS_Movie_Recognition/Data/Metadata.csv")
    test_dataset = dataset.sample(n=TEST_SIZE, random_state=42)

    print("Resources loaded.")

def image_to_embedding(image: Image.Image):
    """
    Preprocesses PIL image using CLIP preprocess and returns normalized embedddings (numpy float32).
    """
    img = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = clip_model.encode_image(img)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].astype("float32")

def multi_crop_embeddings(image: Image.Image):
    """
    Returns a list of embeddings from different crops (including full image).
    """
    embeddings = []
    w, h = image.size
    for (l, t, r, b) in CROP_BOXES:
        left = int(l*w)
        top = int(t*h)
        right = int(r*w)
        bottom = int(b*h)
        crop = image.crop((left, top, right, bottom))
        emb = image_to_embedding(crop)
        embeddings.append(emb)
    return embeddings

def vote_from_neighbours(neighbour_indices, neighbour_scores, top_n=5):
    """
    Given lists of neighbor indices and scores, perform a weighted vote to predict movie.
    neighbor_indices: 1D np.array of retrieved indices (int)
    neighbor_scores: 1D np.array of similarity scores (float)
    """
    # map index -> (movie_name -> accumulated_score)
    vote_map = {}
    for idx, score in zip(neighbour_indices, neighbour_scores):
        record = index_map[idx]
        movie = record["movie_name"]
        vote_map.setdefault(movie, 0.0)
        vote_map[movie] += float(score)

    # sort by score
    sorted_votes = sorted(vote_map.items(), key=lambda x:x[1], reverse=True)
    if not sorted_votes:
        return None, 0.0

    top_movie, top_score = sorted_votes[0]
    total_score = sum(v for _, v in sorted_votes)
    confidence = top_score / total_score if total_score>0 else 0.0

    # prepare top k list
    top_k_movies = [{"movie_name":m, "score":s} for m, s, in sorted_votes[:top_n]]
    return {"prediction":top_movie, "vote_score":top_score, "confidence":confidence, "top_movies":top_k_movies}



def predict_movie_core(image_path):
    image = Image.open(image_path).convert("RGB")

    if MULTI_CROP:
        q_embeddings = multi_crop_embeddings(image)
    else:
        q_embeddings = [image_to_embedding(image)]

    agg_neighbour_indices = []
    agg_neighbour_scores = []

    for q in q_embeddings:
        q = np.expand_dims(q.astype("float32"), axis=0)
        D, I = faiss_index.search(q, TOP_K)
        D = D[0]
        I = I[0]
        agg_neighbour_indices.extend(I.tolist())
        agg_neighbour_scores.extend(D.tolist())

    agg_neighbour_indices = np.array(agg_neighbour_indices, dtype=int)
    agg_neighbour_scores = np.array(agg_neighbour_scores, dtype=float)

    sort_idx = np.argsort(-agg_neighbour_scores)
    agg_neighbour_indices = agg_neighbour_indices[sort_idx]
    agg_neighbour_scores = agg_neighbour_scores[sort_idx]

    top_n = min(len(agg_neighbour_indices), VOTE_TOP_N)
    chosen_indices = agg_neighbour_indices[:top_n]
    chosen_scores = agg_neighbour_scores[:top_n]

    prediction_info = vote_from_neighbours(chosen_indices, chosen_scores, top_n=10)

    return prediction_info



@app.route("/results")
def show_results():
    rows = []
    count = 0

    for movie_name, image_path, timestamp in test_dataset.itertuples(index=False, name=None):
        pred = predict_movie_core(image_path)
        pred_movie = pred["prediction"] if pred else "Unknown"
        conf = round(pred["confidence"] * 100, 2) if pred else 0

        if pred_movie == movie_name: count += 1

        rows.append(f"""
            <tr>
                <td>{image_path}</td>
                <td>{movie_name}</td>
                <td>{timestamp}</td>
                <td>{pred_movie}</td>
                <td>{conf}%</td>
            </tr>
        """)

    html_rows = "\n".join(rows)

    return f"""
    <html>
    <head>
        <title>Dataset Predictions</title>
        <style>
            table, th, td {{
                border: 1px solid black;
                border-collapse: collapse;
                padding: 8px;
            }}
        </style>
    </head>
    <body>
        <h2>Predictions for Sampled Dataset</h2>
        <h3>Accuracy: {(count*100)/TEST_SIZE}</h3>
        <table>
            <tr>
                <th>Image Path</th>
                <th>Ground Truth Movie</th>
                <th>Timestamp</th>
                <th>Predicted Movie</th>
                <th>Confidence</th>
            </tr>
            {html_rows}
        </table>
    </body>
    </html>
    """

if __name__ == "__main__":
    load_resources()
    app.run(host="0.0.0.0", port=5000, debug=True)