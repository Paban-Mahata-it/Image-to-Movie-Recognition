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

def load_resources():
    global clip_model, preprocess, faiss_index, index_map, meta_index
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

# Home page + upload form
@app.route("/", methods=["GET"])
def home():
    return """
    <html>
    <head><title>Movie Recognition</title></head>
    <body>
        <h2>Upload an image to predict the movie</h2>
        <form action="/predict_movie" method="POST" enctype="multipart/form-data">
            <input type="file" name="image" required>
            <br><br>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """

@app.route("/predict_movie", methods=["POST"])
def predict_movie():
    if "image" not in request.files:
        return jsonify({"error":"No image file provided. Use from field 'image'."}), 400

    file = request.files["image"]
    try:
        image = Image.open(BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error":"Failed to read image file.", "details":str(e)}), 400
    
    # generate one or more query embeddings
    if MULTI_CROP:
        q_embeddings = multi_crop_embeddings(image)
    else:
        q_embeedings = [image_to_embedding(image)]

    # aggregate neighbours across crops
    agg_neighbour_indices = []
    agg_neighbour_scores = []

    for q in q_embeddings:
        q = np.expand_dims(q.astype("float32"), axis=0)
        D, I = faiss_index.search(q, TOP_K) # D: scores, I: indixes
        # D shape(1, top_k), I shape(1, top_k)
        D = D[0]
        I = I[0]
        agg_neighbour_indices.extend(I.tolist())
        agg_neighbour_scores.extend(D.tolist())

    # convert to numpy array
    agg_neighbour_indices = np.array(agg_neighbour_indices, dtype=int)
    agg_neighbour_scores = np.array(agg_neighbour_scores, dtype=float)

    # sort neighbours by score descending
    sort_idx = np.argsort(-agg_neighbour_scores) # sorted indices
    agg_neighbour_indices = agg_neighbour_indices[sort_idx]
    agg_neighbour_scores = agg_neighbour_scores[sort_idx]

    # top N neighbours for voting
    top_n = min(len(agg_neighbour_indices), VOTE_TOP_N)
    chosen_indices = agg_neighbour_indices[:top_n]
    chosen_scores = agg_neighbour_scores[:top_n]

    prediction_info = vote_from_neighbours(chosen_indices, chosen_scores, top_n=10)
    if prediction_info is None:
        return jsonify({"error":"No match found"}), 200

    q0 = np.expand_dims(q_embeddings[0].astype("float32"), axis=0)
    D0, I0 = faiss_index.search(q0, TOP_K)
    D0 = D0[0].tolist()
    I0 = I0[0].tolist()
    nearest_frames = []
    for idx, score in zip(I0, D0):
        rec = index_map[idx]
        nearest_frames.append({
            "movie_name": rec["movie_name"],
            "timestamp": rec.get("timestamp", ""),
            "frame_path": rec.get("frame_path", ""),
            "score": float(score)
        })

    result = {
        "prediction": prediction_info["prediction"],
        "confidence": float(prediction_info["confidence"]),
        "vote_score": float(prediction_info["vote_score"]),
        "top_candidates": prediction_info["top_movies"],
        "nearest_frames": nearest_frames
    }
    return jsonify(result), 200

if __name__ == "__main__":
    load_resources()
    app.run(host="0.0.0.0", port=5000, debug=False)