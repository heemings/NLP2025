# main.py
import os
import csv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

# ============================================
# FastAPI 초기 설정
# ============================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Utility: 코사인 유사도
# ============================================
def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

# ============================================
# 1️⃣ Home
# ============================================
@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("../frontend/templates/play.html")


# ============================================
# 2️⃣ Drama API
# ============================================
@app.get("/drama")
def get_drama():
    drama = []
    path = "../data/drama_cliche.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drama.append({
                "title": row["title"],
                "keywords": row["keywords"],
                "cliche_score": float(row["cliche_score"])
            })
    drama = sorted(drama, key=lambda x: x["cliche_score"], reverse=True)
    return drama


# ============================================
# 3️⃣ Movie API
# ============================================
@app.get("/movie")
def get_movie():
    movie = []
    path = "../data/movie_cliche.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie.append({
                "title": row["title"],
                "keywords": row["keywords"],
                "cliche_score": float(row["cliche_score"])
            })
    movie = sorted(movie, key=lambda x: x["cliche_score"], reverse=True)
    return movie


# ============================================
# 4️⃣ Play API
# ============================================
@app.get("/play")
def get_play():
    play = []
    path = "../data/play_cliche.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            play.append({
                "title": row["title"],
                "keywords": row["keywords"],
                "cliche_score": float(row["cliche_score"])
            })
    play = sorted(play, key=lambda x: x["cliche_score"], reverse=True)
    return play


# ============================================
# 5️⃣ Drama Recommender (임베딩 메모리 저장)
# ============================================
DRAMA_CSV = "../data/drama_cliche.csv"
DRAMA_NUMERIC = ["semantic_density", "emotion_score", "entropy", "cliche_score"]
drama_df = pd.read_csv(DRAMA_CSV, encoding="utf-8-sig")
drama_df[DRAMA_NUMERIC] = drama_df[DRAMA_NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(0)
drama_df["title"] = drama_df["title"].fillna("").astype(str)
drama_df["description"] = drama_df["description"].fillna("").astype(str)
drama_df["keywords"] = drama_df["keywords"].fillna("").astype(str)

print("[INFO] Drama SBERT 모델 로딩...")
drama_model = SentenceTransformer("jhgan/ko-sbert-multitask")
drama_texts = [f"제목: {r['title']} / 설명: {r['description']} / 키워드: {r['keywords']}" 
               for _, r in drama_df.iterrows()]
drama_text_emb = drama_model.encode(drama_texts, show_progress_bar=True)
drama_scaler = StandardScaler()
drama_numeric_scaled = drama_scaler.fit_transform(drama_df[DRAMA_NUMERIC].values)
drama_embeddings = np.concatenate([drama_text_emb, drama_numeric_scaled], axis=1)
print("[INFO] Drama 임베딩 준비 완료")


@app.get("/drama_recommend")
def drama_recommend(title: str = Query(...), top_k: int = 10):
    idx_list = drama_df.index[drama_df["title"] == title].tolist()
    if not idx_list:
        return {"message": f"'{title}' 드라마를 찾을 수 없습니다."}
    q_idx = idx_list[0]
    q_emb = drama_embeddings[q_idx]
    sims = [(i, cosine(q_emb, drama_embeddings[i])) for i in range(len(drama_embeddings)) if i != q_idx]
    sims.sort(key=lambda x: x[1], reverse=True)
    top_idx = [i for i, _ in sims[:top_k]]
    top_sim = [s for _, s in sims[:top_k]]
    res = drama_df.iloc[top_idx].copy()
    res["similarity"] = top_sim
    return res[["title", "similarity"]].to_dict(orient="records")


# ============================================
# 6️⃣ Movie Recommender
# ============================================
MOVIE_CSV = "../data/movie_cliche.csv"
MOVIE_NUMERIC = ["semantic_density", "emotion_score", "entropy", "cliche_score"]
movie_df = pd.read_csv(MOVIE_CSV, encoding="utf-8-sig")
movie_df[MOVIE_NUMERIC] = movie_df[MOVIE_NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(0)
movie_df["title"] = movie_df["title"].fillna("").astype(str)
movie_df["keywords"] = movie_df["keywords"].fillna("").astype(str)
movie_df["cleaned_genre"] = movie_df["cleaned_genre"].fillna("").astype(str)

print("[INFO] Movie SBERT 모델 로딩...")
movie_model = SentenceTransformer("jhgan/ko-sbert-multitask")
movie_texts = [f"제목: {r['title']} / 장르: {r['cleaned_genre']} / 키워드: {r['keywords']}" 
               for _, r in movie_df.iterrows()]
movie_text_emb = movie_model.encode(movie_texts, show_progress_bar=True)
movie_scaler = StandardScaler()
movie_numeric_scaled = movie_scaler.fit_transform(movie_df[MOVIE_NUMERIC].values)
movie_embeddings = np.concatenate([movie_text_emb, movie_numeric_scaled], axis=1)
print("[INFO] Movie 임베딩 준비 완료")


@app.get("/movie_recommend")
def movie_recommend(title: str = Query(...), top_k: int = 10):
    idx_list = movie_df.index[movie_df["title"] == title].tolist()
    if not idx_list:
        return {"message": f"'{title}' 영화를 찾을 수 없습니다."}
    q_idx = idx_list[0]
    q_emb = movie_embeddings[q_idx]
    sims = [(i, cosine(q_emb, movie_embeddings[i])) for i in range(len(movie_embeddings)) if i != q_idx]
    sims.sort(key=lambda x: x[1], reverse=True)
    top_idx = [i for i, _ in sims[:top_k]]
    top_sim = [s for _, s in sims[:top_k]]
    res = movie_df.iloc[top_idx].copy()
    res["similarity"] = top_sim
    return res[["title", "cleaned_genre", "cliche_score", "similarity"]].to_dict(orient="records")


# ============================================
# 7️⃣ Play Recommender (build_recommendations.csv 기반)
# ============================================
PLAY_RECOMMEND_CSV = "../data/recommendations_top10.csv"
if os.path.exists(PLAY_RECOMMEND_CSV):
    play_rec_df = pd.read_csv(PLAY_RECOMMEND_CSV, encoding="utf-8-sig")
else:
    play_rec_df = None


@app.get("/play_recommend")
def play_recommend(title: str = Query(...), top_k: int = 10):
    if play_rec_df is None:
        return {"message": "recommendations_top10.csv 파일이 없습니다."}
    df_filtered = play_rec_df[play_rec_df["기준작품"] == title].sort_values("유사도", ascending=False).head(top_k)
    if df_filtered.empty:
        return {"message": f"'{title}' 작품을 찾을 수 없습니다."}
    return df_filtered.to_dict(orient="records")
