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
import requests
from bs4 import BeautifulSoup
import urllib.parse

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

def get_movie_online_info(movie_name: str):
    query = urllib.parse.quote(movie_name)
    url = f"https://search.naver.com/search.naver?query={query}"

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")

        # 줄거리
        desc_tag = soup.select_one("div.cm_info_box span.desc._text")
        if not desc_tag:
            desc_tag = soup.select_one("span.desc._text")

        description = desc_tag.get_text(strip=True) if desc_tag else None

        # 포스터 이미지 후보들
        poster_selectors = [
            "div.cm_info_box a.thumb img",
            "a.thumb img",
            "div.thumb img",
            "a.thumb._item img",
            "img._img",
        ]

        poster_url = None
        for sel in poster_selectors:
            tag = soup.select_one(sel)
            if tag and tag.get("src"):
                poster_url = tag["src"]
                break

        return {"description": description, "poster": poster_url}

    except Exception as e:
        print("영화 크롤링 오류:", e)
        return {"description": None, "poster": None}


def get_drama_online_info(drama_name: str):
    query = urllib.parse.quote(drama_name)
    url = f"https://search.naver.com/search.naver?query={query}"

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")

        # ============================
        # 1) 줄거리: cm_info_box 안의 desc._text 우선
        # ============================
        desc_tag = soup.select_one("div.cm_info_box span.desc._text")
        if not desc_tag:  # 혹시 구조가 다를 때를 대비해서 fallback
            desc_tag = soup.select_one("span.desc._text")

        description = desc_tag.get_text(strip=True) if desc_tag else None

        # ============================
        # 2) 포스터 이미지: 대표 포스터 먼저
        # ============================
        poster_selectors = [
            "div.cm_info_box a.thumb img",  # 대표 포스터 (지금 보내준 HTML 구조)
            "a.thumb img",                  # 그 외 thumb 이미지
            "div.thumb img",                # thumb 박스 안 이미지
            "a.thumb._item img",            # 예전 구조
            "img._img",                     # 네이버 검색 기본 이미지
        ]

        poster_url = None
        for sel in poster_selectors:
            tag = soup.select_one(sel)
            if tag and tag.get("src"):
                poster_url = tag["src"]
                break

        print("[DEBUG] description:", (description[:40] + "...") if description else None)
        print("[DEBUG] poster_url:", poster_url)

        return {
            "description": description,
            "poster": poster_url
        }

    except Exception as e:
        print("크롤링 오류:", e)
        return {"description": None, "poster": None}




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

    # ===================
    # 1) CSV 검색
    # ===================
    idx_list = drama_df.index[drama_df["title"] == title].tolist()

    if idx_list:
        q_idx = idx_list[0]
        q_emb = drama_embeddings[q_idx]

        sims = [(i, cosine(q_emb, drama_embeddings[i]))
                for i in range(len(drama_embeddings)) if i != q_idx]
        sims.sort(key=lambda x: x[1], reverse=True)

        top_idx = [i for i, _ in sims[:top_k]]
        top_sim = [s for _, s in sims[:top_k]]

        res = drama_df.iloc[top_idx].copy()
        res["similarity"] = top_sim

        return {
            "source": "csv",
            "description": drama_df.iloc[q_idx]["description"],
            "poster": None,  # CSV에는 없음
            "recommendations": res[["title", "keywords", "similarity"]].to_dict(orient="records")
        }

    # ===================
    # 2) CSV에 없음 → 네이버 검색
    # ===================
    online = get_drama_online_info(title)
    desc = online["description"]
    poster = online["poster"]

    if not desc:
        return {"message": f"'{title}' 줄거리도, CSV 데이터도 없음."}

    # SBERT 임베딩 생성
    query_text = f"제목: {title} / 설명: {desc}"
    query_emb_text = drama_model.encode([query_text])[0]
    dummy_numeric = np.zeros(len(DRAMA_NUMERIC))
    q_emb = np.concatenate([query_emb_text, dummy_numeric], axis=0)

    # 유사도 계산
    sims = [(i, cosine(q_emb, drama_embeddings[i])) 
            for i in range(len(drama_embeddings))]
    sims.sort(key=lambda x: x[1], reverse=True)

    top_idx = [i for i, _ in sims[:top_k]]
    top_sim = [s for _, s in sims[:top_k]]

    res = drama_df.iloc[top_idx].copy()
    res["similarity"] = top_sim

    return {
        "source": "online",
        "description": desc,
        "poster": poster,
        "recommendations": res[["title", "keywords", "similarity"]].to_dict(orient="records")
    }



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

    # 1) CSV에서 exact match 먼저 시도
    idx_list = movie_df.index[movie_df["title"] == title].tolist()

    if idx_list:
        q_idx = idx_list[0]
        q_emb = movie_embeddings[q_idx]

        # 유사도 계산
        sims = [
            (i, cosine(q_emb, movie_embeddings[i]))
            for i in range(len(movie_embeddings))
            if i != q_idx
        ]
        sims.sort(key=lambda x: x[1], reverse=True)

        top_idx = [i for i, _ in sims[:top_k]]
        top_sim = [s for _, s in sims[:top_k]]

        res = movie_df.iloc[top_idx].copy()
        res["similarity"] = top_sim

        # CSV에는 포스터/줄거리 없음 → 네이버 검색으로 보충
        online = get_movie_online_info(title)
        desc = online["description"]
        poster = online["poster"]

        return {
            "source": "csv",
            "description": desc,
            "poster": poster,
            "recommendations": res[
                ["title", "cleaned_genre", "keywords", "cliche_score", "similarity"]
            ].to_dict(orient="records")
        }

    # 2) CSV에 없는 영화 → 네이버 검색 fallback
    online = get_movie_online_info(title)
    desc = online["description"]
    poster = online["poster"]

    if not desc:
        return {"message": f"'{title}' 줄거리도, CSV 데이터도 없음."}

    # SBERT query embedding 구성
    query_text = f"제목: {title} / 설명: {desc}"
    query_emb_text = movie_model.encode([query_text])[0]

    # numeric은 0으로
    dummy_numeric = np.zeros(len(MOVIE_NUMERIC))
    q_emb = np.concatenate([query_emb_text, dummy_numeric], axis=0)

    # 전체 영화와 유사도 계산
    sims = [
        (i, cosine(q_emb, movie_embeddings[i]))
        for i in range(len(movie_embeddings))
    ]
    sims.sort(key=lambda x: x[1], reverse=True)

    top_idx = [i for i, _ in sims[:top_k]]
    top_sim = [s for _, s in sims[:top_k]]

    res = movie_df.iloc[top_idx].copy()
    res["similarity"] = top_sim

    return {
        "source": "online",
        "description": desc,
        "poster": poster,
        "recommendations": res[
            ["title", "cleaned_genre", "keywords", "cliche_score", "similarity"]
        ].to_dict(orient="records")
    }


# ============================================
# 7️⃣ Play Recommender (build_recommendations.csv 기반)
# ============================================
# ============================================
# 7️⃣ Play Recommender (SBERT + Numeric 동일 구조)
# ============================================

# ============================================
# 7️⃣ Play Recommender (6개 컬럼 기반)
# ============================================

PLAY_CSV = "../data/play_cliche.csv"

# CSV 컬럼 정의 (사용자가 제공한 형태)
PLAY_NUMERIC = ["장수", "repetition_ratio", "기본점수", "cliche_score"]
PLAY_STRING = ["title", "keywords"]

play_df = pd.read_csv(PLAY_CSV, encoding="utf-8-sig")

# -----------------------------
#  문자열 컬럼 전처리
# -----------------------------
play_df["title"] = play_df["title"].fillna("").astype(str)
play_df["keywords"] = play_df["keywords"].fillna("").astype(str)

# -----------------------------
#  숫자 컬럼 전처리
# -----------------------------
play_df[PLAY_NUMERIC] = play_df[PLAY_NUMERIC].apply(
    pd.to_numeric, errors="coerce"
).fillna(0)

print("[INFO] Play SBERT 모델 로딩...")
play_model = SentenceTransformer("jhgan/ko-sbert-multitask")

# -----------------------------
#  SBERT 입력 텍스트 구성
# -----------------------------
play_texts = [
    f"제목: {r['title']} / 키워드: {r['keywords']}"
    for _, r in play_df.iterrows()
]

play_text_emb = play_model.encode(play_texts, show_progress_bar=True)

# -----------------------------
#  숫자 특징 스케일링
# -----------------------------
play_scaler = StandardScaler()
play_numeric_scaled = play_scaler.fit_transform(play_df[PLAY_NUMERIC].values)

# -----------------------------
#  최종 임베딩 결합
# -----------------------------
play_embeddings = np.concatenate([play_text_emb, play_numeric_scaled], axis=1)

print("[INFO] Play 임베딩 준비 완료")


@app.get("/play_recommend")
def play_recommend(title: str = Query(...), top_k: int = 10):
    """
    title = 희곡 제목
    top_k = 추천 개수
    """

    # 1) 제목으로 작품 검색
    idx_list = play_df.index[play_df["title"] == title].tolist()
    if not idx_list:
        return {"message": f"'{title}' 희곡을 찾을 수 없습니다."}

    q_idx = idx_list[0]
    q_emb = play_embeddings[q_idx]

    # 2) cosine similarity 계산
    sims = [
        (i, cosine(q_emb, play_embeddings[i]))
        for i in range(len(play_embeddings))
        if i != q_idx
    ]

    sims.sort(key=lambda x: x[1], reverse=True)

    # 3) 상위 K개 선택
    top_idx = [i for i, _ in sims[:top_k]]
    top_sim = [s for _, s in sims[:top_k]]

    res = play_df.iloc[top_idx].copy()
    res["similarity"] = top_sim

    # 4) 필요한 컬럼만 반환
    return res[
        ["title", "keywords", "기본점수", "cliche_score", "similarity"]
    ].to_dict(orient="records")
