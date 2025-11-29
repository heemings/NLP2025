import os
import csv
import re
import joblib
import torch
import numpy as np
import pandas as pd
import requests
import urllib.parse
from bs4 import BeautifulSoup
import sys
import __main__ 
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# NLP 라이브러리
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from kiwipiepy import Kiwi
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

# ============================================
# 1️⃣ Home (경로 수정 완료)
# ============================================
@app.get("/", response_class=HTMLResponse)
def home():
    # 요청하신 대로 frontend 폴더 바로 아래 play.html을 바라보게 수정했습니다.
    return FileResponse("../frontend/templates/play.html")

# --- 크롤링 함수들 ---
def get_movie_online_info(movie_name: str):
    try:
        query = urllib.parse.quote(movie_name)
        url = f"https://search.naver.com/search.naver?query={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(res.text, "html.parser")
        
        desc_tag = soup.select_one("div.cm_info_box span.desc._text") or soup.select_one("span.desc._text")
        description = desc_tag.get_text(strip=True) if desc_tag else None
        
        poster_url = None
        for sel in ["div.cm_info_box a.thumb img", "a.thumb img", "div.thumb img", "img._img"]:
            tag = soup.select_one(sel)
            if tag and tag.get("src"):
                poster_url = tag["src"]
                break
        return {"description": description, "poster": poster_url}
    except:
        return {"description": None, "poster": None}

def get_drama_online_info(drama_name: str):
    # 드라마도 영화와 로직 동일하게 처리
    return get_movie_online_info(drama_name)


# ============================================
# 2️⃣ ~ 7️⃣ 검색 및 추천 API (기존 유지)
# ============================================
@app.get("/drama")
def get_drama():
    data = []
    if os.path.exists("../data/drama_cliche.csv"):
        with open("../data/drama_cliche.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                data.append({"title": row["title"], "keywords": row["keywords"], "cliche_score": float(row["cliche_score"])})
    return sorted(data, key=lambda x: x["cliche_score"], reverse=True)

@app.get("/movie")
def get_movie():
    data = []
    if os.path.exists("../data/movie_cliche.csv"):
        with open("../data/movie_cliche.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                data.append({"title": row["title"], "keywords": row["keywords"], "cliche_score": float(row["cliche_score"])})
    return sorted(data, key=lambda x: x["cliche_score"], reverse=True)

@app.get("/play")
def get_play():
    data = []
    if os.path.exists("../data/play_cliche.csv"):
        with open("../data/play_cliche.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                data.append({"title": row["title"], "keywords": row["keywords"], "cliche_score": float(row["cliche_score"])})
    return sorted(data, key=lambda x: x["cliche_score"], reverse=True)

# 추천 모델 로딩 (SBERT)
sbert_model = SentenceTransformer("jhgan/ko-sbert-multitask")

# [Drama Recommender]
drama_df = pd.DataFrame()
drama_embeddings = None
if os.path.exists("../data/drama_cliche.csv"):
    drama_df = pd.read_csv("../data/drama_cliche.csv", encoding="utf-8-sig").fillna("")
    # 숫자 변환
    cols = ["semantic_density", "emotion_score", "entropy", "cliche_score"]
    drama_df[cols] = drama_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    # 임베딩
    texts = [f"제목: {r['title']} / 설명: {r['description']} / 키워드: {r['keywords']}" for _, r in drama_df.iterrows()]
    emb_txt = sbert_model.encode(texts, show_progress_bar=False)
    emb_num = StandardScaler().fit_transform(drama_df[cols].values)
    drama_embeddings = np.concatenate([emb_txt, emb_num], axis=1)

@app.get("/drama_recommend")
def drama_recommend(title: str = Query(...), top_k: int = 10):
    if drama_df.empty: return {"message": "데이터 없음"}
    
    # CSV 검색
    idx_list = drama_df.index[drama_df["title"] == title].tolist()
    if idx_list:
        q_idx = idx_list[0]
        sims = [(i, cosine(drama_embeddings[q_idx], drama_embeddings[i])) for i in range(len(drama_df)) if i != q_idx]
        sims.sort(key=lambda x: x[1], reverse=True)
        res = drama_df.iloc[[i for i, _ in sims[:top_k]]].copy()
        res["similarity"] = [s for _, s in sims[:top_k]]
        return {"source": "csv", "description": drama_df.iloc[q_idx]["description"], "poster": None, "recommendations": res[["title", "keywords", "similarity"]].to_dict(orient="records")}
    
    # 온라인 검색
    online = get_drama_online_info(title)
    if not online["description"]: return {"message": "정보 없음"}
    
    q_vec_txt = sbert_model.encode([f"제목: {title} / 설명: {online['description']}"])[0]
    q_vec = np.concatenate([q_vec_txt, np.zeros(4)]) # 숫자형은 0 처리
    
    sims = [(i, cosine(q_vec, drama_embeddings[i])) for i in range(len(drama_df))]
    sims.sort(key=lambda x: x[1], reverse=True)
    res = drama_df.iloc[[i for i, _ in sims[:top_k]]].copy()
    res["similarity"] = [s for _, s in sims[:top_k]]
    return {"source": "online", "description": online["description"], "poster": online["poster"], "recommendations": res[["title", "keywords", "similarity"]].to_dict(orient="records")}


# [Movie Recommender]
movie_df = pd.DataFrame()
movie_embeddings = None
if os.path.exists("../data/movie_cliche.csv"):
    movie_df = pd.read_csv("../data/movie_cliche.csv", encoding="utf-8-sig").fillna("")
    cols = ["semantic_density", "emotion_score", "entropy", "cliche_score"]
    movie_df[cols] = movie_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    texts = [f"제목: {r['title']} / 장르: {r['cleaned_genre']} / 키워드: {r['keywords']}" for _, r in movie_df.iterrows()]
    emb_txt = sbert_model.encode(texts, show_progress_bar=False)
    emb_num = StandardScaler().fit_transform(movie_df[cols].values)
    movie_embeddings = np.concatenate([emb_txt, emb_num], axis=1)

@app.get("/movie_recommend")
def movie_recommend(title: str = Query(...), top_k: int = 10):
    if movie_df.empty: return {"message": "데이터 없음"}
    idx_list = movie_df.index[movie_df["title"] == title].tolist()
    
    if idx_list:
        q_idx = idx_list[0]
        sims = [(i, cosine(movie_embeddings[q_idx], movie_embeddings[i])) for i in range(len(movie_df)) if i != q_idx]
        sims.sort(key=lambda x: x[1], reverse=True)
        res = movie_df.iloc[[i for i, _ in sims[:top_k]]].copy()
        res["similarity"] = [s for _, s in sims[:top_k]]
        online = get_movie_online_info(title)
        return {"source": "csv", "description": online["description"], "poster": online["poster"], "recommendations": res[["title", "cleaned_genre", "keywords", "cliche_score", "similarity"]].to_dict(orient="records")}

    online = get_movie_online_info(title)
    if not online["description"]: return {"message": "정보 없음"}
    
    q_vec_txt = sbert_model.encode([f"제목: {title} / 설명: {online['description']}"])[0]
    q_vec = np.concatenate([q_vec_txt, np.zeros(4)])
    
    sims = [(i, cosine(q_vec, movie_embeddings[i])) for i in range(len(movie_df))]
    sims.sort(key=lambda x: x[1], reverse=True)
    res = movie_df.iloc[[i for i, _ in sims[:top_k]]].copy()
    res["similarity"] = [s for _, s in sims[:top_k]]
    return {"source": "online", "description": online["description"], "poster": online["poster"], "recommendations": res[["title", "cleaned_genre", "keywords", "cliche_score", "similarity"]].to_dict(orient="records")}


# [Play Recommender]
play_df = pd.DataFrame()
play_embeddings = None
if os.path.exists("../data/play_cliche.csv"):
    play_df = pd.read_csv("../data/play_cliche.csv", encoding="utf-8-sig").fillna("")
    cols = ["장수", "repetition_ratio", "기본점수", "cliche_score"]
    play_df[cols] = play_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    texts = [f"제목: {r['title']} / 키워드: {r['keywords']}" for _, r in play_df.iterrows()]
    emb_txt = sbert_model.encode(texts, show_progress_bar=False)
    emb_num = StandardScaler().fit_transform(play_df[cols].values)
    play_embeddings = np.concatenate([emb_txt, emb_num], axis=1)

@app.get("/play_recommend")
def play_recommend(title: str = Query(...), top_k: int = 10):
    if play_df.empty: return {"message": "데이터 없음"}
    idx_list = play_df.index[play_df["title"] == title].tolist()
    if not idx_list: return {"message": "정보 없음"}
    
    q_idx = idx_list[0]
    sims = [(i, cosine(play_embeddings[q_idx], play_embeddings[i])) for i in range(len(play_df)) if i != q_idx]
    sims.sort(key=lambda x: x[1], reverse=True)
    res = play_df.iloc[[i for i, _ in sims[:top_k]]].copy()
    res["similarity"] = [s for _, s in sims[:top_k]]
    return res[["title", "keywords", "기본점수", "cliche_score", "similarity"]].to_dict(orient="records")


# ============================================
# 8️⃣ Script Analysis API (핵심 수정 적용)
# ============================================
MODEL_DIR = "../model"

class ScriptInput(BaseModel):
    text: str

kiwi = Kiwi()
KEEP_TAGS = {"NNG", "NNP", "VV", "VA", "MAG", "XR"}

def norm_text(t: str) -> str:
    if not t: return ""
    t = str(t).replace("\x00", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def kiwi_tokenize(text: str):
    text = norm_text(text)
    toks = []
    for tk in kiwi.tokenize(text):
        if tk.tag in KEEP_TAGS:
            toks.append(tk.form)
    return toks

# KeyBERT에서 사용할 벡터라이저
keybert_vectorizer = TfidfVectorizer(
    tokenizer=kiwi_tokenize,
    ngram_range=(1, 3),
    min_df=1,
    max_df=1.0
)

class KoBERTEncoder:
    def __init__(self, model_name: str = "skt/kobert-base-v1", device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 16, max_length: int = 256):
        if isinstance(texts, str): texts = [texts]
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            sent_emb = (out.last_hidden_state * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
            embs.append(sent_emb.detach().cpu())
        return torch.cat(embs, dim=0).numpy()

# --- 모델 로딩 ---
print("[INFO] Loading Script Analysis Models...")
setattr(__main__, 'kiwi_tokenize', kiwi_tokenize) # 노트북 함수 인식

tfidf, ohe, reg, kw_model = None, None, None, None

try:
    if os.path.exists(os.path.join(MODEL_DIR, "tfidf_kiwi_meta.joblib")):
        # 1. 껍데기(모델) 로드
        loaded_tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_kiwi_meta.joblib"))
        
        # 2. ★ 중요: 새 분석기에 알맹이(단어장)만 이식 (점수 고정 해결책) ★
        tfidf = TfidfVectorizer(
            tokenizer=kiwi_tokenize, # 여기서 함수 직접 연결
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=200_000,
        )
        tfidf.vocabulary_ = loaded_tfidf.vocabulary_
        tfidf.idf_ = loaded_tfidf.idf_
        
        ohe = joblib.load(os.path.join(MODEL_DIR, "genre_ohe.joblib"))
        reg = joblib.load(os.path.join(MODEL_DIR, "regressor_elasticnet_meta.joblib"))
        
        kobert_encoder = KoBERTEncoder("skt/kobert-base-v1")
        kw_model = KeyBERT(model=kobert_encoder)
        print("[INFO] Analysis Models Loaded Successfully!")
    else:
        print("[WARN] Model files not found in ../model")
except Exception as e:
    print(f"[ERROR] Loading Models: {e}")

META_KW_MEAN = 0.5 
META_KW_DIV  = 0.5
GENRE_MODE   = 0

@app.post("/api/predict")
async def predict_script(input_data: ScriptInput):
    if not tfidf or not kw_model:
        return {"error": "모델이 준비되지 않았습니다."}

    text = input_data.text.strip()
    if not text:
        return {"error": "내용을 입력해주세요."}

    t_norm = norm_text(text)

    try:
        # TF-IDF 변환
        X_txt = tfidf.transform([t_norm])
        
        # 점수 예측
        meta_c = csr_matrix([[META_KW_MEAN, META_KW_DIV]])
        g = ohe.transform([[GENRE_MODE]])
        X_full = hstack([X_txt, meta_c, g]).tocsr()
        score_raw = float(np.clip(reg.predict(X_full)[0], 0, 1))
        
        # 키워드 추출
        keywords_raw = kw_model.extract_keywords(
            t_norm,
            vectorizer=keybert_vectorizer,
            keyphrase_ngram_range=(1, 3),
            use_mmr=True,
            diversity=0.5,
            top_n=5
        )
        keywords = [w for w, score in keywords_raw]

        return {
            "cliche_score": round(score_raw * 100.0, 2),
            "keywords": keywords,
            "message": "완료"
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"error": "분석 실패"}