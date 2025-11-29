import os
import re
import numpy as np
import joblib
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from kiwipiepy import Kiwi
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import hstack, csr_matrix
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 1. 경로 및 설정 (수정됨)
# -----------------------------
# 사용자의 요청대로 BASE_DIR 변수 제거 및 상대 경로 적용
MODEL_DIR = "../model"       # backend 상위의 model 폴더
TEMPLATES_DIR = "../frontend" # backend 상위의 frontend 폴더

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Kiwi 토크나이저 설정
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

# KoBERT Encoder 클래스
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
        if isinstance(texts, str):
            texts = [texts]
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            last = out.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            sent_emb = (last * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
            embs.append(sent_emb.detach().cpu())
        return torch.cat(embs, dim=0).numpy()

# 모델 초기화
print("Loading Models... (This might take a while)")
try:
    # 경로가 ../model 로 변경됨
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_kiwi_meta.joblib"))
    ohe = joblib.load(os.path.join(MODEL_DIR, "genre_ohe.joblib"))
    reg = joblib.load(os.path.join(MODEL_DIR, "regressor_elasticnet_meta.joblib"))
    
    # KoBERT + KeyBERT 로드
    kobert_encoder = KoBERTEncoder("skt/kobert-base-v1")
    kw_model = KeyBERT(model=kobert_encoder)
    print("All Models Loaded Successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print(f"⚠️ '{MODEL_DIR}' 폴더에 .joblib 파일들이 있는지 확인해주세요!")

# ★중요★: 학습 노트북 결과에서 나온 값으로 수정 필요 (임시값 적용됨)
META_KW_MEAN = 0.5 
META_KW_DIV  = 0.5
GENRE_MODE   = 0

# -----------------------------
# 2. 데이터 구조 정의
# -----------------------------
class ScriptInput(BaseModel):
    text: str

# -----------------------------
# 3. API 엔드포인트
# -----------------------------

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("play.html", {"request": request})

@app.post("/api/predict")
async def predict_script(input_data: ScriptInput):
    text = input_data.text.strip()
    if not text:
        return {"error": "대본 내용을 입력해주세요."}

    t_norm = norm_text(text)

    # 1. 클리셰 점수 예측 (학생 모델)
    try:
        X_txt = tfidf.transform([t_norm])
        meta_c = csr_matrix([[META_KW_MEAN, META_KW_DIV]])
        g = ohe.transform([[GENRE_MODE]])
        
        X_full = hstack([X_txt, meta_c, g]).tocsr()
        score_raw = float(np.clip(reg.predict(X_full)[0], 0, 1))
        score_percent = round(score_raw * 100.0, 2)
    except Exception as e:
        return {"error": f"예측 중 오류 발생: {str(e)}"}

    # 2. 키워드 추출 (KeyBERT)
    try:
        keywords_raw = kw_model.extract_keywords(
            t_norm,
            keyphrase_ngram_range=(1, 3),
            use_mmr=True,
            diversity=0.5,
            top_n=5
        )
        keywords = [w for w, score in keywords_raw]
    except Exception as e:
        keywords = ["키워드 추출 실패"]

    return {
        "cliche_score": score_percent,
        "keywords": keywords,
        "message": "분석 완료"
    }