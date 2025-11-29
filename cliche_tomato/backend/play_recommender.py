# play_recommender.py
import os
import sys
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


CSV_PATH = "play_cliche.csv"
MODEL_NAME = "jhgan/ko-sbert-multitask"

# 숫자 특징 (희곡용)
NUMERIC_COLS = ["장수", "repetition_ratio", "기본점수", "cliche_score"]


def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


class PlayRecommender:
    def __init__(self, csv_path: str, model_name: str = MODEL_NAME):
        self.csv_path = csv_path
        self.model_name = model_name

        self.df = None
        self.model = None
        self.scaler = None
        self.embeddings = None

    def load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV 파일 없음: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path, encoding="utf-8-sig")

        # drama_recommender 구조 그대로: 필수 컬럼 검사
        required_cols = [
            "제목", "장르클러스터", "나온모든키워드"
        ] + NUMERIC_COLS

        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"필수 컬럼 없음: {col}")

        # 문자열 컬럼 전처리
        self.df["제목"] = self.df["제목"].fillna("").astype(str)
        self.df["장르클러스터"] = self.df["장르클러스터"].fillna("").astype(str)
        self.df["나온모든키워드"] = self.df["나온모든키워드"].fillna("").astype(str)

        # 숫자형 전처리
        for col in NUMERIC_COLS:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)

    def build_model(self):
        print(f"[INFO] SBERT 로딩 중... ({self.model_name})")
        self.model = SentenceTransformer(self.model_name)

        numeric_mat = self.df[NUMERIC_COLS].values
        self.scaler = StandardScaler()
        self.scaler.fit(numeric_mat)

    def build_embeddings(self):
        print("[INFO] 희곡 텍스트 임베딩 생성 중...")

        texts = []
        for _, row in self.df.iterrows():
            t = (
                f"제목: {row['제목']} "
                f"/ 장르클러스터: {row['장르클러스터']} "
                f"/ 키워드: {row['나온모든키워드']}"
            )
            texts.append(t)

        text_emb = self.model.encode(texts, show_progress_bar=True)
        numeric_scaled = self.scaler.transform(self.df[NUMERIC_COLS].values)

        self.embeddings = np.concatenate([text_emb, numeric_scaled], axis=1)
        print("[INFO] 최종 임베딩 shape =", self.embeddings.shape)

    def fit(self):
        print("[STEP] CSV 로드")
        self.load_data()
        print("[STEP] 모델 준비")
        self.build_model()
        print("[STEP] 임베딩 구성")
        self.build_embeddings()
        print("[DONE] 희곡 추천 모델 준비 완료!")

    def recommend_by_title(self, title: str, top_k: int = 10):
        if self.embeddings is None:
            raise RuntimeError("fit()을 먼저 실행하세요.")

        idx_list = self.df.index[self.df["제목"] == title].tolist()
        if not idx_list:
            print(f"[WARN] 제목 '{title}' 의 희곡이 없음.")
            candidates = self.df[self.df["제목"].str.contains(title)]
            if not candidates.empty:
                print("[INFO] 비슷한 제목 검색:")
                for t in candidates["제목"].head(10):
                    print("  -", t)
            return None

        q_idx = idx_list[0]
        q_emb = self.embeddings[q_idx]

        sims = []
        for i in range(len(self.embeddings)):
            if i == q_idx:
                continue
            sim = cosine(q_emb, self.embeddings[i])
            sims.append((i, sim))

        sims.sort(key=lambda x: x[1], reverse=True)

        top_idx = [i for i, _ in sims[:top_k]]
        top_sim = [s for _, s in sims[:top_k]]

        res = self.df.iloc[top_idx].copy()
        res["similarity"] = top_sim

        # 드라마 버전과 동일: title + similarity만 출력
        return res[["제목", "similarity"]]


def main():
    csv_path = CSV_PATH
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    print(f"[INFO] 사용 CSV 파일: {csv_path}")

    R = PlayRecommender(csv_path)
    R.fit()

    while True:
        query = input("\n검색할 희곡 제목 (종료: q): ").strip()
        if query.lower() == "q":
            print("종료합니다.")
            break

        res = R.recommend_by_title(query, top_k=10)
        if res is None:
            continue

        print(f"\n▶ '{query}' 와 유사한 희곡 TOP 10")
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
