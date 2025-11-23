# drama_recommender.py
import os
import sys
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


CSV_PATH = "drama_cliche.csv"
MODEL_NAME = "jhgan/ko-sbert-multitask"
NUMERIC_COLS = ["semantic_density", "emotion_score", "entropy", "cliche_score"]


def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


class DramaRecommender:
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

        required_cols = [
            "title", "description", "keywords",
            "semantic_density", "emotion_score", "entropy", "cliche_score"
        ]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"필수 컬럼 없음: {col}")

        self.df["title"] = self.df["title"].fillna("").astype(str)
        self.df["description"] = self.df["description"].fillna("").astype(str)
        self.df["keywords"] = self.df["keywords"].fillna("").astype(str)

        for col in NUMERIC_COLS:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df[NUMERIC_COLS] = self.df[NUMERIC_COLS].fillna(0.0)

    def build_model(self):
        print(f"[INFO] SBERT 로딩 중... ({self.model_name})")
        self.model = SentenceTransformer(self.model_name)

        print("[INFO] 숫자 feature 스케일러 준비...")
        numeric_mat = self.df[NUMERIC_COLS].values
        self.scaler = StandardScaler()
        self.scaler.fit(numeric_mat)

    def build_embeddings(self):
        print("[INFO] 드라마 텍스트 임베딩 생성 중...")

        texts = []
        for _, row in self.df.iterrows():
            t = (
                f"제목: {row['title']} "
                f"/ 설명: {row['description']} "
                f"/ 키워드: {row['keywords']}"
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
        print("[DONE] 드라마 추천 모델 준비완료!")

    def recommend_by_title(self, title: str, top_k: int = 10):
        if self.embeddings is None:
            raise RuntimeError("fit()을 먼저 실행하세요.")

        idx_list = self.df.index[self.df["title"] == title].tolist()
        if not idx_list:
            print(f"[WARN] 제목 '{title}' 의 드라마 없음.")
            candidates = self.df[self.df["title"].str.contains(title)]
            if not candidates.empty:
                print("[INFO] 비슷한 제목 검색:")
                for t in candidates["title"].head(10):
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

        # 🔥 title + similarity 만 반환
        return res[["title", "similarity"]]


def main():
    csv_path = CSV_PATH
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    print(f"[INFO] 사용 CSV 파일: {csv_path}")

    R = DramaRecommender(csv_path)
    R.fit()

    while True:
        query = input("\n검색할 드라마 제목 (종료: q): ").strip()
        if query.lower() == "q":
            print("종료합니다.")
            break

        res = R.recommend_by_title(query, top_k=10)
        if res is None:
            continue

        print(f"\n▶ '{query}' 와 유사한 드라마 TOP 10")
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
