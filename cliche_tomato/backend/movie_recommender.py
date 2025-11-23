# movie_recommender.py
import os
import sys
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


# ==========================
# 설정
# ==========================
CSV_PATH = "movie_clean.csv"  # 네 영화 CSV 파일 이름 (원하는 걸로 바꿔줘)
MODEL_NAME = "jhgan/ko-sbert-multitask"  # Ko-SBERT
NUMERIC_COLS = ["semantic_density", "emotion_score", "entropy", "cliche_score"]


def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


class MovieRecommender:
    def __init__(self, csv_path: str, model_name: str = MODEL_NAME):
        self.csv_path = csv_path
        self.model_name = model_name

        self.df = None
        self.model = None
        self.scaler = None
        self.embeddings = None

    # -----------------------------
    def load_data(self):
        """CSV 로드 및 기본 전처리"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path, encoding="utf-8-sig")

        required_cols = [
            "title", "keywords", "cleaned_genre",
            "semantic_density", "emotion_score", "entropy", "cliche_score"
        ]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"필수 컬럼이 없습니다: {col}")

        # 텍스트 컬럼
        self.df["title"] = self.df["title"].fillna("").astype(str)
        self.df["keywords"] = self.df["keywords"].fillna("").astype(str)
        self.df["cleaned_genre"] = self.df["cleaned_genre"].fillna("").astype(str)

        # 숫자 컬럼
        for col in NUMERIC_COLS:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df[NUMERIC_COLS] = self.df[NUMERIC_COLS].fillna(0.0)

    # -----------------------------
    def build_model(self):
        """SBERT 모델 및 숫자 feature 스케일러 준비"""
        print(f"[INFO] SBERT 모델 로딩 중... ({self.model_name})")
        self.model = SentenceTransformer(self.model_name)

        print("[INFO] 숫자 feature 스케일링 학습 중...")
        numeric_mat = self.df[NUMERIC_COLS].values
        self.scaler = StandardScaler()
        self.scaler.fit(numeric_mat)

    # -----------------------------
    def build_embeddings(self):
        """각 영화의 최종 임베딩(text + numeric) 생성"""
        print("[INFO] 텍스트 임베딩 생성 중...")

        texts = []
        for _, row in self.df.iterrows():
            t = (
                f"제목: {row['title']} / "
                f"장르: {row['cleaned_genre']} / "
                f"키워드: {row['keywords']}"
            )
            texts.append(t)

        text_emb = self.model.encode(texts, show_progress_bar=True)

        print("[INFO] 숫자 feature 임베딩 결합 중...")
        numeric_scaled = self.scaler.transform(self.df[NUMERIC_COLS].values)

        # 최종 임베딩 = [텍스트 임베딩, 숫자 feature]
        self.embeddings = np.concatenate([text_emb, numeric_scaled], axis=1)
        print("[INFO] 임베딩 shape:", self.embeddings.shape)

    # -----------------------------
    def fit(self):
        """전체 파이프라인 실행"""
        print("[STEP] 데이터 로드")
        self.load_data()
        print("[STEP] 모델 & 스케일러 준비")
        self.build_model()
        print("[STEP] 임베딩 생성")
        self.build_embeddings()
        print("[DONE] 영화 추천기 준비 완료")

    # -----------------------------
    def recommend_by_title(self, title: str, top_k: int = 10):
        """영화 제목으로 유사한 영화 top_k 개 추천 (자기 자신 제외)"""
        if self.embeddings is None:
            raise RuntimeError("먼저 fit()을 호출하여 임베딩을 생성하세요.")

        # 정확히 같은 제목 찾기
        idx_list = self.df.index[self.df["title"] == title].tolist()
        if not idx_list:
            print(f"[WARN] '{title}' 제목의 영화를 찾을 수 없습니다.")
            # 부분 검색으로 후보 제안
            candidates = self.df[self.df["title"].str.contains(title, na=False)]
            if not candidates.empty:
                print("[INFO] 비슷한 제목 후보:")
                for t in candidates["title"].unique()[:10]:
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
        top_k = min(top_k, len(sims))
        top_idx = [idx for idx, _ in sims[:top_k]]
        top_sim = [sim for _, sim in sims[:top_k]]

        result = self.df.iloc[top_idx].copy()
        result["similarity"] = top_sim

        # 보고 싶을만한 컬럼만 정리
        return result[[
            "title",
            "cleaned_genre",
            "cliche_score",
            "semantic_density",
            "emotion_score",
            "entropy",
            "similarity"
        ]]


# ==========================
# CLI 실행 예시
# ==========================
def main():
    csv_path = CSV_PATH
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    print(f"[INFO] 사용 CSV: {csv_path}")

    recommender = MovieRecommender(csv_path)
    recommender.fit()

    while True:
        print("\n------------------------------------")
        query = input("유사한 영화를 보고 싶은 제목 입력 (종료: q): ").strip()
        if query.lower() == "q":
            print("종료합니다.")
            break

        res = recommender.recommend_by_title(query, top_k=10)
        if res is None:
            continue

        print(f"\n[ '{query}' 와 유사한 영화 TOP 10 ]")
        for _, row in res.iterrows():
            print(
                f"- {row['title']} "
                f"| 장르={row['cleaned_genre']} "
                f"| cliche={row['cliche_score']:.2f} "
                f"| sim={row['similarity']:.4f}"
            )


if __name__ == "__main__":
    main()
