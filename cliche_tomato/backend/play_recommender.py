# build_recommendations.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


CSV_PATH = "play_clean.csv"      # 원본 데이터
OUTPUT_CSV = "recommendations_top10.csv"
MODEL_NAME = "jhgan/ko-sbert-multitask"


def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


def main():
    print("[1] CSV 로드 중...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    required = ["제목", "장수", "repetition_ratio", "기본점수",
                "최종점수", "장르클러스터", "나온모든키워드"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 없음: {c}")

    # 숫자형 컬럼 전처리
    numeric_cols = ["장수", "repetition_ratio", "기본점수", "최종점수"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # 빈값 전처리
    df["제목"] = df["제목"].fillna("").astype(str)
    df["나온모든키워드"] = df["나온모든키워드"].fillna("").astype(str)

    print("[2] SBERT 모델 로드...")
    model = SentenceTransformer(MODEL_NAME)

    print("[3] 문장 임베딩 생성...")
    texts = []
    for _, row in df.iterrows():
        txt = (
            f"제목: {row['제목']} "
            f"장르클러스터: {row['장르클러스터']} "
            f"키워드: {row['나온모든키워드']}"
        )
        texts.append(txt)

    text_emb = model.encode(texts, show_progress_bar=True)

    print("[4] 숫자 feature 스케일링...")
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(df[numeric_cols])

    print("[5] 텍스트 + 숫자 feature 결합...")
    embeddings = np.concatenate([text_emb, numeric_scaled], axis=1)
    print("임베딩 shape =", embeddings.shape)

    print("[6] 각 작품별 유사도 계산 및 상위 10개 선정...")

    results = []
    n = len(df)

    for i in range(n):
        title_i = df.iloc[i]["제목"]
        emb_i = embeddings[i]

        sims = []
        for j in range(n):
            if i == j:
                continue
            sim = cosine(emb_i, embeddings[j])
            sims.append((j, sim))

        # 유사도 높은 순 정렬
        sims.sort(key=lambda x: x[1], reverse=True)

        top10 = sims[:10]

        for j, sim in top10:
            results.append({
                "기준작품": title_i,
                "유사작품": df.iloc[j]["제목"],
                "유사도": sim
            })

    print("[7] CSV 저장:", OUTPUT_CSV)
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n=== 완료! ===")
    print("recommendations_top10.csv 파일이 생성되었습니다.")


if __name__ == "__main__":
    main()
