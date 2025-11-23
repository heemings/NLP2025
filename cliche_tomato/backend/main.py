from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import csv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse


app = FastAPI()

# --- CORS 허용 (HTML이 API를 호출할 수 있게) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("../frontend/templates/play.html")

@app.get("/drama")
def get_drama():
    drama = []
    with open("../data/drama_cliche.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drama.append({
                "title": row["title"],
                "keywords": row["keywords"],
                "cliche_score": float(row["cliche_score"])
            })

    # 점수 높은 순 정렬
    drama = sorted(drama, key=lambda x: x["cliche_score"], reverse=True)
    return drama

@app.get("/movie")
def get_movie():
    movie = []
    with open("../data/movie_cliche.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie.append({
                "title": row["title"],
                "keywords": row["keywords"],
                "cliche_score": float(row["cliche_score"])
            })

    # 점수 높은 순 정렬
    movie = sorted(movie, key=lambda x: x["cliche_score"], reverse=True)
    return movie

@app.get("/play")
def get_play():
    play = []
    with open("../data/play_cliche.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            play.append({
                "title": row["title"],
                "keywords": row["keywords"],
                "cliche_score": float(row["cliche_score"])
            })

    # 점수 높은 순 정렬
    play = sorted(play, key=lambda x: x["cliche_score"], reverse=True)
    return play