# import requests
# from bs4 import BeautifulSoup
# import urllib.parse
# import time

# # ---------------------
# # 1. 드라마 줄거리 수집 함수
# # ---------------------
# def get_drama_description(drama_name):
#     query = urllib.parse.quote(drama_name)
#     url = f"https://search.naver.com/search.naver?query={query}"

#     headers = {
#         "User-Agent": "Mozilla/5.0"
#     }

#     try:
#         res = requests.get(url, headers=headers, timeout=5)
#         soup = BeautifulSoup(res.text, "html.parser")

#         desc_tag = soup.find("span", class_="desc _text")

#         if desc_tag:
#             return desc_tag.get_text(strip=True)
#         else:
#             return None
#     except:
#         return None


# # ---------------------
# # 2. 사용자 입력 반복
# # ---------------------
# print("\n=== 네이버 드라마 줄거리 검색기 ===")
# print("드라마 이름을 입력하세요. 종료하려면 'exit' 입력.")

# while True:
#     title = input("\n검색할 드라마 제목: ")

#     if title.lower() == "exit":
#         print("프로그램 종료.")
#         break

#     print(f"\n▶ '{title}' 줄거리 검색 중...\n")

#     desc = get_drama_description(title)

#     if desc:
#         print("=== 줄거리 ===")
#         print(desc)
#     else:
#         print("⚠ 줄거리를 찾을 수 없습니다. 제목을 다시 확인해보세요.")

#     time.sleep(0.5)


import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import pandas as pd

# ---------------------
# 1. 드라마 줄거리 수집 함수
# ---------------------
def get_drama_description(drama_name):
    query = urllib.parse.quote(drama_name)
    url = f"https://search.naver.com/search.naver?query={query}"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")

        desc_tag = soup.find("span", class_="desc _text")

        if desc_tag:
            return desc_tag.get_text(strip=True)
        else:
            return None
    except:
        return None


# ---------------------
# 2. CSV 파일 로드
# ---------------------
csv_path = "/Users/heemings/Desktop/school/3-2/자연어처리/cliche_tomato/data/movie_cliche.csv"
df = pd.read_csv(csv_path)

titles = df["title"].dropna().tolist()   # NaN 제거

print(f"\n총 {len(titles)}개 제목 검색 시작...\n")

# ---------------------
# 3. 자동 검색 후 결과 저장
# ---------------------
results = []

for title in titles:
    print(f"▶ '{title}' 줄거리 검색 중...")

    desc = get_drama_description(title)

    results.append({
        "title": title,
        "description": desc if desc else "NOT FOUND"
    })

    time.sleep(0.5)

# ---------------------
# 4. CSV로 저장
# ---------------------
out_path = "/Users/heemings/Desktop/school/3-2/자연어처리/cliche_tomato/data/movie_cliche_before.csv"
pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")

print("\n=== 완료! ===")
print(f"결과 저장: {out_path}")
