import pickle
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from tqdm import tqdm

# 1. NLTK 불용어 로딩
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 2. 키워드 추출 함수 정의
def extract_keywords(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # 특수문자 제거 + 소문자 변환
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
    return filtered

# 3. 데이터 로드
with open('arxiv_2021_2023.pkl', 'rb') as f:
    full_data = pickle.load(f)

# 4. 분석 대상 연도
years = ['2021', '2022', '2023']

# 5. 연도별 키워드 분석
for year in years:
    # 연도 필터링
    year_data = [
        paper for paper in full_data
        if 'update_date' in paper and paper['update_date'].startswith(year)
    ]
    print(f"\n[{year}] 논문 수: {len(year_data)}")

    # 키워드 수집
    keywords = []
    for paper in tqdm(year_data, desc=f"[{year}] 키워드 추출 중"):
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        text = f"{title} {abstract}"
        keywords.extend(extract_keywords(text))

    # 상위 15개 출력
    counter = Counter(keywords)
    print(f"\n[{year}] 중심 키워드 Top 15:")
    for word, count in counter.most_common(15):
        print(f"{word:20} | Count: {count}")