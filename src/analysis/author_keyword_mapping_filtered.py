import pickle
import re
import nltk
from tqdm import tqdm
from collections import defaultdict, Counter

# NLTK 불용어 목록 다운로드 및 로딩
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# 1. 키워드 추출 함수 정의 (등장 순서 기준 상위 10개)
def extract_keywords(text, top_n=10):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # 특수문자 제거 및 소문자화
    tokens = text.split()  # 공백 기준 토큰화
    filtered = [w for w in tokens if w not in stop_words and len(w) > 2]  # 불용어 및 짧은 단어 제거

    seen = set()
    keywords = []
    for word in filtered:
        if word not in seen:
            seen.add(word)
            keywords.append(word)
        if len(keywords) == top_n:
            break
    return keywords

# 2. 데이터 불러오기 (2023년 논문만 필터링)
with open('arxiv_2021_2023.pkl', 'rb') as f:
    full_data = pickle.load(f)

data = [
    paper for paper in full_data
    if 'update_date' in paper and paper['update_date'].startswith('2023')
]

print("2023년 논문 수:", len(data))

# 3. 저자 → 키워드 매핑 딕셔너리 초기화
author_keywords = defaultdict(list)

# 4. 필터링할 국가/기관 키워드 목록 (소문자 기준)
filtered_names = {
    'usa', 'germany', 'france', 'italy', 'japan', 'uk', 'china', 'spain', 'canada', 'india',
    'korea', 'australia', 'switzerland', 'netherlands', 'united states', 'europe', 'russia',
    'institution', 'university', 'center', 'institute', 'team', 'group', 'department',
    'cnrs', 'csic', 'esa'
}

# 5. 논문별 키워드 추출 및 저자 연결
print("저자-키워드 매핑 생성 중...")
for paper in tqdm(data, desc="Processing papers"):
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    authors_raw = paper.get('authors', '')
    authors = [a.strip() for a in authors_raw.split(',') if a.strip()]
    text = f"{title} {abstract}"
    keywords = extract_keywords(text, top_n=10)

    for author in authors:
        author_lower = author.lower()

        # 필터링 조건 설정
        if (
            any(badword in author_lower for badword in filtered_names) or
            re.fullmatch(r'\d+\)?', author.strip()) or
            len(author.split()) < 1 or len(author.strip()) < 3
        ):
            continue

        author_keywords[author].extend(keywords)

# 6. 저자별 키워드 빈도 계산
author_keyword_freq = {
    author: Counter(keywords)
    for author, keywords in author_keywords.items()
}

# 7. 상위 저자 10명과 자주 사용하는 키워드 출력
print("\n상위 저자 10명과 자주 사용하는 키워드:")
for author, kw_freq in sorted(author_keyword_freq.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]:
    top_keywords = kw_freq.most_common(5)
    print(f"\n저자: {author}")
    for word, count in top_keywords:
        print(f"  {word} ({count}회)")