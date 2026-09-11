import pickle
import re
import nltk
from tqdm import tqdm
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# 1. 불용어 로딩
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 2. 키워드 추출 함수
def extract_keywords(text, top_n=10):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words and len(w) > 2]

    seen = set()
    keywords = []
    for word in filtered:
        if word not in seen:
            seen.add(word)
            keywords.append(word)
        if len(keywords) == top_n:
            break
    return keywords

# 3. 데이터 불러오기
with open('arxiv_2021_2023.pkl', 'rb') as f:
    full_data = pickle.load(f)

data = [
    paper for paper in full_data
    if 'update_date' in paper and paper['update_date'].startswith('2023')
]

# 4. 국가 및 기관명 필터링 목록 정의 (소문자 기준)
filtered_names = {
    'usa', 'germany', 'france', 'italy', 'japan', 'uk', 'china', 'spain', 'canada', 'india',
    'korea', 'australia', 'switzerland', 'netherlands', 'united states', 'europe', 'russia',
    'institution', 'university', 'center', 'institute', 'team', 'group', 'department',
    'cnrs', 'csic', 'esa'
}

# 5. 저자-키워드 매핑
author_keywords = defaultdict(list)
for paper in tqdm(data, desc="Processing papers"):
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    authors_raw = paper.get('authors', '')
    authors = [a.strip() for a in authors_raw.split(',') if a.strip()]
    text = f"{title} {abstract}"
    keywords = extract_keywords(text, top_n=10)

    for author in authors:
        author_lower = author.lower()
        if (
            any(word in author_lower for word in filtered_names) or
            re.fullmatch(r'\d+\)?', author.strip()) or
            len(author.strip()) < 3
        ):
            continue
        author_keywords[author].extend(keywords)

# 6. 저자별 키워드 빈도 계산
author_keyword_freq = {
    author: Counter(keywords)
    for author, keywords in author_keywords.items()
}

# 7. 상위 저자 30명 선택
top_authors = sorted(
    author_keyword_freq.items(),
    key=lambda x: sum(x[1].values()),
    reverse=True
)[:30]

author_names = []
keyword_docs = []
for author, kw_counter in top_authors:
    words = []
    for word, count in kw_counter.items():
        words.extend([word] * count)
    author_names.append(author)
    keyword_docs.append(" ".join(words))

# 8. 벡터화
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(keyword_docs)

# 9. KMeans 클러스터링
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X)

# 10. 결과 저장
df_result = pd.DataFrame({
    'Author': author_names,
    'Cluster': labels,
    'Doc': keyword_docs
})

# 11. 클러스터별 대표 키워드 출력
print("\n클러스터별 대표 키워드:")
for i in range(5):
    docs_in_cluster = df_result[df_result['Cluster'] == i]['Doc']
    all_words = " ".join(docs_in_cluster).split()
    common = Counter(all_words).most_common(5)
    print(f"\nCluster {i}:")
    for word, count in common:
        print(f"  {word} ({count}회)")

# 12. 시각화를 위한 차원 축소 (PCA)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

# 13. 시각화
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=labels,
    cmap='tab10',
    s=100,
    edgecolors='k'
)

# 일부 저자 라벨 표시
for i, name in enumerate(author_names):
    if i % 2 == 0:
        plt.text(X_reduced[i, 0]+0.3, X_reduced[i, 1], name, fontsize=9)

plt.title("저자–키워드 기반 클러스터링 (KMeans, PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("author_cluster_plot.png", dpi=300)
plt.show()

print("\n시각화 이미지 'author_cluster_plot.png' 저장 완료")