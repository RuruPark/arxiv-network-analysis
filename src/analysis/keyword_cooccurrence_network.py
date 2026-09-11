import pickle
import igraph as ig
import cairo
import re
import nltk
from tqdm import tqdm
from itertools import combinations
from collections import Counter
import os

# NLTK 리소스 다운로드 (최초 1회만 실행)
nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. 데이터 불러오기 (2023년 논문만 필터링)
with open('arxiv_2021_2023.pkl', 'rb') as f:
    full_data = pickle.load(f)

data = [
    paper for paper in full_data
    if 'update_date' in paper and paper['update_date'].startswith('2023')
]

print("2023년 논문 수:", len(data))

# 2. 불용어 목록 설정
stop_words = set(stopwords.words('english'))

# 3. 키워드 추출 함수 정의 (등장 순서 기준 상위 N개 키워드 선택)
def extract_keywords(text, top_n=10):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # 특수문자 제거 및 소문자화
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

# 4. 키워드 공출현 엣지 생성
all_edges = []
print("논문 처리 중...")
for paper in tqdm(data, desc="Processing papers"):
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    text = f"{title} {abstract}"

    # 논문별 등장 순서 기준 상위 10개 키워드만 사용
    keywords = extract_keywords(text, top_n=10)

    # 키워드 간 공출현 쌍 생성
    all_edges.extend([tuple(sorted(pair)) for pair in combinations(keywords, 2)])

# 5. 엣지 빈도수 계산 후 필터링 (3회 이상 공출현한 키워드 페어만 사용)
edge_counts = Counter(all_edges)
filtered_edges = [edge for edge, count in edge_counts.items() if count >= 3]

# 6. 노드 및 인덱싱
keyword_set = set()
for a, b in filtered_edges:
    keyword_set.add(a)
    keyword_set.add(b)

keyword_list = list(keyword_set)
keyword_idx = {word: idx for idx, word in enumerate(keyword_list)}
indexed_edges = [(keyword_idx[a], keyword_idx[b]) for a, b in filtered_edges]

# 7. 그래프 생성
g = ig.Graph()
g.add_vertices(len(keyword_list))
g.vs["name"] = keyword_list
g.add_edges(indexed_edges)

# 8. 네트워크 분석 지표 계산
degree = g.degree()
pagerank = g.pagerank()
components_all = g.components()
giant = components_all.giant()
clustering_coeffs = g.transitivity_local_undirected(mode="zero")
avg_clustering = sum(clustering_coeffs) / len(clustering_coeffs)

print(f"총 노드 수: {g.vcount()}")
print(f"총 엣지 수: {g.ecount()}")
print(f"연결 성분 개수: {len(components_all)}")
print(f"최대 연결 성분 노드 수: {giant.vcount()}")
print(f"전체 평균 클러스터링 계수: {avg_clustering:.4f}")

# 9. 중심성 상위 30개 키워드로 서브그래프 생성 및 시각화
top30 = sorted(zip(range(len(degree)), degree), key=lambda x: x[1], reverse=True)[:30]
top_indices = [idx for idx, _ in top30]
subgraph = g.subgraph(top_indices)
layout = subgraph.layout("fr")

ig.plot(
    subgraph,
    target="keyword_top30_highres.png",
    layout=layout,
    bbox=(2500, 2500),
    margin=50,
    vertex_size=25,
    vertex_label=subgraph.vs["name"],
    edge_width=0.6
)

print("\n'keyword_top30_highres.png' 이미지 저장 완료")

# 10. 중심성 상위 30개 키워드 출력
print("\n상위 30개 키워드 (Degree 중심성 기준):")
for idx, deg in top30:
    print(f"{g.vs[idx]['name']:20s} | Degree: {deg}")