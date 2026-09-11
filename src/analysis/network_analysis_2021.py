import pickle
import re
import nltk
from tqdm import tqdm
from collections import defaultdict, Counter
from itertools import combinations
import igraph as ig

# 불용어 로딩
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# 1. 키워드 추출 함수
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

# 2. 국가 및 기관명 필터링 목록
filtered_names = {
    'usa', 'germany', 'france', 'italy', 'japan', 'uk', 'china', 'spain', 'canada', 'india',
    'korea', 'australia', 'switzerland', 'netherlands', 'united states', 'europe', 'russia',
    'institution', 'university', 'center', 'institute', 'team', 'group', 'department',
    'cnrs', 'csic', 'esa', 'lab'
}

# 3. 분석 연도 설정
TARGET_YEAR = 2021

# 4. 데이터 로딩 및 해당 연도 필터링
with open('arxiv_2021_2023.pkl', 'rb') as f:
    full_data = pickle.load(f)

data = [
    paper for paper in full_data
    if 'update_date' in paper and paper['update_date'].startswith(str(TARGET_YEAR))
]

print(f"[{TARGET_YEAR}] 논문 수: {len(data)}")

# 5. 저자 협업 네트워크 구성 (필터링 포함)
coauthor_edges = []
for paper in data:
    authors_raw = paper.get('authors', '')
    authors_all = [a.strip() for a in authors_raw.split(',') if a.strip()]

    # 저자 필터링
    authors = []
    for author in authors_all:
        author_lower = author.lower()
        if (
            any(word in author_lower for word in filtered_names) or
            re.fullmatch(r'\d+\)?', author.strip()) or
            len(author.strip()) < 3 or
            re.search(r'[^a-zA-Z\s\.]', author)
        ):
            continue
        authors.append(author)

    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            coauthor_edges.append(tuple(sorted((authors[i], authors[j]))))

coauthor_counts = Counter(coauthor_edges)
coauthor_nodes = set(a for edge in coauthor_counts for a in edge)
coauthor_idx = {a: i for i, a in enumerate(coauthor_nodes)}
coauthor_edges_indexed = [(coauthor_idx[a], coauthor_idx[b]) for a, b in coauthor_counts]

g_author = ig.Graph()
g_author.add_vertices(len(coauthor_nodes))
g_author.vs['name'] = list(coauthor_nodes)
g_author.add_edges(coauthor_edges_indexed)

print(f"[{TARGET_YEAR}] 저자 네트워크: 노드 {g_author.vcount()}, 엣지 {g_author.ecount()}")

# 6. 상위 30 저자 시각화
degree = g_author.degree()
top30 = sorted(zip(range(len(degree)), degree), key=lambda x: x[1], reverse=True)[:30]
top_indices = [idx for idx, _ in top30]
sub_author = g_author.subgraph(top_indices)
layout = sub_author.layout("fr")

ig.plot(
    sub_author,
    target=f"author_network_{TARGET_YEAR}_top30.png",
    layout=layout,
    bbox=(2000, 2000),
    margin=50,
    vertex_size=25,
    vertex_label=sub_author.vs["name"],
    edge_width=0.6
)
print(f"[{TARGET_YEAR}] author_network_{TARGET_YEAR}_top30.png 저장 완료")

print(f"\n[{TARGET_YEAR}] 상위 30 저자 (Degree 중심성 기준):")
for idx, deg in top30:
    print(f"{g_author.vs[idx]['name']:25s} | Degree: {deg}")

# 7. 키워드 공출현 네트워크 구성
keyword_edges = []
for paper in data:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    keywords = extract_keywords(text, top_n=10)
    keyword_edges.extend([tuple(sorted(pair)) for pair in combinations(keywords, 2)])

keyword_counts = Counter(keyword_edges)
filtered_keyword_edges = [edge for edge, count in keyword_counts.items() if count >= 3]
keyword_nodes = set(w for edge in filtered_keyword_edges for w in edge)
keyword_idx = {w: i for i, w in enumerate(keyword_nodes)}
keyword_edges_indexed = [(keyword_idx[a], keyword_idx[b]) for a, b in filtered_keyword_edges]

g_keyword = ig.Graph()
g_keyword.add_vertices(len(keyword_nodes))
g_keyword.vs['name'] = list(keyword_nodes)
g_keyword.add_edges(keyword_edges_indexed)

print(f"[{TARGET_YEAR}] 키워드 네트워크: 노드 {g_keyword.vcount()}, 엣지 {g_keyword.ecount()}")

# 8. 상위 30 키워드 시각화
degree_kw = g_keyword.degree()
top30_kw = sorted(zip(range(len(degree_kw)), degree_kw), key=lambda x: x[1], reverse=True)[:30]
top_indices_kw = [idx for idx, _ in top30_kw]
sub_keyword = g_keyword.subgraph(top_indices_kw)
layout_kw = sub_keyword.layout("fr")

ig.plot(
    sub_keyword,
    target=f"keyword_network_{TARGET_YEAR}_top30.png",
    layout=layout_kw,
    bbox=(2000, 2000),
    margin=50,
    vertex_size=25,
    vertex_label=sub_keyword.vs["name"],
    edge_width=0.6
)
print(f"[{TARGET_YEAR}] keyword_network_{TARGET_YEAR}_top30.png 저장 완료")

print(f"\n[{TARGET_YEAR}] 상위 30 키워드 (Degree 중심성 기준):")
for idx, deg in top30_kw:
    print(f"{g_keyword.vs[idx]['name']:20s} | Degree: {deg}")