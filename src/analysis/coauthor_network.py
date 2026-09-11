import igraph as ig
import pickle
import cairo
from tqdm import tqdm
from collections import Counter

# 1. pkl 불러오기
with open('arxiv_2021_2023.pkl', 'rb') as f:
    data = pickle.load(f)

print("전체 논문 수:", len(data))

# 2. 저자명 정제 함수 정의
def is_valid_author(name):
    name = name.strip()
    return (
        len(name) > 2 and
        not name.isdigit() and
        not name.isupper() and
        not name.endswith(')') and
        not any(c in name for c in [';', ':', '/', '\\', '(', ')', '[', ']', '{', '}', '.', '*'])
    )

# 3. 저자 추출 및 엣지 구성
author_set = set()
edges = set()

for paper in tqdm(data, desc="저자 엣지 추출 중"):
    authors_raw = paper.get('authors', '')
    authors = [
        a.strip() for a in authors_raw.split(',')
        if a.strip() and is_valid_author(a.strip())
    ]
    
    if len(authors) > 50:
        continue

    author_set.update(authors)
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            edge = tuple(sorted((authors[i], authors[j])))
            edges.add(edge)

# 4. 저자 리스트 및 엣지 변환
author_list = list(author_set)
author_idx = {name: idx for idx, name in enumerate(author_list)}
indexed_edges = [(author_idx[a1], author_idx[a2]) for a1, a2 in edges]

# 5. 그래프 생성
g = ig.Graph()
g.add_vertices(len(author_list))
g.vs["name"] = author_list
g.add_edges(indexed_edges)

# 6. 네트워크 분석
print(f"\n총 노드 수: {g.vcount()}")
print(f"총 엣지 수: {g.ecount()}")

degree = g.degree()
pagerank = g.pagerank()

components_all = g.components()
giant = components_all.giant()

# 7. 중심성 상위 저자 출력
top_degree = sorted(zip(g.vs["name"], degree), key=lambda x: x[1], reverse=True)[:10]
print("\nDegree 중심성 상위 저자:")
for name, d in top_degree:
    print(f" - {name}: {d}")

print(f"\n최대 연결 성분 노드 수: {giant.vcount()}")
print(f"연결 성분 개수: {len(components_all)}")

# 8. 클러스터링 계수 출력
clustering_coeffs = g.transitivity_local_undirected(mode="zero")
avg_clustering = sum(clustering_coeffs) / len(clustering_coeffs)
print(f"전체 평균 클러스터링 계수: {avg_clustering:.4f}")

# 9. 중심성 상위 30명 시각화
top30 = sorted(zip(range(len(degree)), degree), key=lambda x: x[1], reverse=True)[:30]
top_indices = [idx for idx, _ in top30]
subgraph = g.subgraph(top_indices)
layout = subgraph.layout("fr")

ig.plot(
    subgraph,
    target="coauthor_top30_highres.png",
    layout=layout,
    bbox=(2500, 2500),
    margin=50,
    vertex_size=25,
    vertex_label=subgraph.vs["name"],
    edge_width=0.6
)

print("\n'coauthor_top30_highres.png'로 고해상도 이미지 저장 완료")