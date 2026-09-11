import json
import os
import pickle
from tqdm import tqdm

# 경로 설정
json_filename = 'arxiv-metadata-oai-snapshot.json'
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, json_filename)
output_path = os.path.join(current_dir, 'arxiv_2021_2023.pkl')

# 2021~2023년 데이터만 저장
def parse_filtered_years(json_path, output_path):
    records = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="2021~2023년 JSON 필드 파싱 중"):
            try:
                paper = json.loads(line)
                # 날짜 형식: 'update_date': '2020-05-04'
                year = int(paper.get('update_date', '')[:4])
                if 2021 <= year <= 2023:
                    records.append(paper)
            except:
                continue  # 파싱 실패 시 무시

    with open(output_path, 'wb') as f:
        pickle.dump(records, f)
    print(f"총 {len(records)}건 저장 완료: {output_path}")

if __name__ == '__main__':
    parse_filtered_years(json_path, output_path)