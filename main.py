from paddlenlp import Taskflow
import json
from tqdm import tqdm

# 读取 schema 文件
with open('./data/3label.txt', 'r') as file:
    schema = [line.strip() for line in file]

# 初始化分类任务
my_cls = Taskflow("zero_shot_text_classification", model="utc-base", schema=schema, device_id=-1)

# 定义生成器逐行读取文件
def read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                yield stripped_line

# 分批处理数据
batch_size = 128  # 每批处理的行数
results = []

# 计算文件总行数
total_lines = sum(1 for _ in read_lines('./data/test.txt'))

# 使用 tqdm 显示进度条
with tqdm(total=total_lines, desc="Processing") as pbar:
    batch = []
    for line in read_lines('./data/test.txt'):
        batch.append(line)
        if len(batch) == batch_size:
            print(1)
            batch_result = my_cls(batch)
            results.extend(batch_result)
            batch = []
        pbar.update(1)

    # 处理剩余的行
    if batch:
        batch_result = my_cls(batch)
        results.extend(batch_result)
        pbar.update(len(batch))

# 将结果保存为 JSON 文件
with open('./result/result.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
