import re
import os
import pandas as pd
import matplotlib.pyplot as plt

# 定义清洗函数
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_special_characters(text):
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5\s]', '', text)  # 保留字母、数字和中文字符
    return text

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def clean_tag(text):
    clean_text = re.sub(r'#.*? ', '', text)
    clean_text = re.sub(r'@.*? ', '', clean_text)  # 去掉@xxx
    clean_text = re.sub(r'#\w+', '', clean_text)  # 去掉#xxx
    clean_text = re.sub(r'@\w+', '', clean_text)  # 去掉@xxx
    return clean_text

def remove_specific_characters_regex(text):
    text = re.sub('车友圈广场 ', '', text)
    text = re.sub('车友圈广场', '', text)
    return text

def clean_text(text):
    text = clean_tag(text)
    text = remove_specific_characters_regex(text)
    text = remove_extra_spaces(text)
    # 如果有需要，可以添加更多清洗步骤
    return text

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建文件路径
file_path = os.path.join(current_dir, 'datasets', '上汽数据-宋Plus.xlsx')

# 读取数据
saic_data = pd.read_excel(file_path)

# 处理缺失值并转换数据类型
saic_data['内文'] = saic_data['内文'].fillna('').astype(str)

# 对内文进行清洗
saic_data['清洗后内文'] = saic_data['内文'].apply(clean_text)

# 文本分割
essays = saic_data['清洗后内文'].tolist()
single_sentences_list = []
for essay in essays:
    single_sentences_list.extend(re.split(r'(?<=[。?？!！])', essay))

# 过滤空文本
single_sentences_list = list(set(single_sentences_list))
single_sentences_list = [sentence for sentence in single_sentences_list if sentence]

# 过滤句子长度
filtered_sentences_len = [sentence for sentence in single_sentences_list if 3 <= len(sentence) < 200]

# 过滤非中文文本
chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
filtered_sentences_NoneAD = [sentence for sentence in filtered_sentences_len if '广告' not in sentence]
filtered_sentences_CN = [sentence for sentence in filtered_sentences_NoneAD if chinese_pattern.search(sentence)]

# 打印结果
print(f"共有 {len(filtered_sentences_CN)} 个句子包含中文字符")

# 保存结果
with open('filtered_sentences_CN.txt', 'w', encoding='utf-8') as file:
    for sentence in filtered_sentences_CN:
        file.write(sentence + '\n')

# 可视化句子长度分布
sentence_lengths = [len(sentence) for sentence in filtered_sentences_CN]
plt.hist(sentence_lengths, bins=500)
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.title('Length Distribution of Sentences')
plt.show()