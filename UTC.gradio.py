#paddlenlp版本要求2.5.2，别忘记重新pip 一下
import gradio as gr
from paddlenlp import Taskflow
import pandas as pd
import os


schema = ["病情诊断", "治疗方案", "病因分析", "指标解读", "就医建议", "疾病表述", "后果表述", "注意事项", "功效作用", "医疗费用", "其他"]

# my_cls = Taskflow("zero_shot_text_classification", model="utc-base", schema=schema)
my_cls = Taskflow("zero_shot_text_classification", model="utc-base", schema=schema, task_path='/home/aistudio/checkpoint/model_best/plm')


def information_extraction(input_text):
    output_text =my_cls(input_text)
    return output_text
    
demo = gr.Interface(fn=information_extraction, inputs="text", outputs="text", examples=["老年斑为什么都长在面部和手背上","老成都市哪家内痔医院比较好怎么样最好？","中性粒细胞比率偏低"],title="基于UTC的医疗意图多分类",description="本项目提供基于通用文本分类 UTC（Universal Text Classification） 模型微调的文本分类端到端应用方案，打通数据标注-模型训练-模型调优-预测部署全流程，可快速实现文本分类产品落地。",live=True)

demo.launch()   



