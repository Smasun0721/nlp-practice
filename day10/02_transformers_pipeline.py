import time

import torch
from transformers import pipeline
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def dm01_test_classification():
    load_start_time = time.time()
    model = pipeline(
        task="sentiment-analysis",
        model="../PretrainedModel/chinese_sentiment",
        device=device,
    )

    print(f"模型加载耗时: {time.time() - load_start_time:.2f}s")

    eval_start_time = time.time()
    output = model("我喜欢这个！")
    print(output)

    print(f"模型推理耗时: {time.time() - eval_start_time:.2f}s")


def dm02_test_feature_extraction():
    load_start_time = time.time()
    model = pipeline(
        task="feature-extraction",
        model="../PretrainedModel/bert-base-chinese",
        device=device,
    )

    print(f"模型加载耗时: {time.time() - load_start_time:.2f}s")

    eval_start_time = time.time()
    output = model("进程已结束，退出代码为 0")
    # print(output)
    print(torch.tensor(output).shape)

    print(f"模型推理耗时: {time.time() - eval_start_time:.2f}s")


def dm03_test_fill_mask():
    load_start_time = time.time()
    model = pipeline(
        task="fill-mask",
        model="../PretrainedModel/chinese-bert-wwm",
        device=device,
    )

    print(f"模型加载耗时: {time.time() - load_start_time:.2f}s")

    eval_start_time = time.time()
    output = model("进程已[MASK]束，退出代码为 0")
    print(output)

    print(f"模型推理耗时: {time.time() - eval_start_time:.2f}s")


# 已过时
def dm04_test_question_answering():
    load_start_time = time.time()
    context = ""
    question = [""]
    model = pipeline(
        task="question-answering",
        model="../PretrainedModel/chinese_pretrain_mrc_roberta_wwm_ext_large",
        device=device,
    )

    print(f"模型加载耗时: {time.time() - load_start_time:.2f}s")

    eval_start_time = time.time()
    output = model(
        question=question,
        context=context,
    )
    print(output)

    print(f"模型推理耗时: {time.time() - eval_start_time:.2f}s")


# 已过时
def dm05_test_summarization():
    load_start_time = time.time()
    model = pipeline(
        task="summarization",
        model="../PretrainedModel/distilbart-cnn-12-6",
        device=device,
    )

    print(f"模型加载耗时: {time.time() - load_start_time:.2f}s")

    eval_start_time = time.time()
    output = model(
        "sdviufhndfhnuvyfhnyiuabvabfiuvwabnv4ekutfjcmvkuejBgntiuEQWNgvhkujebhnIUbnegAESGREhny4jn"
    )
    print(output)

    print(f"模型推理耗时: {time.time() - eval_start_time:.2f}s")


def dm06_test_ner():
    load_start_time = time.time()
    model = pipeline(
        task="ner",
        model="../PretrainedModel/roberta-base-finetuned-cluener2020-chinese",
        device=device,
    )

    print(f"模型加载耗时: {time.time() - load_start_time:.2f}s")

    eval_start_time = time.time()
    output = model("马云在杭州创办了阿里巴巴集团。")
    print(output)

    print(f"模型推理耗时: {time.time() - eval_start_time:.2f}s")


if __name__ == "__main__":
    # dm01_test_classification()
    # dm02_test_feature_extraction()
    # dm03_test_fill_mask()
    # dm04_test_question_answering()
    # dm05_test_summarization()
    dm06_test_ner()
