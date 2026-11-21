import time

import torch

# AutoConfig: 自动配置
# AutoModel: 自动加载模型
# AutoTokenizer: 自动加载与模型匹配的分词器
from transformers import AutoConfig, AutoModel, AutoTokenizer

# AutoModelForSequenceClassification: 文本分类
# AutoModelForMaskedLM: 掩码语言建模
# AutoModelForQuestionAnswering: 问题问答模型
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
)

# AutoModelForSeq2SeqLM: 序列到序列
# AutoModelForTokenClassification: Token的分类
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification

from rich import print

device = "cuda" if torch.cuda.is_available() else "cpu"


def dm01_text_classification():
    load_start_time = time.time()
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("../PretrainedModel/chinese_sentiment")
    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        "../PretrainedModel/chinese_sentiment"
    ).to(device)

    print(f"模型加载耗时: {time.time() - load_start_time:.2f}s")

    # 定义文本
    context = "我最喜欢的水果是苹果。"

    eval_start_time = time.time()

    # 编码阶段
    inputs = tokenizer(
        text=context,  # 待编码文本
        return_tensors="pt",  # 返回类型 pytorch
        padding="max_length",  # 填充长度
        truncation=True,  # 是否截断
        max_length=512,  # 最大长度
    ).to(device)

    # 开启模型推理模式
    model.eval()

    # 解码阶段
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = torch.argmax(probs).item()
    print(f"模型推理耗时: {time.time() - eval_start_time:.2f}s")
    print(
        {
            "text": context,
            "label": model.config.id2label[pred_id],
            "score": round(probs[pred_id].item(), 4),
        }
    )


def dm02_feature_extraction():
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("../PretrainedModel/bert-base-chinese")
    # 加载模型 - 不带头
    model = AutoModel.from_pretrained("../PretrainedModel/bert-base-chinese")

    context = ["我了个Apex", "是人类啊"]

    # 编码阶段
    inputs = tokenizer(
        text=context,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=30,
    )

    model.eval()

    # 解码阶段
    logits = model(**inputs)

    print(logits)


def dm03_fill_mask():
    model_path = "../PretrainedModel/chinese-bert-wwm"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)

    context = "我真的很[MASK]你"

    inputs = tokenizer(
        text=context,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=30,
    )

    model.eval()

    with torch.no_grad():
        logits = model(**inputs).logits

    # 1. 找到 [MASK] 的位置
    mask_pos = (
        (inputs["input_ids"] == tokenizer.mask_token_id)
        .nonzero(as_tuple=True)[1]
        .item()
    )

    # 2. 提取该位置的 logits（形状是 [21128]）
    mask_logits = logits[0][mask_pos]

    # 3. 把原始打分变成概率（Softmax），便于人类理解
    mask_probs = torch.softmax(mask_logits, dim=-1)

    # 4. 核心：使用 topk 取出前 5 个最高概率和对应的词 ID
    top_k = 5
    top_probs, top_indices = torch.topk(mask_probs, k=top_k)

    # 5. 循环打印结果
    print(f"原句: 我真的很[MASK]你")
    print(f"[MASK] 预测 Top {top_k}:")
    for prob, idx in zip(top_probs, top_indices):
        # 用 .item() 转成普通数字，传入列表给 decode
        word = tokenizer.decode([idx.item()])
        # prob.item() 是概率，乘 100 变成百分比形式
        print(f"  - 候选词: '{word}' \t 概率: {prob.item() * 100:.2f}%")


def dm04_question_answering():
    model_path = "../PretrainedModel/chinese_pretrain_mrc_roberta_wwm_ext_large"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)

    context = "而且，这里还会遇到一个经典坑：偏移量映射（offset mapping）。因为 tokenizer 可能会把中文切成子词，甚至加上特殊符号，导致 token 的下标和原文字符串的下标对不上。到时候如果你遇到截取出来的答案是乱码或者错位，记得再来找我，那是下一个大坑（可以用 return_offsets_mapping=True 解决）。"
    question = ["经典坑是什么？", "截取出来的答案是乱码或者错位该怎么解决?"]

    for q in question:
        inputs = tokenizer(
            text=q,
            text_pair=context,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=30,
        ).to(device)
        model.eval()
        with torch.no_grad():
            output = model(**inputs)

        start_logits = output.start_logits[0]
        end_logits = output.end_logits[0]

        # 1. 核心：获取 sequence_ids，把非上下文的位置得分全部设为 -1e9
        # sequence_ids(0) 会返回每个 token 属于哪一句：0=问题, 1=上下文, None=特殊符号
        sequence_ids = inputs.sequence_ids(0)
        for i, seq_id in enumerate(sequence_ids):
            if seq_id != 1:  # 如果不是上下文，直接抹杀它成为答案的可能
                start_logits[i] = -1e9
                end_logits[i] = -1e9

        # 2. 获取最高分的 start 和 end 索引
        start = torch.argmax(start_logits).item()
        end = torch.argmax(end_logits).item()

        # 防御性编程：如果 end 跑到了 start 前面，直接强制修正
        if end < start:
            end = start

        # 3. 切片并解码
        answer_ids = inputs["input_ids"][0][start : end + 1]

        # 解决方法一：用 convert_tokens_to_string 完美拼接中文，去掉空格
        tokens = tokenizer.convert_ids_to_tokens(answer_ids)
        answer = tokenizer.convert_tokens_to_string(tokens)

        # 解决方法二：使用 decode 时去掉特殊符号（如果答案里出现了 [SEP] 会被自动过滤）
        # answer = tokenizer.decode(answer_ids, skip_special_tokens=True).replace(" ", "")

        print(f"问题: {q}")
        print(f"答案: {answer}\n")


def dm05_summerization():
    model_path = "../PretrainedModel/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    text = (
        "Artificial intelligence is rapidly transforming the healthcare industry, offering "
        "unprecedented opportunities for early disease detection and personalized treatment plans. "
        "Recent studies indicate that AI algorithms can analyze medical imaging, such as X-rays "
        "and MRIs, with an accuracy that often matches or even surpasses human radiologists. "
        "By identifying subtle patterns invisible to the human eye, these systems can detect "
        "conditions like cancer at much earlier stages, significantly improving patient survival rates. "
        "However, the widespread adoption of AI in clinical settings is not without challenges. "
        "Concerns regarding data privacy, algorithmic bias, and the 'black box' nature of some "
        "deep learning models remain significant barriers. Healthcare professionals emphasize "
        "that AI should not replace human doctors but rather serve as a powerful assistive tool. "
        "Furthermore, integrating these complex systems into existing hospital infrastructures "
        "requires substantial investment and rigorous regulatory approval. Despite these hurdles, "
        "experts predict that the future of medicine will heavily rely on human-AI collaboration, "
        "ultimately leading to more efficient healthcare systems and better patient outcomes worldwide."
    )

    inputs = tokenizer(text=text, return_tensors="pt", truncation=True).to(device)

    model.eval()
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=130,  # 限制最大长度
            min_length=30,  # 限制最小长度
            num_beams=4,  # 集束搜索，让句子更通顺
            early_stopping=True,
            forced_bos_token_id=0,  # 解决警告，强制以 <s> 开始
        )

    print(tokenizer.decode(output[0], skip_special_tokens=True))


def dm06_ner():
    model_path = "../PretrainedModel/roberta-base-finetuned-cluener2020-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    config = AutoConfig.from_pretrained(model_path)

    context = "我爱北京天安门, 天安门上太阳升!"

    inputs = tokenizer(text=context, return_tensors="pt", truncation=True).to(device)
    print(inputs)

    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits

    print(logits.shape)
    for input_idx,ner_idx in zip(inputs["input_ids"][0], logits[0]):
        print(f'{tokenizer.decode(input_idx)}: {config.id2label[torch.argmax(ner_idx).item()]}')



if __name__ == "__main__":
    # dm01_text_classification()
    # dm02_feature_extraction()
    # dm03_fill_mask()
    # dm04_question_answering()
    # dm05_summerization()
    dm06_ner()
