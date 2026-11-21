import fasttext_new as fasttext


def dm01_basic():
    print()
    print("=" * 32, "基础训练", "=" * 32)
    # 模型训练
    model = fasttext.train_supervised("./data/cooking_train.txt", epoch=100)

    # 模型评估
    train = model.test("./data/cooking_train.txt")
    print("训练集：", train)

    test = model.test("./data/cooking_valid.txt")
    print("测试集：", test)


def dm02_pre():
    print()
    print("=" * 32, "数据预处理", "=" * 32)
    # 模型训练
    model = fasttext.train_supervised("./data/cooking.pre.train", epoch=100)

    # 模型评估
    train = model.test("./data/cooking.pre.train")
    print("训练集：", train)

    test = model.test("./data/cooking.pre.valid")
    print("测试集：", test)


def dm03_lr():
    print()
    print("=" * 32, "数据预处理 + 学习率", "=" * 32)
    # 模型训练
    model = fasttext.train_supervised(
        "./data/cooking.pre.train", epoch=100, lr=0.5
    )

    # 模型评估
    train = model.test("./data/cooking.pre.train")
    print("训练集：", train)

    test = model.test("./data/cooking.pre.valid")
    print("测试集：", test)


def dm04_ngram():
    print()
    print("=" * 32, "数据预处理 + 学习率 + N-gram", "=" * 32)
    # 模型训练
    model = fasttext.train_supervised(
        "./data/cooking.pre.train", epoch=100, lr=0.5, wordNgrams=2
    )

    # 模型评估
    train = model.test("./data/cooking.pre.train")
    print("训练集：", train)

    test = model.test("./data/cooking.pre.valid")
    print("测试集：", test)


def dm05_loss():
    print()
    print("=" * 32, "数据预处理 + 学习率 + N-gram + 层次softmax损失函数", "=" * 32)
    # 模型训练
    model = fasttext.train_supervised(
        "./data/cooking.pre.train", epoch=100, lr=0.5, wordNgrams=2, loss="hs"
    )

    # 模型评估
    train = model.test("./data/cooking.pre.train")
    print("训练集：", train)

    test = model.test("./data/cooking.pre.valid")
    print("测试集：", test)


def dm06_auto():
    print()
    print("=" * 32, "数据预处理 + 自动调优", "=" * 32)
    # 模型训练
    model = fasttext.train_supervised(
        input="./data/cooking.pre.train",
        autotuneValidationFile="./data/cooking.pre.valid",
        autotuneDuration=30,
    )

    print(f"学习率 (lr): {model.lr}")
    print(f"训练轮数 (epoch): {model.epoch}")
    print(f"词向量维度 (dim): {model.dim}")
    print(f"N-gram 设置: {model.wordNgrams}")
    print(f"损失函数: {model.loss}")
    print(f"最小词频: {model.minCount}")

    # 模型评估
    train = model.test("./data/cooking.pre.train")
    print("训练集：", train)

    test = model.test("./data/cooking.pre.valid")
    print("测试集：", test)


def dm07_multi_label():
    print()
    print("=" * 32, "数据预处理 + 学习率 + N-gram + 多分类损失函数ova", "=" * 32)
    # 模型训练
    model = fasttext.train_supervised(
        "./data/cooking.pre.train", epoch=100, lr=0.5, wordNgrams=2, loss="ova"
    )

    # 模型评估
    train = model.test("./data/cooking.pre.train")
    print("训练集：", train)

    test = model.test("./data/cooking.pre.valid")
    print("测试集：", test)


def dm08_save():
    print()
    print("=" * 32, "模型保存", "=" * 32)
    # 模型训练
    model = fasttext.train_supervised(
        "./data/cooking.pre.train", epoch=100, lr=0.5, wordNgrams=2, loss="ova"
    )

    # 模型保存
    model.save_model("./model/cokking_model.model")


def dm09_load():
    print()
    print("=" * 32, "模型加载", "=" * 32)
    model = fasttext.load_model("./model/cokking_model.model")

    # 模型预测
    result = model.predict(
        "How do I fix a cast iron pot that was heated empty for hours?"
    )
    print(result)

    # 模型评估
    train = model.test("./data/cooking.pre.train")
    print("训练集：", train)

    test = model.test("./data/cooking.pre.valid")
    print("测试集：", test)


if __name__ == "__main__":
    # dm01_basic()    # (3000, 0.5056666666666667, 0.21868242756234685)
    # dm02_pre()      # (3000, 0.5533333333333333, 0.2392965258757388)
    # dm03_lr()       # (3000, 0.5683333333333334, 0.24578347989044255)
    # dm04_ngram()    # (3000, 0.626, 0.270722214213637)
    # dm05_loss()     # (3000, 0.583, 0.25212627937148624)
    # dm06_auto()     # (3000, 0.577, 0.24953149776560474)  300s
    # dm07_multi_label()  # (3000, 0.6123333333333333, 0.2648118783335736)
    # dm08_save()
    dm09_load()
