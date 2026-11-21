def stander_len(sentence_list, n):
    result = []
    for sentence in sentence_list:
        if len(sentence) > n:
            result.append(sentence[:n])
        else:
            result.append(sentence + [0] * (n - len(sentence)))
    return result


if __name__ == "__main__":
    x_train = [
        [1, 25, 5, 3, 2, 33, 5, 5, 12],
        [4, 2, 5, 23, 1, 24, 2, 12, 1, 24],
        [12, 5, 2],
    ]
    result = stander_len(x_train, 10)
    print(result)
