# coding: utf-8
# 2022/4/25 @ tongshiwei

import random

random.seed(0)
SAMPLE_NUM = int(1e4)


def pseudo_data():
    return [[random.random() for _ in range(512)] for _ in range(SAMPLE_NUM)]


def pseudo_seq_data():
    return [[random.random() for _ in range(random.randint(0, 512))] for _ in range(SAMPLE_NUM)]
