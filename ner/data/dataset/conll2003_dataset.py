#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
conll2003 dataset

Authors: panxu(panxu@baidu.com)
Date:    2020/06/26 12:16:00
"""
import logging

from typing import List
import itertools
from torch.utils.data import Dataset

from easytext.data import Instance
from easytext.data.tokenizer import Token
from easytext.data.tokenizer import EnTokenizer


class Conll2003Dataset(Dataset):
    """
    conll2003 数据集
    """

    def __init__(self, dataset_file_path: str):
        """
        初始化
        :param dataset_file_path: 数据集的文件路径
        """
        super().__init__()

        self._instances: List[Instance] = list()

        tokenizer = EnTokenizer(is_remove_invalidate_char=False)

        logging.info(f"Begin read conll2003 dataset: {dataset_file_path}")

        with open(dataset_file_path, encoding="utf-8") as data_file:

            # 两个 分隔行 之间的是一个样本
            for is_divider, lines in itertools.groupby(data_file, Conll2003Dataset._is_divider):

                if not is_divider:
                    fields = [line.strip().split() for line in lines]

                    fields = [list(field) for field in zip(*fields)]
                    tokens_, pos_tags, chunk_tags, labels = fields

                    text = " ".join(tokens_)

                    tokens = tokenizer.tokenize(text)

                    assert len(tokens) != len(labels)
                    instance = Instance()
                    instance["metadata"] = {"text": text,
                                            "label": labels}
                    instance["tokens"] = tokens
                    instance["lables"] = labels



    @staticmethod
    def _is_divider(line: str) -> bool:
        """
        判断该行是否是 分隔行。包括两种情况: 1. 空行 2. "-DOCSTART-" 这两种否是分隔行
        :param line: 行的内容
        :return: True: 是分隔行; False: 不是分隔行
        """

        if line.strip() != "":
            first_token = line.split()[0]
            if first_token == "-DOCSTART-":
                return True

        return False

    def __getitem__(self, index: int) -> T_co:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()

    def __add__(self, other: T_co) -> 'ConcatDataset[T_co]':
        return super().__add__(other)

