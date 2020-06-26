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

import itertools
from torch.utils.data import Dataset


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

        logging.info(f"Begin read conll2003 dataset: {dataset_file_path}")

        with open(dataset_file_path, encoding="utf-8") as data_file:

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens_, pos_tags, chunk_tags, ner_tags = fields
                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in tokens_]

                    yield self.text_to_instance(tokens, pos_tags, chunk_tags, ner_tags)

    @staticmethod
    def _is_divider(line: str) -> bool:
        """
        判断该行是否是 分割的。包括两种情况: 1. 空行 2. "-DOCSTART-" 这两种否是分割的
        :param line: 行的内容
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

