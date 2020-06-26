#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 conll2003 dataset

Authors: panxu(panxu@baidu.com)
Date:    2020/06/26 22:07:00
"""
import os
import pytest

from ner import ROOT_PATH
from ner.tests import ASSERT
from ner.data.dataset import Conll2003Dataset


@pytest.fixture(scope="class")
def conll2003_dataset() -> Conll2003Dataset:
    """
    数据集生成
    :return: conll2003 数据集
    """
    dataset_file_path = "data/conll2003/sample.txt"
    dataset_file_path = os.path.join(ROOT_PATH, dataset_file_path)

    return Conll2003Dataset(dataset_file_path=dataset_file_path)


def test_conll2003_dataset(conll2003_dataset):
    """
    测试 conll2003 数据集
    :param conll2003_dataset: 数据集
    :return: None
    """

    ASSERT.assertEqual(2, len(conll2003_dataset))

    instance0 = conll2003_dataset[0]

    ASSERT.assertEqual(11, len(instance0["tokens"]))

    instance1 = conll2003_dataset[1]

    expect_labels = ["B-LOC", "O"]

    ASSERT.assertListEqual(expect_labels, instance1["sequence_label"])

