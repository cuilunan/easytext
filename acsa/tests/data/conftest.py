#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
conf test

Authors: PanXu
Date:    2020/07/12 15:57:00
"""
import os
import pytest

from acsa import ROOT_PATH
from acsa.data.dataset import SemEvalDataset
from acsa.data.dataset import ACSASemEvalDataset


@pytest.fixture(scope="package")
def sem_eval_dataset():
    sample_dataset_file_path = "data/dataset/SemEval-2014-Task-4-REST/sample.xml"
    sample_dataset_file_path = os.path.join(ROOT_PATH, sample_dataset_file_path)

    dataset = SemEvalDataset(dataset_file_path=sample_dataset_file_path)

    return dataset


@pytest.fixture(scope="package")
def acsa_sem_eval_dataset():
    sample_dataset_file_path = "data/dataset/SemEval-2014-Task-4-REST/sample.xml"
    sample_dataset_file_path = os.path.join(ROOT_PATH, sample_dataset_file_path)

    dataset = ACSASemEvalDataset(dataset_file_path=sample_dataset_file_path)

    return dataset