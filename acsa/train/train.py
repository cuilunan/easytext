#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
模型训练

Authors: PanXu
Date:    2020/07/18 18:32:00
"""
from typing import Dict

import torch
from torch.utils.data import DataLoader

from easytext.trainer import Trainer
from easytext.data import Vocabulary, PretrainedVocabulary, LabelVocabulary

from acsa.data.dataset import SemEvalDataset
from acsa.data.dataset import ACSASemEvalDataset
from acsa.data import VocabularyCollate
from acsa.data import ACSAModelCollate

from acsa.models import ATAELstm


class Train:
    """
    ACSA 模型训练
    """

    def __init__(self, config: Dict):
        self.config = config

    def build_vocabulary(self):
        training_dataset_file_path = self.config["training_dataset_file_path"]

        dataset = SemEvalDataset(dataset_file_path=training_dataset_file_path)

        collate_fn = VocabularyCollate()
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=10,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=collate_fn)

        tokens = list()
        categories = list()
        labels = list()
        for collate_dict in data_loader:
            tokens.append(collate_dict["tokens"])
            categories.append(collate_dict["category"])
            labels.append(collate_dict["labels"])

        token_vocabulary = Vocabulary(
            tokens=tokens,
            padding=Vocabulary.PADDING,
            unk=Vocabulary.UNK,
            special_first=True
        )

        category_vocabulary = LabelVocabulary(labels=categories, padding=None)
        label_vocabulary = LabelVocabulary(labels=labels, padding=None)

        return {"token_vocabulary": token_vocabulary,
                "category_vocabulary": category_vocabulary,
                "label_vocabulary": label_vocabulary}



    def build_model(self):
        pass

    def __call__(self):

        pass