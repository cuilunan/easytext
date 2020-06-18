#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
对模型进行训练

Authors: panxu(panxu@baidu.com)
Date:    2020/06/17 18:31:00
"""

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from easytext.data import Vocabulary, LabelVocabulary
from easytext.trainer import Trainer

from event.event_detection_without_tirgger.data import ACEDataset
from event.event_detection_without_tirgger.data import EventDataset
from event.event_detection_without_tirgger.data import EventCollate
from event.event_detection_without_tirgger.data import EventVocabularyCollate
from event.event_detection_without_tirgger.loss import EventLoss
from event.event_detection_without_tirgger.metrics import EventF1MetricAdapter
from event.event_detection_without_tirgger.models import EventModel
from event.event_detection_without_tirgger.optimizer import EventOptimizerFactory


class Train:
    """
    训练的入口
    """

    def __call__(self, config: Dict):
        serialize_dir = config["serialize_dir"]
        train_dataset_file_path = config["train_dataset_file_path"]
        validation_dataset_file_path = config["validation_dataset_file_path"]

        # 构建词汇表
        ace_dataset = ACEDataset(train_dataset_file_path)
        vocab_data_loader = DataLoader(dataset=ace_dataset,
                                       batch_size=10,
                                       shuffle=False, num_workers=0,
                                       collate_fn=EventVocabularyCollate())

        tokens: List[List[str]] = list()
        event_types: List[List[str]] = list()
        entity_tags: List[List[str]] = list()

        for colleta_dict in vocab_data_loader:
            tokens.extend(colleta_dict["tokens"])
            event_types.extend(colleta_dict["event_types"])
            entity_tags.extend(colleta_dict["entity_tags"])

        word_vocabulary = Vocabulary(tokens=tokens,
                                     padding=Vocabulary.PADDING,
                                     unk=Vocabulary.UNK,
                                     special_first=True)

        event_type_vocabulary = Vocabulary(tokens=event_types,
                                           padding="",
                                           unk="Negative",
                                           special_first=True)

        entity_tag_vocabulary = LabelVocabulary(labels=entity_tags,
                                                padding=LabelVocabulary.PADDING)

        model = EventModel(alpha=0.5,
                           activate_score=True,
                           sentence_vocab=word_vocabulary,
                           sentence_embedding_dim=300,
                           entity_tag_vocab=entity_tag_vocabulary,
                           entity_tag_embedding_dim=50,
                           event_type_vocab=event_type_vocabulary,
                           event_type_embedding_dim=300,
                           lstm_hidden_size=300,
                           lstm_encoder_num_layer=1,
                           lstm_encoder_droupout=0.4)

        trainer = Trainer(
            serialize_dir=serialize_dir,
            num_epoch=2,
            model=model,
            loss=EventLoss(),
            optimizer_factory=EventOptimizerFactory(),
            metrics=EventF1MetricAdapter(event_type_vocabulary=event_type_vocabulary),
            patient=10,
            num_check_point_keep=5,
            cuda_devices=None
        )

        train_dataset = EventDataset(dataset_file_path=train_dataset_file_path,
                                     event_type_vocabulary=event_type_vocabulary)
        validation_dataset = EventDataset(dataset_file_path=validation_dataset_file_path,
                                          event_type_vocabulary=event_type_vocabulary)

        event_collate = EventCollate(word_vocabulary=word_vocabulary,
                                     event_type_vocabulary=event_type_vocabulary,
                                     entity_tag_vocabulary=entity_tag_vocabulary,
                                     sentence_max_len=512)
        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_size=2,
                                       num_workers=0,
                                       collate_fn=event_collate)

        validation_data_loader = DataLoader(dataset=validation_dataset,
                                            batch_size=2,
                                            num_workers=0,
                                            collate_fn=event_collate)

        trainer.train(train_data_loader=train_data_loader,
                      validation_data_loader=validation_data_loader)


if __name__ == '__main__':

    config = {
        "serialize_dir": "/Users/panxu/Serialize/event_detection_without_tirgger",
        "train_dataset_file_path": "/Users/panxu/MyProjects/github/easytext/data/event/event_detection_without_tirgger/tests/training_data_sample.txt",
        "validation_dataset_file_path": "/Users/panxu/MyProjects/github/easytext/data/event/event_detection_without_tirgger/tests/training_data_sample.txt"
    }

    Train()(config=config)
