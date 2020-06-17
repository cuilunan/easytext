#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
event detection without trigger

Authors: panxu(panxu@baidu.com)
Date:    2020/01/31 09:11:00
"""
import json
import logging
from typing import Dict

import torch
from torch import LongTensor
from torch import Tensor
from torch.nn import Embedding
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from easytext.data import Vocabulary, LabelVocabulary
from easytext.model import Model
from easytext.model import ModelOutputs
from easytext.utils.nn import nn_util


class EventModelOutputs(ModelOutputs):
    """
    Event Model 的输出数据
    """

    def __init__(self, logits: torch.Tensor, event_type: torch.LongTensor):
        super().__init__(logits)
        self.event_type = event_type


class EventModel(Model):
    """
    event detection without trigger

    ACL 2019 reference: https://www.aclweb.org/anthology/N19-1080/
    """

    def __init__(self,
                 alpha: float,
                 activate_score: bool,
                 sentence_vocab: Vocabulary,
                 sentence_embedding_dim: int,
                 entity_tag_vocab: Vocabulary,
                 entity_tag_embedding_dim: int,
                 event_type_vocab: Vocabulary,
                 event_type_embedding_dim: int,
                 lstm_hidden_size: int,
                 lstm_encoder_num_layer: int,
                 lstm_encoder_droupout: float):

        super().__init__()

        self._alpha = alpha
        self._activate_score = activate_score
        self._sentence_vocab = sentence_vocab
        self._sentence_embedder = Embedding(self._sentence_embedder.size,
                                             embedding_dim=sentence_embedding_dim,
                                             padding_idx=self._sentence_vocab.padding_index)

        self._entity_tag_vocab = entity_tag_vocab
        self._entity_tag_embedder = Embedding(self._entity_tag_vocab.size,
                                              embedding_dim=entity_tag_embedding_dim,
                                              padding_idx=self._entity_tag_vocab.padding_index)

        self._event_type_vocab = event_type_vocab
        self._event_type_embedder_1 = Embedding(self._entity_tag_vocab.size,
                                              embedding_dim=event_type_embedding_dim)
        self._event_type_embedder_2 = Embedding(self._entity_tag_vocab.size,
                                                embedding_dim=event_type_embedding_dim)

        # lstm 作为encoder
        self._lstm = LSTM(input_size=(sentence_embedding_dim + entity_tag_embedding_dim),
                          hidden_size=lstm_hidden_size,
                          num_layers=lstm_encoder_num_layer,
                          batch_first=True,
                          dropout=lstm_encoder_droupout,
                          bidirectional=False)

    def reset_parameters(self):
        pass

    def forward(self,
                sentence: LongTensor,
                entity_tag: LongTensor,
                event_type: LongTensor,
                metadata: Dict = None) -> EventModelOutputs:
        """
        模型运行
        :param sentence: shape: (B, SeqLen), 句子的 index tensor
        :param entity_tag: shape: (B, SeqLen), 句子的 实体 index tensor
        :param event_type: shape: (B,), event type 的 tensor
        :param metadata: metadata 数据，不参与模型运算
        """

        assert sentence.dim() == 2, f"Sentence 的维度 {sentence.dim()} !=2, 应该是(B, SeqLen)"

        # sentence, entity_tag 使用的是同一个 mask
        mask = nn_util.sequence_mask(sentence,
                                     self._sentence_vocab.index(self._sentence_vocab.padding))

        # shape: B * SeqLen * sentence_embedding_dim
        sentence_embedding = self._sentence_embedder(sentence)

        # shape: B * SeqLen * entity_tag_embedding_dim
        entity_tag_embedding = self._entity_tag_embedder(entity_tag)

        # shape: B * SeqLen * InputSize, InputSize = sentence_embedding_dim + entity_tag_embedding_dim
        sentence_embedding = torch.cat((sentence_embedding, entity_tag_embedding),
                                       dim=-1)

        # 使用 lstm sequence encoder 进行 encoder
        packed_sentence_embedding = pack_padded_sequence(input=sentence_embedding,
                             batch_first=True,
                             enforce_sorted=False)
        packed_sequence, (h_n, c_n) = self._lstm(packed_sentence_embedding)

        # shape: B * SeqLen * InputSize
        sentence_encoding: Tensor = pad_packed_sequence(packed_sequence,
                                                        batch_first=True)

        # shape: B * InputSize
        event_type_embedding_1: Tensor = self._event_type_embedder_1(event_type)

        # attention
        # shape: B * InputSize * 1
        event_type_embedding_1_tmp = event_type_embedding_1.unsqueeze(-1)

        # shape: (B * SeqLen * InputSize) bmm (B * InputSize * 1) = B * SeqLen * 1
        attention_logits = sentence_encoding.bmm(event_type_embedding_1_tmp)

        # shape: B * SeqLen
        attention_logits = attention_logits.squeeze(-1)

        # Shape: B * SeqLen
        tmp_attention_logits = torch.exp(attention_logits) * mask

        # Shape: B * Seqlen
        tmp_attenttion_logits_sum = torch.sum(tmp_attention_logits, dim=-1, keepdim=True)

        # Shape: B * SeqLen
        attention = tmp_attention_logits / tmp_attenttion_logits_sum

        # Score1 计算, Shape: B * 1
        score1 = torch.sum(attention_logits * attention, dim=-1, keepdim=True)

        # global score

        # 获取最后一个hidden, shape: B * INPUT_SIZE
        hidden_last = h_n

        # event type 2, shape: B * INPUT_SIZE
        event_type_embedding_2: Tensor = self._event_type_embedder_2(event_type)

        # shape: B * INPUT_SIZE
        tmp = hidden_last * event_type_embedding_2

        # shape: B * 1
        score2 = torch.sum(tmp, dim=-1, keepdim=True)

        # 最终的score, B * 1
        score = score1 * self._alpha + score2 * (1 - self._alpha)
        if self._activate_score:  # 使用 sigmoid函数激活
            score = torch.sigmoid(score)

        return EventModelOutputs(logits=score, event_type=event_type)

