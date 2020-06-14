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
import torch.nn.functional as F

from easytext.data import Vocabulary, LabelVocabulary
from easytext.model import Model
from easytext.model import ModelOutputs
from easytext.utils.nn import nn_util


class EventModelOutputs(ModelOutputs):
    """
    Event Model 的输出数据
    """

    def __init__(self, logits: torch.Tensor):
        super().__init__(logits)


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
                 event_type_vocab: LabelVocabulary,
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

        return ModelOutputs(logits=score)

        if label is not None:
            # 计算loss, 注意，这里的loss，后续 follow paper 要修改成带有 beta 的loss.
            loss_ok = self._loss(score.squeeze(-1), label.float())
            loss_mse_ok = F.mse_loss(score.squeeze(-1), label.float())

            # 下面的代码 因为维度不一致，会导致无法收敛, 这个问题需要查看
            loss_no = self._loss(score, label.float())
            loss_mse_no = F.mse_loss(score, label.float())

            loss = loss_ok

            logging.info(f"score: {score}\nlabel: {label}\n")
            logging.info(f"\nloss_ok: {loss_ok}, loss_mse_ok: {loss_mse_ok}\n"
                         f"loss_no: {loss_no}, loss_mse_no: {loss_mse_no}\n")

            output_dict["loss"] = loss

            for _, metric in self._metrics.items():
                metric(y_pred, label)

            y_pred_list = y_pred.tolist()
            label_list = label.tolist()
            event_type_list = event_type.tolist()

            logging.debug("-" * 80)
            for yy, ll, ee in zip(y_pred_list, label_list, event_type_list):
                if ll == 1:
                    ee = self.vocab.get_token_from_index(ee,
                                                         namespace=EventDetectionWithoutTriggerDatasetReader.EVENT_TYPE_NAMESPACE)
                    logging.debug(f"{ee}:[{yy},{ll}]")
            logging.debug("+" * 80)

            # 计算 f1 metric
            for _, f1_metric in self._f1_metrics.items():
                f1_mask = self.mask_for_f1(f1_metric.event_type, event_type)
                f1_metric(y_pred, label, f1_mask)

        return output_dict

    def mask_for_f1(self, target_event_type: str, event_types: LongTensor) -> LongTensor:
        """
        通过 target event type 来计算 mask
        :param target_event_type:
        :param event_types:
        :return: 某个 target event type的mask
        """
        if target_event_type == "all":
            negative_event_index = self.vocab.get_token_index(
                NEGATIVE_EVENT_TYPE,
                EventDetectionWithoutTriggerDatasetReader.EVENT_TYPE_NAMESPACE)

            negative = torch.full_like(event_types, negative_event_index)

            # 不是 negative 的保留, negative mask 成 0
            mask = torch.ne(negative, event_types)
        else:
            target_event_index = self.vocab.get_token_index(
                target_event_type,
                EventDetectionWithoutTriggerDatasetReader.EVENT_TYPE_NAMESPACE)
            target_event = torch.full_like(event_types, target_event_index)

            # target event mask, 与 target event type 一致的保留
            mask = torch.eq(target_event, event_types)
        return mask.long()

    def _f1_metric(self,
                   target_event_type: str,
                   predictions: LongTensor,
                   golden_labels: LongTensor,
                   event_types: LongTensor):
        """
        计算 target event type的f1
        :param target_event_type: 某个event type.  target_event_type="all"，说明是计算全部的f1.
        :param predictions: 预测结果
        :param golden_labels: golden labels
        :param event_types: 事件类型tensor
        :return:
        """
        mask = self.mask_for_f1(target_event_type=target_event_type,
                                event_types=event_types)

        f1_measure = self._f1_metrics[target_event_type]

        return f1_measure(predictions=predictions, gold_labels=golden_labels, mask=mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        获取metrics结果
        :param reset:
        :return:
        """

        metrics = {name: metric.get_metric(reset) for name, metric in self._metrics.items()}

        for name, f1_metric in self._f1_metrics.items():
            f1_value_dict: Dict = f1_metric.get_metric(reset)
            metrics.update(f1_value_dict)
        return metrics

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """解码"""
        return output_dict
