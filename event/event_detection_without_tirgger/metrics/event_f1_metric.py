#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
计算 event detection f1 metric

Authors: panxu(panxu@baidu.com)
Date:    2020/06/15 19:17:00
"""
from typing import Tuple, Dict

import torch
from torch import Tensor

from easytext.metrics import Metric, ModelMetricAdapter, ModelTargetMetric
from easytext.model import ModelOutputs


class EventF1Metric(Metric):

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: torch.LongTensor) -> Dict:
        pass

    @property
    def metric(self) -> Dict:
        pass

    def reset(self):
        pass


class EventF1MetricAdapter(ModelMetricAdapter):

    def __call__(self,
                 model_outputs: ModelOutputs,
                 golden_labels: Tensor) -> Tuple[Dict, ModelTargetMetric]:

        pass

    @property
    def metric(self) -> Tuple[Dict, ModelTargetMetric]:
        pass

    def reset(self) -> "Metric":
        pass