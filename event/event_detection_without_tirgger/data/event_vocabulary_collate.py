#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
基于

Authors: panxu
Date:    2020/06/08 12:13:00
"""
from typing import Iterable, List

from easytext.data import Instance


class EventVocabularyCollate:
    """
    用来计算 ACE Event
    """

    def __call__(self, instances: Iterable[Instance]) -> List[List[str]]:

        return [instance["event_types"] for instance in instances]

