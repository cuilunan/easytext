#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 ace dataset

Authors: panxu
Date:    2020/06/09 10:12:00
"""

import os

from easytext.utils import bio

from event import ROOT_PATH
from event.event_detection_without_tirgger.tests import ASSERT
from event.event_detection_without_tirgger.data import ACEDataset


def test_ace_dataset():
    """
    测试 ace dataset
    """
    training_data_file_path = "data/event/event_detection_without_tirgger/tests/training_data_sample.txt"
    training_data_file_path = os.path.join(ROOT_PATH, training_data_file_path)
    ace_dataset = ACEDataset(dataset_file_path=training_data_file_path)

    ASSERT.assertEqual(len(ace_dataset), 3)

    expect_tokens = ["even", "as", "the", "secretary", "of", "homeland", "security", "was", "putting"]
    instance_0 = ace_dataset[0]

    instance_0_tokens = [t.text for t in instance_0["sentence"]][0:len(expect_tokens)]
    ASSERT.assertListEqual(expect_tokens, instance_0_tokens)

    expect_event_types = {"Movement:Transport"}
    instance_0_event_types = set(instance_0["event_types"])
    ASSERT.assertSetEqual(expect_event_types, instance_0_event_types)

    expect_tags = [
        {
            "text": "Secretary",
            "entity-type": "PER:Individual",
            "head": {
                "text": "Secretary",
                "start": 38,
                "end": 39
            },
            "entity_id": "CNN_CF_20030303.1900.00-E1-2",
            "start": 38,
            "end": 39
        },
        {
            "text": "the secretary of homeland security",
            "entity-type": "PER:Individual",
            "head": {
                "text": "secretary",
                "start": 3,
                "end": 4
            },
            "entity_id": "CNN_CF_20030303.1900.00-E1-188",
            "start": 2,
            "end": 7
        },
        {
            "text": "his",
            "entity-type": "PER:Individual",
            "head": {
                "text": "his",
                "start": 9,
                "end": 10
            },
            "entity_id": "CNN_CF_20030303.1900.00-E1-190",
            "start": 9,
            "end": 10
        },
        {
            "text": "Secretary Ridge",
            "entity-type": "PER:Individual",
            "head": {
                "text": "Ridge",
                "start": 39,
                "end": 40
            },
            "entity_id": "CNN_CF_20030303.1900.00-E1-198",
            "start": 38,
            "end": 40
        },
        {
            "text": "American",
            "entity-type": "GPE:Nation",
            "head": {
                "text": "American",
                "start": 29,
                "end": 30
            },
            "entity_id": "CNN_CF_20030303.1900.00-E3-196",
            "start": 29,
            "end": 30
        },
        {
            "text": "homeland security",
            "entity-type": "ORG:Government",
            "head": {
                "text": "homeland security",
                "start": 5,
                "end": 7
            },
            "entity_id": "CNN_CF_20030303.1900.00-E55-162",
            "start": 5,
            "end": 7
        },
        {
            "text": "his people",
            "entity-type": "PER:Group",
            "head": {
                "text": "people",
                "start": 10,
                "end": 11
            },
            "entity_id": "CNN_CF_20030303.1900.00-E88-171",
            "start": 9,
            "end": 11
        },
        {
            "text": "a 30-foot Cuban patrol boat with four heavily armed men",
            "entity-type": "VEH:Water",
            "head": {
                "text": "boat",
                "start": 21,
                "end": 22
            },
            "entity_id": "CNN_CF_20030303.1900.00-E96-192",
            "start": 17,
            "end": 27
        },
        {
            "text": "Cuban",
            "entity-type": "GPE:Nation",
            "head": {
                "text": "Cuban",
                "start": 19,
                "end": 20
            },
            "entity_id": "CNN_CF_20030303.1900.00-E97-193",
            "start": 19,
            "end": 20
        },
        {
            "text": "four heavily armed men",
            "entity-type": "PER:Group",
            "head": {
                "text": "men",
                "start": 26,
                "end": 27
            },
            "entity_id": "CNN_CF_20030303.1900.00-E98-194",
            "start": 23,
            "end": 27
        },
        {
            "text": "American shores",
            "entity-type": "LOC:Region-General",
            "head": {
                "text": "shores",
                "start": 30,
                "end": 31
            },
            "entity_id": "CNN_CF_20030303.1900.00-E99-195",
            "start": 29,
            "end": 31
        },
        {
            "text": "the Coast Guard",
            "entity-type": "ORG:Government",
            "head": {
                "text": "Coast Guard",
                "start": 36,
                "end": 38
            },
            "entity_id": "CNN_CF_20030303.1900.00-E102-197",
            "start": 35,
            "end": 38
        },
        {
            "text": "last month",
            "entity-type": "TIM:time",
            "head": {
                "text": "last month",
                "start": 14,
                "end": 16
            },
            "entity_id": "CNN_CF_20030303.1900.00-T4-1",
            "start": 14,
            "end": 16
        },
        {
            "text": "now",
            "entity-type": "TIM:time",
            "head": {
                "text": "now",
                "start": 40,
                "end": 41
            },
            "entity_id": "CNN_CF_20030303.1900.00-T5-1",
            "start": 40,
            "end": 41
        }
    ]

    expect_tags = {(tag["entity-type"], tag["head"]["start"], tag["head"]["end"]) for tag in expect_tags}
    instance_0_entity_tag = [t.text for t in instance_0["entity_tag"]]
    spans = bio.decode_one_sequence_label_to_span(instance_0_entity_tag)

    tags = {(span["label"], span["begin"], span["end"]) for span in spans}

    ASSERT.assertSetEqual(expect_tags, tags)

