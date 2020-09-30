import copy
import json
import math
import os
import numpy as np
from loguru import logger
import torch.utils.data as Data
import torch
from torch.utils.data import DataLoader


def convert_examples_to_features_sequence_with_speaker(
    examples,
    tokenizer,
    label_list,
    max_length=512,
    pad_token=0,
    pad_token_segment_id=0,
    to_pair=False,
    speaker_dict=None
):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = {}
    for (index, example) in enumerate(examples):
        if to_pair:
            inputs = tokenizer.encode_plus(example.speaker, example.text,
                                           add_special_tokens=True, max_length=max_length)
        else:
            inputs = tokenizer.encode_plus(example.text, add_special_tokens=True, max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        speaker_id = speaker_dict.get(example.speaker, 0)
        label = label_map[example.label]
        assert len(input_ids) == len(attention_mask)
        assert len(input_ids) == len(token_type_ids)

        if index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("speaker_id:{}".format(speaker_id))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

        if example.dialogue_id not in features:
            features[example.dialogue_id] = [[input_ids, attention_mask, token_type_ids, speaker_id, label]]
        else:
            features[example.dialogue_id].append([input_ids, attention_mask, token_type_ids, speaker_id, label])

    return features


class BuildDataSetSequenceWithSpeaker(Data.Dataset):
    """
    [input_ids, attention_mask, token_type_ids, label]
    """
    def __init__(self, features, config):
        self.dialogue_feature = []
        for dialogue in features.values():
            lens = len(dialogue)
            if lens > config.dialogue_max_len:
                slice_nums = 2
                while math.ceil(lens/slice_nums) > config.dialogue_max_len:
                    slice_nums += 1
                slice_utterance = math.ceil(lens/slice_nums)
                cursor = 0
                while cursor < lens:
                    cur_dialogue = dialogue[cursor: cursor+slice_utterance]
                    self.dialogue_feature.append(cur_dialogue)
                    cursor += slice_utterance
            else:
                self.dialogue_feature.append(dialogue)

    def __getitem__(self, index):
        dialogue = self.dialogue_feature[index]

        return {
            "dialogue": dialogue,
        }

    def __len__(self):
        return len(self.dialogue_feature)


def SequenceWithSpeakerDataLoader(dataset, device, batch_size=1, shuffle=False):

    def _collate_fn(batch):
        dialogue_list = batch[0]['dialogue']
        input_ids = [item[0] for item in dialogue_list]
        attention_mask = [item[1] for item in dialogue_list]
        token_type_ids = [item[2] for item in dialogue_list]
        speaker_ids = [item[3] for item in dialogue_list]
        labels_ids = [item[4] for item in dialogue_list]

        input_ids = torch.Tensor(input_ids).clone().detach().long().to(device)
        attention_mask = torch.LongTensor(attention_mask).clone().detach().to(device)
        token_type_ids = torch.LongTensor(token_type_ids).clone().detach().to(device)
        speaker_ids = torch.Tensor(speaker_ids).clone().detach().long().to(device)
        labels_ids = torch.Tensor(labels_ids).clone().detach().long().to(device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "speaker_ids": speaker_ids,
            "labels_ids": labels_ids,
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn)
