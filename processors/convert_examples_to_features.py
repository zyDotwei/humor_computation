import copy
import json
import os
import numpy as np
from loguru import logger
import torch.utils.data as Data
import torch
from torch.utils.data import DataLoader


def convert_examples_to_features_base(
    examples,
    tokenizer,
    label_list,
    max_length=512,
    pad_token=0,
    pad_token_segment_id=0,
    to_pair=False,
    speaker_dict=None,
):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
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

        label = label_map[example.label]
        assert len(input_ids) == len(attention_mask)
        assert len(input_ids) == len(token_type_ids)

        if index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

        features.append([input_ids, attention_mask, token_type_ids, label])

    return features


def convert_examples_to_features_pair(        #sentence pair
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

    features = []
    for (index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(example.text_latter, example.text_former,
                                       add_special_tokens=True, max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        label = label_map[example.label]
        assert len(input_ids) == len(attention_mask)
        assert len(input_ids) == len(token_type_ids)
        speaker = example.speaker
        try:
            speaker_id = speaker_dict[speaker]
        except:  # 测试集可能有oov speaker
            speaker_id = 0

        if index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("speaker_ids: %s" % str(speaker_id))
        features.append([input_ids, attention_mask, token_type_ids, speaker_id, label])

    return features



class BuildDataSetBase(Data.Dataset):
    """
    [input_ids, attention_mask, token_type_ids, label]
    """
    def __init__(self, features, config=None):
        self.features = features

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids = np.array(feature[0])
        attention_mask = np.array(feature[1])
        token_type_ids = np.array(feature[2])
        speaker_ids = feature[3]
        labels_ids = feature[4]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "speaker_id": speaker_ids,
            "labels_ids": labels_ids,
        }

    def __len__(self):
        return len(self.features)


def BaseDataLoader(dataset, device, batch_size=1, shuffle=False):

    def _collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        labels_ids = [item['labels_ids'] for item in batch]
        speaker_ids = [item['speaker_ids'] for item in batch]

        input_ids = torch.Tensor(input_ids).clone().detach().long().to(device)
        attention_mask = torch.LongTensor(attention_mask).clone().detach().to(device)
        token_type_ids = torch.LongTensor(token_type_ids).clone().detach().to(device)
        labels_ids = torch.Tensor(labels_ids).clone().detach().long().to(device)
        speaker_ids = torch.LongTensor(speaker_ids).clone().detach().to(device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "speaker_ids": speaker_ids,
            "labels_ids": labels_ids,
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn)
