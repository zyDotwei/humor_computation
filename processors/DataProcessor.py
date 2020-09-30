import re
import os
import csv
import numpy as np
import copy
import json
from collections import Counter
from loguru import logger


class InputExample(object):
    """
    A single training/test example.
    """

    def __init__(self, guid, dialogue_id, utterance_id, speaker, text, label):
        self.guid = guid
        self.dialogue_id = dialogue_id
        self.utterance_id = utterance_id
        self.speaker = speaker
        self.text = text
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputPairExample(object):
    """
    sentence pair training/test example.
    """

    def __init__(self, guid, dialogue_id, utterance_id, speaker, text_former, text_latter, label):
        self.guid = guid
        self.dialogue_id = dialogue_id
        self.utterance_id = utterance_id
        self.speaker = speaker
        self.text_former = text_former
        self.text_latter = text_latter
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"




class DataProcessor:

    def __init__(self,
                 data_dir,
                 do_lower_case=True,
                 language='cn',
                 do_preprocessing=False,
                 split=(0.8, 0.2),
                 sentence_pair_tag=False,
                 combined_sentence_num=1
                 ):
        """
        :param data_dir:
        :param do_lower_case:
        :param language: "cn" or "en"
        :param do_preprocessing:
        """
        self.data_dir = data_dir
        self.do_lower_case = do_lower_case
        self.language = language
        self.do_preprocessing = do_preprocessing
        self.pair_tag = sentence_pair_tag
        self.combine_n = combined_sentence_num
        self.train_examples, self.dev_examples = self._split_train_dev(split)

    def _split_train_dev(self, split):
        train_examples, dev_examples = [], []
        diag_examples_dict = {}
        train_data = self._read_csv(os.path.join(self.data_dir, self.language + "_train.csv"), True)
        for example in train_data:
            if example[0] not in diag_examples_dict:
                diag_examples_dict[example[0]] = [example]
            else:
                diag_examples_dict[example[0]].append(example)
        diag_idx = list(diag_examples_dict.keys())
        np.random.shuffle(diag_idx)
        lens = len(diag_idx)
        split_point = int(split[0] * lens) + 1
        train_idx = diag_idx[:split_point]
        dev_idx = diag_idx[split_point:]
        for idx in train_idx:
            train_examples.extend(diag_examples_dict[idx])
        for idx in dev_idx:
            dev_examples.extend(diag_examples_dict[idx])

        return train_examples, dev_examples

    def get_train_examples(self):
        return self._create_examples(self.train_examples, 'train')

    def get_dev_examples(self):
        return self._create_examples(self.dev_examples, 'dev')

    def get_test_examples(self, file_name='dev.csv'):
        return self._create_examples(
            self._read_csv(os.path.join(self.data_dir, self.language + '_' + file_name), False), 'test'
        )

    def get_labels(self):
        return ['0', '1']

    def get_speaker_map(self, threshold):
        speaker_map = {}
        train_examples = self._read_csv(os.path.join(self.data_dir, self.language + '_' + 'train.csv'), False)
        train_speaker = [example[2] for example in train_examples]
        speaker_counter = dict(Counter(train_speaker))
        index_ = 1
        for key_, value_ in speaker_counter.items():
            if value_ > threshold:
                speaker_map[key_] = index_
                index_ += 1
        logger.info("{} speakers has been differently mapped, total:{}".format(index_, len(speaker_counter)))
        return speaker_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        if not self.pair_tag:  #single sentence
            examples = []
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i+1)
                dialogue_id = line[0]
                utterance_id = line[1]
                speaker = line[2]
                text = line[3]
                label = line[4]
                examples.append(InputExample(guid=guid, dialogue_id=dialogue_id, utterance_id=utterance_id,
                                             speaker=speaker, text=text, label=label))
        else:  # pair sentence
            examples = []
            former_dialogue_id = -1
            if self.language == 'en':
                basic_former_text = ['dialogue begains!']
            else:
                basic_former_text = ['对话开始了']
            former_text_list = basic_former_text
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i + 1)
                dialogue_id = line[0]
                utterance_id = line[1]
                speaker = line[2]
                text = line[3]
                label = line[4]
                if dialogue_id == former_dialogue_id:  # 判断句子是否为相同dialogue
                    examples.append(InputPairExample(guid=guid, dialogue_id=dialogue_id, utterance_id=utterance_id,
                                                     speaker=speaker, text_former=former_text, text_latter=text,
                                                     label=label))
                    if len(former_text_list) > self.combine_n - 1:
                        former_text_list.pop(0)
                        former_text_list.append(text)
                    else:
                        former_text_list.append(text)
                    former_text = ''.join(former_text_list)
                    former_dialogue_id = dialogue_id
                else:
                    former_text_list = basic_former_text
                    former_text = former_text_list[0]
                    examples.append(InputPairExample(guid=guid, dialogue_id=dialogue_id, utterance_id=utterance_id,
                                                     speaker=speaker, text_former=former_text, text_latter=text,
                                                     label=label))
                    former_text_list.append(text)
                    former_dialogue_id = dialogue_id
                    former_text = ''.join(former_text_list)
        return examples

    def _read_csv(self, input_file, have_label=True):
        """
        :param input_file:
        :return: list [Dialogue_id, Utterance_id, Speaker, Sentence, label]
        """
        data_list = []
        sentences_len = {}
        with open(input_file, "r", encoding="utf-8") as f:
            tsv_list = list(csv.reader(f, delimiter=','))
            for line in tsv_list[1:]:
                dialogue_id = line[1].strip()
                utterance_id = line[2].strip()
                speaker = line[3].strip()
                sentence = line[4].strip()
                if not have_label:
                    label = '0'
                else:
                    label = line[5].strip()

                if self.do_lower_case:
                    speaker = speaker.lower()
                    sentence = sentence.lower()
                if self.do_preprocessing:
                    sentence = self._preprocessing(sentence)
                _lens = len(sentence)
                if _lens not in sentences_len:
                    sentences_len[_lens] = 1
                else:
                    sentences_len[_lens] += 1
                data_list.append([dialogue_id, utterance_id, speaker, sentence, label])
        sentences_len = sorted(sentences_len.items(), key=lambda x: x[1], reverse = True)
        logger.info('{}, unique:{}, data len:{}'.format(input_file, len(sentences_len), list(sentences_len)[:100]))
        return data_list

    def _preprocessing(self, text):
        if self.language == 'en':
            text = re.sub('\u0092', '\'', text)
            text = re.sub('[^ 0-9a-zA-Z;,.!?\']', '', text)
            text = re.sub(' {2,}', ' ', text)
        else:
            if '##' in text:
                text = re.sub('##', '，', text)
            if '#' in text:
                text = re.sub('#', '', text)
            text = re.sub(' {2,}', ' ', text)
            # if re.search('-{3,}', text):
            #     text = re.sub('-{3,}', '-', text)
            # elif re.search('-', text):
            #     text = re.sub('-', '', text)
            # text = re.sub('-', '', text)
        return text


if __name__ == "__main__":
    np.random.seed(369)
    processor = DataProcessor(
        data_dir='../train_dev_data/',
        do_lower_case=True,
        language='cn',
        do_preprocessing=True,
        sentence_pair_tag=True,
        combined_sentence_num=2
    )
    #print(processor.get_speaker_map(100))
    train_examples = processor.get_train_examples()
    dev_examples = processor.get_dev_examples()
    print(len(train_examples)+len(dev_examples))
    print(len(train_examples))
    print(train_examples[:3])
    print(len(dev_examples))
    print(dev_examples[:3])
