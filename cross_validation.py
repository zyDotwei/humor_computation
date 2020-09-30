from torch.utils.data import DataLoader
import numpy as np
import copy
import torch
from loguru import logger

from processors.convert_examples_to_features import \
    convert_examples_to_features_base, BuildDataSetBase, BaseDataLoader
from processors.convert_examples_to_features import \
    convert_examples_to_features_pair, BuildDataSetBase, BaseDataLoader
from processors.convert_examples_to_features_sequence import \
    convert_examples_to_features_sequence, BuildDataSetSequence, SequenceDataLoader
from processors.convert_examples_to_features_sequence_with_speaker import \
    convert_examples_to_features_sequence_with_speaker, BuildDataSetSequenceWithSpeaker, SequenceWithSpeakerDataLoader

from utils.train_eval import model_train, model_evaluate, model_metrics, model_save


MODEL_CLASSES = {
    'bert':  (convert_examples_to_features_base, BuildDataSetBase, BaseDataLoader),
    'bert_pair':  (convert_examples_to_features_pair, BuildDataSetBase, BaseDataLoader),
    'sequence_bert':  (convert_examples_to_features_sequence, BuildDataSetSequence, SequenceDataLoader),
    'sequence_with_speaker_bert': (convert_examples_to_features_sequence_with_speaker, BuildDataSetSequenceWithSpeaker,
                                   SequenceWithSpeakerDataLoader),
}


class KFoldDataLoader(object):

    def __init__(self, examples, k_fold=5):
        self.k_fold = k_fold
        self.cur_fold = 0

        self.examples_key = []  # 'sentences1 ...'
        self.examples_dict = {}  # 'sentences1: [[], []...]'
        self.creat_group_dict(examples)

        np.random.shuffle(self.examples_key)
        self.group_lens = len(self.examples_key)
        self.step = self.group_lens // k_fold

    def creat_group_dict(self, examples):

        for example in examples:
            exists = self.examples_dict.get(example.dialogue_id, None)
            if exists is None:
                self.examples_dict[example.dialogue_id] = [example]
            else:
                self.examples_dict[example.dialogue_id].append(example)
        self.examples_key = list(self.examples_dict.keys())

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_fold < self.k_fold:
            if self.cur_fold == 0:
                dev_key = self.examples_key[:self.step]
                train_key = self.examples_key[self.step:]
            elif self.cur_fold == self.k_fold - 1:
                train_key = self.examples_key[:self.cur_fold * self.step]
                dev_key = self.examples_key[self.cur_fold * self.step:]
            else:
                train_key = self.examples_key[:self.cur_fold * self.step] + \
                             self.examples_key[(self.cur_fold + 1) * self.step:]
                dev_key = self.examples_key[self.cur_fold * self.step:(self.cur_fold + 1) * self.step]

            train_data = []
            dev_data = []
            for train in train_key:
                train_data.extend(self.examples_dict[train])
            for dev in dev_key:
                dev_data.extend(self.examples_dict[dev])
            self.cur_fold += 1

            return train_data, dev_data
        else:
            raise StopIteration


def train_dev_test(
    config,
    model,
    tokenizer,
    train_data=None,
    dev_data=None,
    test_examples=None,
):
    dev_acc = 0.
    dev_f1 = 0.
    predict_label = []

    # 加载模型
    model_example = copy.deepcopy(model).to(config.device)
    best_model = None
    convert_to_features, build_data_set, data_loader = MODEL_CLASSES[config.use_model]
    if train_data:
        config.train_num_examples = len(train_data)
        # 特征转化
        train_features = convert_to_features(
            examples=train_data,
            tokenizer=tokenizer,
            label_list=config.class_list,
            max_length=config.pad_size,
            to_pair=config.to_pair,
            speaker_dict=config.speaker_dict
        )
        train_dataset = build_data_set(train_features, config=config)
        train_loader = data_loader(train_dataset, device=config.device, batch_size=config.batch_size, shuffle=True)

        # dev 数据加载与转换
        if dev_data is not None:
            config.dev_num_examples = len(dev_data)
            dev_features = convert_to_features(
                examples=dev_data,
                tokenizer=tokenizer,
                label_list=config.class_list,
                max_length=config.pad_size,
                to_pair=config.to_pair,
                speaker_dict=config.speaker_dict
            )
            dev_dataset = build_data_set(dev_features, config=config)
            dev_loader = data_loader(dev_dataset, device=config.device, batch_size=config.batch_size, shuffle=True)
        else:
            dev_loader = None

        best_model = model_train(config, model_example, train_loader, dev_loader)

        if dev_data is not None:
            dev_acc, dev_f1 = model_metrics(config, best_model, dev_loader)

    if test_examples is not None or dev_data is not None:
        if test_examples is None:
            test_examples = dev_data
        test_features = convert_to_features(
            examples=test_examples,
            tokenizer=tokenizer,
            label_list=config.class_list,
            max_length=config.pad_size,
            to_pair=config.to_pair,
            speaker_dict=config.speaker_dict
        )
        test_dataset = build_data_set(test_features, config=config)
        test_loader = data_loader(test_dataset, device=config.device, batch_size=config.batch_size, shuffle=False)
        predict_label = model_evaluate(config, model_example, test_loader, test=True)

    return best_model, (dev_acc, dev_f1), predict_label


def k_fold_cross_validation(
        config,
        model,
        tokenizer,
        train_examples,
        test_examples=None,
        save_model=False,
):
    """
    :param config:
    :param train_examples:
    :param model:
    :param tokenizer:
    :param test_examples:
    :param save_model:
    :return: dev_evaluate : tuple (acc, f1) , acc(list), f1(list)
             k_fold_predict_label : list. if not test_examples, k-fold predict on test.
    """
    dev_acc = []
    dev_f1 = []
    test_predict_set = []
    k_fold_loader = KFoldDataLoader(train_examples, k_fold=config.k_fold)
    idx = 0
    for train_data, dev_data in k_fold_loader:
        idx += 1
        logger.info('k-fold CrossValidation: # {}'.format(idx))
        best_model, metrics_result, predict_label = train_dev_test(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            dev_data=dev_data,
            test_examples=test_examples)
        if test_examples:
            test_predict_set.append(predict_label)
        # 清理显存
        if config.device.type == 'gpu':
            torch.cuda.empty_cache()
        dev_acc.append(metrics_result[0])
        dev_f1.append(metrics_result[1])
        if save_model:
            model_save(config, best_model, config.models_name+'_'+str(idx))

    return (dev_acc, dev_f1), test_predict_set


def cross_validation(
        config,
        model,
        tokenizer,
        train_examples=None,
        dev_examples=None,
        pattern='only_train',
        test_examples=None,
):
    if pattern == 'only_train':
        model_example, metrics_result, predict_label = train_dev_test(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_data=train_examples,
            dev_data=dev_examples,
            test_examples=test_examples)
        return model_example, metrics_result, predict_label
    elif pattern == 'k_fold':
        train_examples.extend(dev_examples)
        metrics_result, test_predict_set = k_fold_cross_validation(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_examples=train_examples,
            test_examples=test_examples,
        )
        return None, metrics_result, test_predict_set
    else:
        raise ["pattern error."]




