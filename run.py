import numpy as np
import torch
from time import strftime, localtime
import argparse
from loguru import logger
from importlib import import_module
from transformers import BertTokenizer

from utils.utils import random_seed, set_logger, \
    config_to_json_string, predict_to_save, combined_result
from processors.DataProcessor import DataProcessor
from models.bert import Bert, SequenceBert
from cross_validation import cross_validation


MODEL_CLASSES = {
   'bert':  Bert,
   'sequence_bert': SequenceBert,
   'sequence_with_speaker_bert': SequenceBert
}


def Task(config):
    if config.device.type == 'cuda':
        torch.cuda.set_device(config.device_id)

    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file,
                                              do_lower_case=config.do_lower_case)
    processor = DataProcessor(
        data_dir=config.data_dir,
        do_lower_case=config.do_lower_case,
        language=config.language,
        do_preprocessing=config.do_preprocessing,
        split=config.split
    )
    config.class_list = processor.get_labels()
    config.num_labels = len(config.class_list)
    config.speaker_dict = processor.get_speaker_map(config.speaker_threshold) if config.speaker_tag else None
    train_examples = processor.get_train_examples()
    dev_examples = processor.get_dev_examples()
    test_examples = processor.get_test_examples(config.test_file)

    cur_model = MODEL_CLASSES[config.use_model]
    model = cur_model(config)

    logger.info("self config: {}\n".format(config_to_json_string(config)))

    model_example, metrics_result, predict_label = cross_validation(
        config=config,
        train_examples=train_examples,
        dev_examples=dev_examples,
        model=model,
        tokenizer=tokenizer,
        pattern=config.pattern,
        test_examples=test_examples)

    if config.pattern == 'k_fold':
        logger.info('K({})-fold models dev acc: {}'.format(config.k_fold, metrics_result[0]))
        logger.info('K({})-fold models dev f1: {}'.format(config.k_fold, metrics_result[1]))
        dev_acc = np.array(metrics_result[0]).mean()
        dev_f1 = np.array(metrics_result[1]).mean()
        predict_label = combined_result(predict_label, pattern='average')
    else:
        dev_acc = metrics_result[0]
        dev_f1 = metrics_result[1]
    logger.info("dev evaluate average Acc: {}, F1:{}".format(dev_acc, dev_f1))
    file_name = '{}_{}_{:>.6f}.csv'.format(strftime("%m%d-%H%M%S", localtime()), config.language, dev_f1)
    predict_to_save(predict_label, path=config.result_save_path,
                    file=file_name, prob_threshold=config.prob_threshold)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chinese NER Task')
    parser.add_argument('--config', type=str, required=True,
                        help='choose a config file')
    args = parser.parse_args()

    config_name = args.config
    import_config = import_module('configs.' + config_name)
    config = import_config.Config()

    random_seed(config.seed)
    set_logger(config.logging_dir, to_file=config.is_logging2file)

    Task(config)
