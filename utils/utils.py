import copy
import json
import numpy as np
import pandas as pd
import torch
import random
import os
import sys
from loguru import logger


def config_to_dict(config):

    output = copy.deepcopy(config.__dict__)
    if hasattr(config.__class__, "model_type"):
        output["model_type"] = config.__class__.model_type
    output['device'] = config.device.type
    return output


def config_to_json_string(config):
    """Serializes this instance to a JSON string."""
    return json.dumps(config_to_dict(config), indent=2, sort_keys=True) + '\n'


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(log_dir, to_file=True):
    fmt = "{time} | {level} | {message}"
    if to_file:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_name = os.path.join(log_dir, "runtime_{time}.log")
        logger.add(file_name, format=fmt, level="DEBUG", encoding='utf-8', )
    else:
        logger.remove()
        logger.add(sys.__stdout__, colorize=True, format=fmt, level="INFO")


def predict_to_save(predict_labels, path, file='predict.csv', prob_threshold=0.5):
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = os.path.join(path, file)
    predict_labels = np.array(predict_labels)
    predic = list(np.array(predict_labels >= prob_threshold, dtype='int'))
    df = pd.DataFrame({'ID': [idx for idx in range(len(predic))],
                       'Label': predic})
    df.to_csv(file_name, index=False)
    logger.info('{} 写入成功.'.format(file_name))


def combined_result(all_result, weight=None, pattern='average'):

    def average_result(all_result):  # shape:[num_model, axis]
        all_result = np.asarray(all_result, dtype=np.float)
        return np.mean(all_result, axis=0)

    def weighted_result(all_result, weight):
        all_result = np.asarray(all_result, dtype=np.float)
        return np.average(all_result, axis=0, weights=weight)

    if pattern == 'weighted':
        return weighted_result(all_result, weight)
    elif pattern == 'average':
        return average_result(all_result)
    else:
        raise ValueError("the combined type is incorrect")
