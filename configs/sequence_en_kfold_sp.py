import os
import torch


class Config:

    def __init__(self):

        _pretrain_path = './pretrain_models/bert-base-uncased'
        _config_file = 'bert_config.json'
        _model_file = 'pytorch_model.bin'
        _tokenizer_file = 'vocab.txt'
        self.config_name = os.path.split(__file__)[-1].split(".")[0]
        # data
        self.data_dir = './train_dev_data'
        self.language = 'en'
        self.split = (0.80, 0.20)
        self.test_file = 'dev.csv'
        self.to_pair = False
        # 使用的模型
        self.use_model = 'sequence_with_speaker_bert'
        self.config_file = [os.path.join(_pretrain_path, _config_file)]
        self.model_name_or_path = [os.path.join(_pretrain_path, _model_file)]
        self.tokenizer_file = os.path.join(_pretrain_path, _tokenizer_file)
        self.use_cuda = True
        self.device = torch.device('cuda' if self.use_cuda and torch.cuda.is_available() else 'cpu')
        self.device_id = 0
        self.do_lower_case = True
        self.class_list = []
        self.num_labels = 2
        self.train_num_examples = 0
        self.dev_num_examples = 0
        self.test_num_examples = 0
        self.hidden_dropout_prob = 0.1
        self.hidden_size = [768]
        # train set
        self.early_stop = False
        self.require_improvement = 800
        self.num_train_epochs = 3
        self.batch_size = 1
        self.pad_size = 80
        self.dialogue_max_len = 20
        self.learning_rate = 2e-5
        self.head_learning_rate = 1e-3
        self.weight_decay = 0.01
        self.warmup_proportion = 0.1
        self.batchs_to_out = 50
        self.k_fold = 5
        # logging
        self.is_logging2file = True
        self.logging_dir = './logging' + '/' + self.config_name
        # save
        self.load_save_model = False
        self.save_path = './model_saved' + '/' + self.config_name
        self.seed = 20
        # predict result to save
        self.result_save_path = './results' + '/' + self.config_name
        # 计算loss的方法
        self.loss_method = 'ghmc'  # [ binary, cross_entropy, focal_loss, ghmc]
        # 差分学习率
        self.diff_learning_rate = False
        # train pattern
        self.pattern = 'k_fold'
        # preprocessing
        self.do_preprocessing = True
        # prob
        self.prob_threshold = 0.40
        self.out_prob = True
        # speaker embedding
        self.speaker_tag = True
        self.speaker_threshold = 20  # 根据训练集频次选择speaker
        self.speaker_embedding_dim = 6  # embedding维度(偶数)
