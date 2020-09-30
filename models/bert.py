import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from loss.losses import FocalLoss, GHMC
import numpy as np

def compute_loss(outputs, labels, loss_method='binary'):
    loss = 0.
    if loss_method == 'binary':
        labels = labels.unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
    elif loss_method == 'cross_entropy':
        loss = F.cross_entropy(outputs, labels)
    elif loss_method == 'focal_loss':
        loss = FocalLoss()(outputs, labels.float())
    elif loss_method == 'ghmc':
        loss = GHMC()(outputs, labels.float())
    return loss


class Bert(nn.Module):

    def __init__(self, config, num=0):
        super(Bert, self).__init__()
        self.device = config.device

        model_config = BertConfig.from_pretrained(
            config.config_file[num],
            num_labels=config.num_labels,
        )
        # 计算loss的方法
        self.loss_method = config.loss_method

        self.bert = BertModel.from_pretrained(
            config.model_name_or_path[num],
            config=model_config,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size[num]
        self.speaker_tag = config.speaker_tag
        if self.speaker_tag:
            self.speaker_embedding = nn.Embedding(len(config.speaker_dict)+1, config.speaker_embedding_dim)
            self.hidden_size = config.hidden_size[num] + config.speaker_embedding_dim
        self.pooler = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())
        if self.loss_method in ['binary', 'focal_loss', 'ghmc']:
            self.classifier = nn.Linear(self.hidden_size, 1)
        else:
            self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        
    def forward(self, intputs):
        """
        :param kwargs:
        :return:
        """
        input_ids = intputs.get("input_ids", None)
        attention_mask = intputs.get("attention_mask", None)
        token_type_ids = intputs.get("token_type_ids", None)
        speaker_ids = intputs.get("speaker_ids", None)
        labels_ids = intputs.get("labels_ids", None)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden = outputs[0]
        tag_output = last_hidden[:, 0, :]
        if self.speaker_tag:  #拼接speaker embedding
            speaker_embed = self.speaker_embedding(speaker_ids)
            pooled_input = torch.cat((speaker_embed, tag_output), dim=1)
        pooled_output = self.pooler(pooled_input)
        pooled_output = self.dropout(pooled_output)

        out = self.classifier(pooled_output)
        if labels_ids is not None:
            loss = compute_loss(out, labels_ids, loss_method=self.loss_method)
        else:
            loss = 0

        if self.loss_method in ['binary', 'focal_loss', 'ghmc']:
            out = torch.sigmoid(out).flatten()

        return out, loss


class SequenceBert(nn.Module):

    def __init__(self, config, num=0):
        super(SequenceBert, self).__init__()
        self.hidden_size = config.hidden_size[num]
        self.num_labels = config.num_labels
        # 计算loss的方法
        self.loss_method = config.loss_method
        self.speaker_tag = config.speaker_tag
        if self.speaker_tag:
            self.speaker_embedding = nn.Embedding(len(config.speaker_dict)+1, config.speaker_embedding_dim)
            torch.nn.init.uniform_(self.speaker_embedding.weight, -0.15, 0.15)
            self.hidden_size += config.speaker_embedding_dim
            self.pooler = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())

        model_config = BertConfig.from_pretrained(
            config.config_file[num],
            num_labels=config.num_labels,
        )
        self.bert = BertModel.from_pretrained(
            config.model_name_or_path[num],
            config=model_config,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.GRU(self.hidden_size, self.hidden_size//2,
                           batch_first=True, bidirectional=True)
        if self.loss_method in ['binary', 'focal_loss', 'ghmc']:
            self.classifier = nn.Linear(self.hidden_size, 1)
        else:
            self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, intputs):
        """
        :param kwargs:
        :return:
        """
        input_ids = intputs.get("input_ids", None)
        attention_mask = intputs.get("attention_mask", None)
        token_type_ids = intputs.get("token_type_ids", None)
        labels_ids = intputs.get("labels_ids", None)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if not self.speaker_tag:
            output = outputs[1]  # pooled_output
        else:
            last_hidden = outputs[0]
            output = last_hidden[:, 0, :]
            speaker_ids = intputs.get("speaker_ids", None)
            speaker_emb = self.speaker_embedding(speaker_ids)
            output = torch.cat((output, speaker_emb), dim=1)
            output = self.pooler(output)

        output = self.dropout(output)
        output = output.unsqueeze(0)  # [1, seq_nums, hidden]
        output, _ = self.lstm(output)  # [1, seq_nums, hidden]
        output = self.dropout(output)
        if self.loss_method in ['binary', 'focal_loss', 'ghmc']:
            out_logits = self.classifier(output).view(-1, 1)  # [seq_nums, 1]
        else:
            out_logits = self.classifier(output).view(-1, self.num_labels)  # [seq_nums, 2]

        if labels_ids is not None:
            loss = compute_loss(out_logits, labels_ids, loss_method=self.loss_method)
        else:
            loss = 0

        if self.loss_method in ['binary', 'focal_loss', 'ghmc']:
            out_logits = torch.sigmoid(out_logits).flatten()

        return out_logits, loss

