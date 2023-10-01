# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        if "bert" in args.model_type:
            self.classifier = RobertaClassificationHead(config)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, labels=None):

        if "roberta" in self.args.model_type:
            input_ids = input_ids.view(-1, self.args.block_size)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        else:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs['decoder_hidden_states'][-1]
            eos_mask = input_ids.eq(self.config.eos_token_id)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")

            outputs = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

        logits = self.classifier(outputs)
        prob = F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
