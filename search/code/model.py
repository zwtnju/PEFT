# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, encoder, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            attention_mask = code_inputs.ne(self.tokenizer.pad_token_id)
            if "bert" in self.args.model_type:
                outputs = self.encoder(input_ids=code_inputs, attention_mask=attention_mask)[0]
            else:
                outputs = self.encoder(input_ids=code_inputs, attention_mask=attention_mask,
                                       decoder_input_ids=code_inputs, decoder_attention_mask=attention_mask)[0]
            outputs = (outputs * attention_mask[:, :, None]).sum(1) / attention_mask.sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            attention_mask = nl_inputs.ne(self.tokenizer.pad_token_id)
            if "bert" in self.args.model_type:
                outputs = self.encoder(input_ids=nl_inputs, attention_mask=attention_mask)[0]
            else:
                outputs = self.encoder(input_ids=nl_inputs, attention_mask=attention_mask,
                                       decoder_input_ids=nl_inputs, decoder_attention_mask=attention_mask)[0]
            outputs = (outputs * attention_mask[:, :, None]).sum(1) / attention_mask.sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
