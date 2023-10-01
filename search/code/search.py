# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library model for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import sys
import argparse
import logging
import os
import pickle
import random
import time

import torch
import json
import numpy as np
from tqdm import tqdm

from utils import show_gpu, getAdapter, get_model_size, ADAPTER_TYPE
from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

from transformers import (AdamW, get_linear_schedule_with_warmup, BertConfig, BertModel, BertTokenizer, RobertaConfig,
                          RobertaModel, RobertaTokenizer, T5Config, T5ForConditionalGeneration, T5Tokenizer, BartConfig,
                          BartForConditionalGeneration, BartTokenizer,)

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,
                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url


def convert_examples_to_features(js, tokenizer, args):
    """convert examples to token ids"""
    if args.model_type == 'unixcoder':
        code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
        code_tokens = tokenizer.tokenize(code)[:args.code_length - 4]
        code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.code_length - len(code_ids)
        code_ids += [tokenizer.pad_token_id] * padding_length

        nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
        nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 4]
        nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args.nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id] * padding_length
    else:
        if 'code_tokens' in js:
            code = ' '.join(js['code_tokens'])
        else:
            code = ' '.join(js['function_tokens'])

        nl = ' '.join(js['docstring_tokens'])

        # code_tokens = tokenizer.tokenize(code)[:args.code_length - 2]
        # code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        # code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        # padding_length = args.code_length - len(code_ids)
        # code_ids += [tokenizer.pad_token_id] * padding_length
        #
        # nl_tokens = tokenizer.tokenize(nl)[:args.code_length - 2]
        # nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
        # nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        # padding_length = args.code_length - len(nl_ids)
        # nl_ids += [tokenizer.pad_token_id] * padding_length

        code = code.replace('</s>', '<unk>')
        code_ids = tokenizer.encode(code, max_length=args.code_length, padding='max_length',
                                    truncation=True)
        code_tokens = tokenizer.convert_ids_to_tokens(code_ids)
        nl_ids = tokenizer.encode(nl, max_length=args.nl_length, padding='max_length',
                                  truncation=True)
        nl_tokens = tokenizer.convert_ids_to_tokens(nl_ids)

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'] if "url" in js else js["retrieval_idx"])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {'code_tokens': key.split(), "retrieval_idx": js[key], 'doc': "", 'docstring_tokens': ""}
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)

        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))

        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """ Train the model """
    # get training data
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    # get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * args.num_train_epochs)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader) * args.num_train_epochs)

    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    model.train()
    tr_num, tr_loss, best_mrr = 0, 0, 0
    is_show_train_gpu = True
    for idx in range(args.num_train_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):
            # get inputs
            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)
            # get code and nl vectors
            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)

            # calculate scores and loss
            scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores * 20, torch.arange(code_inputs.size(0), device=scores.device))

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % 100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(tr_loss / tr_num, 5)))
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        checkpoint_prefix = 'checkpoint-best-mrr'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save adapter
        model_to_save = model.module if hasattr(model, 'module') else model

        if args.do_adapter:
            model_to_save.encoder.save_adapter(output_dir, args.adapter_name)
            logger.info("Saving model adapter to %s", output_dir)

        output_dir = os.path.join(output_dir, '{}'.format('pytorch_model.bin'))
        torch.save(model_to_save.state_dict(), output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)

        if is_show_train_gpu:
            show_gpu()
            is_show_train_gpu = False


def evaluate(args, model, tokenizer, file_name, eval_when_training=False):
    query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)

    code_dataset = TextDataset(tokenizer, args, args.codebase_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    code_vecs = []
    nl_vecs = []
    for batch in query_dataloader:
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
            nl_vecs.append(nl_vec.cpu().numpy())

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())
    model.train()
    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)

    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr": float(np.mean(ranks))
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--do_adapter", action='store_true', help="Whether to use adapter in model.")
    parser.add_argument('--adapter_name', type=str, default='search_adapter', help="Adapter name for each layer.")
    parser.add_argument("--adapter_type", type=str, default="houlsby",
                        choices=ADAPTER_TYPE, help="Adapter type to use.")
    parser.add_argument("--adapter_file", type=str, default=None,
                        help="Optional directory to store the pre-trained adapter.")
    parser.add_argument("--cache_path", type=str, default="",
                        help="Directory to store the cached data.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    config = config_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    # add adapter
    if args.do_adapter:
        adapter_config = getAdapter(args.adapter_type)

        if args.adapter_file:
            model.load_adapter(args.adapter_file)
        else:
            # task adapter - only add if not existing
            if args.adapter_name not in model.config.adapters:
                # add a new adapter
                model.add_adapter(args.adapter_name, config=adapter_config)
        # Enable adapter training
        model.train_adapter(args.adapter_name)
        model.set_active_adapters(args.adapter_name)

        logger.info('Used Adapter: {}'.format(args.adapter_type))

    model = Model(model, tokenizer, args)
    logger.info("Training/evaluation parameters %s", args)
    num_param = get_model_size(model)
    num_total_param = get_model_size(model, required=False)
    logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

        # Training
    if args.do_train:
        train_time = time.time()
        train(args, model, tokenizer)
        logger.info(f'training time:{time.time() - train_time:.3f}s')

    # Evaluation
    results = {}
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/pytorch_model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 3)))

    if args.do_test:

        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/pytorch_model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(torch.load(output_dir))
        model.to(args.device)

        test_time = time.time()
        result = evaluate(args, model, tokenizer, args.test_data_file)

        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 3)))

        res_fn = os.path.join(args.output_dir, 'test_results')
        with open(res_fn, 'w') as f:
            for key in sorted(result.keys()):
                f.write("  {} = {}\n".format(key, str(round(result[key], 3))))

        logger.info(f'testing time:{time.time() - test_time:.3f}s')


if __name__ == "__main__":
    main()
