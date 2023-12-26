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

from __future__ import absolute_import
import os
import sys
import pickle
import time

import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn

from utils import ADAPTER_TYPE, get_adapter, show_gpu, get_model_size
from model import Seq2Seq
from tqdm import tqdm
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup, BertConfig, BertModel, BertTokenizer, RobertaConfig,
                          RobertaModel, RobertaTokenizer, T5Config, T5ForConditionalGeneration, T5Tokenizer, BartConfig,
                          BartForConditionalGeneration, BartTokenizer, )

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
}

logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        if "unixcoder" in args.model_name_or_path:
            # source
            source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 5]
            source_tokens = [tokenizer.cls_token, "<encoder-decoder>", tokenizer.sep_token,
                             "<mask0>"] + source_tokens + [
                                tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            source_mask = [1] * (len(source_tokens))
            padding_length = args.max_source_length - len(source_ids)
            source_ids += [tokenizer.pad_token_id] * padding_length
            source_mask += [0] * padding_length

            # target
            if stage == "test":
                target_tokens = tokenizer.tokenize("None")
            else:
                target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
            target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            target_mask = [1] * len(target_ids)
            padding_length = args.max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * padding_length
            target_mask += [0] * padding_length
        else:

            if args.model_type == "t5":
                # source
                source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 1]
                source_tokens = source_tokens + [tokenizer.eos_token]
                source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                source_mask = [1] * (len(source_tokens))
                padding_length = args.max_source_length - len(source_ids)
                source_ids += [tokenizer.pad_token_id] * padding_length
                source_mask += [0] * padding_length

                # target
                if stage == "test":
                    target_tokens = tokenizer.tokenize("None")
                else:
                    target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 1]
                target_tokens = target_tokens + [tokenizer.eos_token]
                target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
                target_mask = [1] * len(target_ids)
                padding_length = args.max_target_length - len(target_ids)
                target_ids += [tokenizer.pad_token_id] * padding_length
                target_mask += [0] * padding_length

            else:
                # source
                source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
                source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
                source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                source_mask = [1] * (len(source_tokens))
                padding_length = args.max_source_length - len(source_ids)
                source_ids += [tokenizer.pad_token_id] * padding_length
                source_mask += [0] * padding_length

                # target
                if stage == "test":
                    target_tokens = tokenizer.tokenize("None")
                else:
                    target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
                target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
                target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
                target_mask = [1] * len(target_ids)
                padding_length = args.max_target_length - len(target_ids)
                target_ids += [tokenizer.pad_token_id] * padding_length
                target_mask += [0] * padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_data, model, tokenizer):
    if args.local_rank == -1:
        train_sampler = SequentialSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size // args.gradient_accumulation_steps)

    num_train_optimization_steps = args.train_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    else:
        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
    # Start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Batch size = %d", args.train_batch_size * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Num epoch = %d", num_train_optimization_steps * args.train_batch_size * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1) // len(train_data))

    model.train()
    dev_dataset = {}
    nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_em, best_loss = 0, 0, 0, 0, 0, 0, 10
    bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
    train_dataloader = cycle(train_dataloader)
    eval_flag = True
    is_show_train_gpu = True
    for step in bar:
        batch = next(train_dataloader)
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids, source_mask, target_mask = batch
        if args.model_type == 'roberta':
            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                               target_ids=target_ids, target_mask=target_mask)
        else:
            outputs = model(input_ids=source_ids, attention_mask=source_mask,
                            labels=target_ids, decoder_attention_mask=target_mask)
            loss = outputs.loss

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        tr_loss += loss.item()
        train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
        if (global_step + 1) % 100 == 0:
            logger.info("  step {} loss {}".format(global_step + 1, train_loss))
        nb_tr_examples += source_ids.size(0)
        nb_tr_steps += 1
        loss.backward()

        if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            eval_flag = True

        if args.local_rank in [-1, 0] and args.evaluate_during_training and (
                (global_step + 1) % args.eval_steps == 0) and eval_flag:
            # Eval model with dev data
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            eval_flag = False

            if args.local_rank == -1:

                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_target_ids, all_source_mask, all_target_mask)
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))

                # Start Evaling model
                model.eval()
                eval_loss, batch_num = 0, 0
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, target_ids, source_mask, target_mask = batch

                    if args.model_type == 'roberta':
                        loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                           target_ids=target_ids, target_mask=target_mask)
                    else:
                        outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                        labels=target_ids, decoder_attention_mask=target_mask)
                        loss = outputs.loss

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    eval_loss += loss.item()
                    batch_num += 1

                # Print loss of dev data
                model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5),
                          'eval_loss': round(eval_loss, 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                if eval_loss < best_loss:
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    logger.info("  " + "*" * 20)
                    best_loss = eval_loss

                    # Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                pred_ids = []
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                    batch = tuple(t.to(args.device) for t in batch)
                    source_ids, source_mask = batch
                    with torch.no_grad():
                        if args.model_type == 'roberta':
                            preds = model(source_ids=source_ids, source_mask=source_mask)

                            top_preds = [pred[0].cpu().numpy() for pred in preds]

                        else:
                            if hasattr(model, 'module'):
                                preds = model.module.generate(source_ids,
                                                              attention_mask=source_mask,
                                                              use_cache=True,
                                                              num_beams=args.beam_size,
                                                              max_length=args.max_target_length)
                            else:
                                preds = model.generate(source_ids,
                                                       attention_mask=source_mask,
                                                       use_cache=True,
                                                       num_beams=args.beam_size,
                                                       max_length=args.max_target_length)
                            top_preds = list(preds.cpu().numpy())
                        pred_ids.extend(top_preds)
                p = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in
                     pred_ids]
                model.train()
                predictions = []
                accs = []

                with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
                        os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
                    # , open(os.path.join(args.output_dir, "dev.origin_output"), 'w') as f2:
                    for ref, gold in zip(p, eval_examples):
                        # t5 special
                        if args.model_type == "t5":
                            ref = ref.replace("{ ", "{").replace(" {", "{").replace("} ", "}").replace(" }", "}")
                            gold.target = gold.target.replace("{ ", "{") \
                                .replace(" {", "{").replace("} ", "}").replace(" }", "}")
                        predictions.append(str(gold.idx) + '\t' + ref)
                        f.write(ref + '\n')
                        f1.write(gold.target + '\n')
                        accs.append(ref == gold.target)

                dev_bleu = round(
                    _bleu(os.path.join(args.output_dir, "dev.gold"), os.path.join(args.output_dir, "dev.output")),
                    2)
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                dev_em = round(np.mean(accs) * 100, 4)
                logger.info("  %s = %s " % ("xMatch", str(dev_em)))
                logger.info("  " + "*" * 20)

                if dev_em > best_em:
                    logger.info("  Best em:%s", dev_em)
                    logger.info("  " + "*" * 20)
                    best_em = dev_em
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-em')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Only save the model it-self

                    if args.do_adapter:
                        if args.model_type == "roberta":
                            model_to_save.encoder.save_adapter(output_dir, args.adapter_name)
                        else:
                            model_to_save.save_adapter(output_dir, args.adapter_name)
                        logger.info("Saving model adapter to %s", output_dir)

                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.local_rank == 0:
                result = {'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                if train_loss < best_loss:
                    logger.info("  Best ppl:%s", round(np.exp(train_loss), 5))
                    logger.info("  Best loss:%s", train_loss)
                    logger.info("  " + "*" * 20)
                    best_loss = train_loss

                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-em')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Only save the model it-self

                    if args.do_adapter:
                        model_to_save.save_adapter(output_dir, args.adapter_name)
                        logger.info("Saving model adapter to %s", output_dir)

                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Saving model checkpoint to %s", output_dir)

        if is_show_train_gpu and step > int(args.train_steps / 3):
            show_gpu()
            is_show_train_gpu = False


def test(args, model, tokenizer):
    logger.info("Test file: {}".format(args.test_filename))
    eval_examples = read_examples(args.test_filename)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_source_ids, all_source_mask)

    # Calculate bleu
    eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    pred_ids = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, source_mask = batch
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]

            else:
                if hasattr(model, 'module'):
                    preds = model.module.generate(source_ids,
                                                  attention_mask=source_mask,
                                                  use_cache=True,
                                                  num_beams=args.beam_size,
                                                  max_length=args.max_target_length)
                else:
                    preds = model.generate(source_ids,
                                           attention_mask=source_mask,
                                           use_cache=True,
                                           num_beams=args.beam_size,
                                           max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
    p = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in pred_ids]
    predictions = []
    accs = []
    with open(os.path.join(args.output_dir, "test.output"), 'w') as f, open(
            os.path.join(args.output_dir, "test.gold"), 'w') as f1:
        for ref, gold in zip(p, eval_examples):
            if args.model_type == "t5":
                ref = ref.replace("{ ", "{").replace(" {", "{").replace("} ", "}").replace(" }", "}")
                gold.target = gold.target.replace("{ ", "{").replace(" {", "{"). \
                    replace("} ", "}").replace(" }", "}")
            predictions.append(str(gold.idx) + '\t' + ref)
            f.write(ref + '\n')
            f1.write(gold.target + '\n')
            accs.append(ref == gold.target)

    dev_bleu = round(
        _bleu(os.path.join(args.output_dir, "test.gold"), os.path.join(args.output_dir, "test.output")), 2)
    logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
    logger.info("  %s = %s " % ("xMatch", str(round(np.mean(accs) * 100, 4))))
    logger.info("  " + "*" * 20)

    res_fn = os.path.join(args.output_dir, 'test_results')
    with open(res_fn, 'w') as f:
        f.write("  {} = {}\n".format("bleu-4", str(dev_bleu)))
        f.write("  {} = {}\n".format("xMatch", str(round(np.mean(accs) * 100, 4))))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--tokenizer_name", default="",
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filenames (source and target files).")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. (source and target files).")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. (source and target files).")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1),
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--do_adapter", action='store_true', help="Whether to use adapter in model.")
    parser.add_argument('--adapter_name', type=str, default='translate_adapter', help="Adapter name for each layer.")
    parser.add_argument("--adapter_type", type=str, default="houlsby",
                        choices=ADAPTER_TYPE, help="Adapter type to use.")
    parser.add_argument("--adapter_file", type=str, default=None,
                        help="Optional directory to store the pre-trained adapter.")

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = torch.cuda.device_count()
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    # build model
    if args.model_type == "roberta":
        encoder = model_class.from_pretrained(args.model_name_or_path, config=config)

        # add adapter
        if args.do_adapter:
            adapter_config = get_adapter(args.adapter_type)

            if args.adapter_file:
                encoder.load_adapter(args.adapter_file)
            else:
                # task adapter - only add if not existing
                if args.adapter_name not in encoder.config.adapters:
                    # add a new adapter
                    encoder.add_adapter(args.adapter_name, config=adapter_config)
            # Enable adapter training
            encoder.train_adapter(args.adapter_name)
            encoder.set_active_adapters(args.adapter_name)

            logger.info('Used Adapter: {}'.format(args.adapter_type))

        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                        beam_size=args.beam_size, max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    else:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        # add {} to t5
        if args.model_type == "t5":
            tokenizer.add_tokens(["{", "}"])
            model.resize_token_embeddings(len(tokenizer))

        # add adapter
        if args.do_adapter:
            adapter_config = get_adapter(args.adapter_type)

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

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab
    logger.info("Training/evaluation parameters %s", args)
    num_param = get_model_size(model)
    num_total_param = get_model_size(model, required=False)
    logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

    if args.do_train:

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the data, and the test will use the cache

        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids, all_target_ids, all_source_mask, all_target_mask)

        if args.local_rank == 0:
            torch.distributed.barrier()

        train_time = time.time()
        train(args, train_data, model, tokenizer)
        logger.info(f'training time:{time.time() - train_time:.3f}s')

    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-em/pytorch_model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))

        model.to(args.device)
        test_time = time.time()
        test(args, model, tokenizer)
        logger.info(f'testing time:{time.time() - test_time:.3f}s')


if __name__ == "__main__":
    main()
