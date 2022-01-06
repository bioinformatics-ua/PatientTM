# coding=utf-8
# AUTHOR: JOÃO F SILVA
# ADAPTED FROM:
#
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import random
import argparse
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import trange, tqdm


import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
#important

from readmission.modeling_readmission import BertForSequenceClassificationOriginal
from data_processor_precomputation import convert_examples_to_features, readmissionProcessor



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    
    parser.add_argument("--readmission_mode", default = None, type=str, help="early notes or discharge summary")
    
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    # parser.add_argument("--warmup_proportion",
    #                     default=0.1,
    #                     type=float,
    #                     help="Proportion of training to perform linear learning rate warmup for. "
    #                          "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")                       
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('-feat','--additional_features',
                        default=None,
                        nargs="*",
                        type=str,
                        choices=["admittime", "daystonextadmit", "duration", "diag_icd9", "diag_ccs", "proc_icd9", "proc_ccs", "ndc"],
                        help='Additional features to use as model input. Please select one or more of the following inputs: [admittime, daystonextadmit, duration, diag_icd9, diag_ccs, proc_icd9, proc_ccs, ndc]')
    parser.add_argument('--icd9_ccs_maxlength',
                        type=int,
                        default=40,
                        help="max length for icd9 and ccs tensors")
    parser.add_argument('--ndc_maxlength',
                        type=int,
                        default=200,
                        help="max length for ndc tensors")
    parser.add_argument('--freeze_bert',
                        default=False,
                        action='store_true',
                        help="Whether to freeze parameters from BERT layers or not. When frozen, these are not updated during model training.")

    args = parser.parse_args()
    
    processors = {
        "readmission": readmissionProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # args.output_dir = os.path.join(args.output_dir,current_time+run_name)
    
    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertForSequenceClassificationOriginal.from_pretrained(args.bert_model, 1)

    model.to(device)
    model.eval()
    
    numFolds = 10
    
    with torch.no_grad():
        test_batch_size = 1
        for i in range(numFolds):
            set_type = "fold" + str(i)
            # filename = "fold" + str(i) + "_text.csv"
            filename = "fold" + str(i) + "_codeprediction_text.csv"
            filepath = os.path.join(args.data_dir, filename)
            examples = processor.get_examples(filepath, set_type)
         
            print("***** Loading Fold {} *****".format(i))
            print("  Num examples = %d", len(examples))
            features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
            hadm_ids  = [f.hadm_id for f in features]
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            tensors = [input_ids, input_mask, segment_ids, label_ids]
            data = TensorDataset(*tensors)
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=test_batch_size)
            
            rows_list = []
            hadm_id_idx = 0
            for input_id, input_mask, segment_id, label_id in tqdm(dataloader, desc="Fold "+str(i)):
                input_id = input_id.to(device)
                input_mask = input_mask.to(device)
                segment_id = segment_id.to(device)
                label_id = label_id.to(device)
                hadm_id = hadm_ids[hadm_id_idx]
                pooled_output, _, _ = model(input_id, segment_id, input_mask, label_id)
                pooled_output = pooled_output.detach().cpu().numpy()
                dictionary = {}
                dictionary.update({'HADM_ID': hadm_id, 'clinicalbert_embedding': pooled_output})
                rows_list.append(dictionary)
                hadm_id_idx+=1
                
            array = np.asarray(rows_list, dtype=object)
            # array_filepath = "../data/extended_folds/" + args.readmission_mode + "/fold" + str(i) + "_text_precomputed.npy"
            array_filepath = "../data/extended_folds/" + args.readmission_mode + "/fold" + str(i) + "_codeprediction_text_precomputed.npy"
            np.save(array_filepath, array)
        
    
main()
