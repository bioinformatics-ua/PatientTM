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

from modeling_readmission import BertForSequenceClassificationOriginal
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
    with torch.no_grad():
    
        test_batch_size = 1
        
        
        test_examples = processor.get_examples(os.path.join(args.data_dir, "test.csv"), "test")
        print("***** Loading testing *****")
        print("  Num examples = %d", len(test_examples))
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
        test_hadm_ids  = [f.hadm_id for f in test_features]
        test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        tensors = [test_input_ids, test_input_mask, test_segment_ids, test_label_ids]
        test_data = TensorDataset(*tensors)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

        rows_list = []
        hadm_id_idx = 0
        for input_id, input_mask, segment_id, label_id in tqdm(test_dataloader, desc="Test Step"):
            input_id = input_id.to(device)
            input_mask = input_mask.to(device)
            segment_id = segment_id.to(device)
            label_id = label_id.to(device)

            hadm_id = test_hadm_ids[hadm_id_idx]
            pooled_output, _, _ = model(input_id, segment_id, input_mask, label_id)
            pooled_output = pooled_output.detach().cpu().numpy()
            dictionary = {}
            dictionary.update({'HADM_ID': hadm_id, 'clinicalbert_embedding': pooled_output})
            rows_list.append(dictionary)
            hadm_id_idx+=1
            
        test_array = np.asarray(rows_list, dtype=object)
        np.save("../data/extended/discharge/test_precomputed_text.npy", test_array)
        
        
        
        
        val_examples = processor.get_examples(os.path.join(args.data_dir, "val.csv"), "val")
        print("***** Loading validation *****")
        print("  Num examples = %d", len(val_examples))
        val_features   = convert_examples_to_features(val_examples, label_list, args.max_seq_length, tokenizer)
        val_all_hadm_ids    = [f.hadm_id for f in val_features]
        val_all_input_ids   = torch.tensor([f.input_ids for f in val_features], dtype=torch.long)
        val_all_input_mask  = torch.tensor([f.input_mask for f in val_features], dtype=torch.long)
        val_all_segment_ids = torch.tensor([f.segment_ids for f in val_features], dtype=torch.long)
        val_all_label_ids   = torch.tensor([f.label_id for f in val_features], dtype=torch.long)
        val_tensors = [val_all_input_ids, val_all_input_mask, val_all_segment_ids, val_all_label_ids]
        val_data   = TensorDataset(*val_tensors)
        val_sampler   = SequentialSampler(val_data)
        val_dataloader   = DataLoader(val_data, sampler=val_sampler, batch_size=test_batch_size)

        rows_list = []
        hadm_id_idx = 0
        for input_id, input_mask, segment_id, label_id in tqdm(val_dataloader, desc="Val Step"):
            input_id = input_id.to(device)
            input_mask = input_mask.to(device)
            segment_id = segment_id.to(device)
            label_id = label_id.to(device)
            
            hadm_id = val_all_hadm_ids[hadm_id_idx]
            pooled_output, _, _ = model(input_id, segment_id, input_mask, label_id)
            pooled_output = pooled_output.detach().cpu().numpy()
            dictionary = {}
            dictionary.update({'HADM_ID': hadm_id, 'clinicalbert_embedding': pooled_output})
            rows_list.append(dictionary)
            hadm_id_idx+=1
            
        val_array = np.asarray(rows_list, dtype=object)
        np.save("../data/extended/discharge/val_precomputed_text.npy", val_array)


        
        
        train_examples = processor.get_examples(os.path.join(args.data_dir, "train.csv"), "train")
        print("***** Loading training *****")
        print("  Num examples = %d", len(train_examples)) 
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)  
        train_all_hadm_ids    = [f.hadm_id for f in train_features]
        train_all_input_ids   = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        train_all_input_mask  = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        train_all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        train_all_label_ids   = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_tensors = [train_all_input_ids, train_all_input_mask, train_all_segment_ids, train_all_label_ids]
        train_data = TensorDataset(*train_tensors)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=test_batch_size)

        rows_list = []
        hadm_id_idx=0
        for input_id, input_mask, segment_id, label_id in tqdm(train_dataloader, desc="Train Step"):
            input_id = input_id.to(device)
            input_mask = input_mask.to(device)
            segment_id = segment_id.to(device)
            label_id = label_id.to(device)

            hadm_id = train_all_hadm_ids[hadm_id_idx]
            pooled_output, _, _ = model(input_id, segment_id, input_mask, label_id)
            pooled_output = pooled_output.detach().cpu().numpy()
            dictionary = {}
            dictionary.update({'HADM_ID': hadm_id, 'clinicalbert_embedding': pooled_output})
            rows_list.append(dictionary)
            hadm_id_idx+=1

        train_array = np.asarray(rows_list, dtype=object)
        np.save("../data/extended/discharge/train_precomputed_text.npy", train_array)
    
        

        
        
#         train_examples = processor.get_examples(os.path.join(args.data_dir, "train.csv"), "train")
#         print("***** Loading training *****")
#         print("  Num examples = %d", len(train_examples)) 
#         train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)  
#         train_all_hadm_ids    = [f.hadm_id for f in train_features]
#         train_all_input_ids   = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
#         train_all_input_mask  = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
#         train_all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
#         train_all_label_ids   = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
#         train_tensors = [train_all_input_ids, train_all_input_mask, train_all_segment_ids, train_all_label_ids]
#         train_data = TensorDataset(*train_tensors)
#         train_sampler = SequentialSampler(train_data)
#         train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=test_batch_size)

#         rows_list = []
#         hadm_id_idx=0
#         for input_id, input_mask, segment_id, label_id in tqdm(train_dataloader, desc="Train Step"):
#             input_id = input_id.to(device)
#             input_mask = input_mask.to(device)
#             segment_id = segment_id.to(device)
#             label_id = label_id.to(device)

#             hadm_id = train_all_hadm_ids[hadm_id_idx]
#             pooled_output, _, _ = model(input_id, segment_id, input_mask, label_id)
#             dictionary = {}
#             dictionary.update({'HADM_ID': hadm_id, 'clinicalbert_embedding': pooled_output})
#             rows_list.append(dictionary)
#             hadm_id_idx+=1

#         train_df = pd.DataFrame(rows_list, columns=['HADM_ID', 'clinicalbert_embedding'])
#         train_df.to_csv('../data/extended/discharge/train_precomputed_text.csv', columns=['HADM_ID', 'clinicalbert_embedding'])


#         val_examples = processor.get_examples(os.path.join(args.data_dir, "val.csv"), "val")
#         print("***** Loading validation *****")
#         print("  Num examples = %d", len(val_examples))
#         val_features   = convert_examples_to_features(val_examples, label_list, args.max_seq_length, tokenizer)
#         val_all_hadm_ids    = [f.hadm_id for f in val_features]
#         val_all_input_ids   = torch.tensor([f.input_ids for f in val_features], dtype=torch.long)
#         val_all_input_mask  = torch.tensor([f.input_mask for f in val_features], dtype=torch.long)
#         val_all_segment_ids = torch.tensor([f.segment_ids for f in val_features], dtype=torch.long)
#         val_all_label_ids   = torch.tensor([f.label_id for f in val_features], dtype=torch.long)
#         val_tensors = [val_all_input_ids, val_all_input_mask, val_all_segment_ids, val_all_label_ids]
#         val_data   = TensorDataset(*val_tensors)
#         val_sampler   = SequentialSampler(val_data)
#         val_dataloader   = DataLoader(val_data, sampler=val_sampler, batch_size=test_batch_size)

#         rows_list = []
#         hadm_id_idx = 0
#         for input_id, input_mask, segment_id, label_id in tqdm(val_dataloader, desc="Val Step"):
#             input_id = input_id.to(device)
#             input_mask = input_mask.to(device)
#             segment_id = segment_id.to(device)
#             label_id = label_id.to(device)
            
#             hadm_id = val_all_hadm_ids[hadm_id_idx]
#             pooled_output, _, _ = model(input_id, segment_id, input_mask, label_id)
#             dictionary = {}
#             dictionary.update({'HADM_ID': hadm_id, 'clinicalbert_embedding': pooled_output})
#             rows_list.append(dictionary)
#             hadm_id_idx+=1

#         val_df = pd.DataFrame(rows_list, columns=['HADM_ID', 'clinicalbert_embedding'])
#         val_df.to_csv('../data/extended/discharge/val_precomputed_text.csv', columns=['HADM_ID', 'clinicalbert_embedding'])


#         test_examples = processor.get_examples(os.path.join(args.data_dir, "test.csv"), "test")
#         print("***** Loading testing *****")
#         print("  Num examples = %d", len(test_examples))
#         test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
#         test_hadm_ids  = [f.hadm_id for f in test_features]
#         test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
#         test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
#         test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
#         test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
#         tensors = [test_input_ids, test_input_mask, test_segment_ids, test_label_ids]
#         test_data = TensorDataset(*tensors)
#         test_sampler = SequentialSampler(test_data)
#         test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)

#         rows_list = []
#         hadm_id_idx = 0
#         for input_id, input_mask, segment_id, label_id in tqdm(test_dataloader, desc="Test Step"):
#             input_id = input_id.to(device)
#             input_mask = input_mask.to(device)
#             segment_id = segment_id.to(device)
#             label_id = label_id.to(device)

#             hadm_id = test_hadm_ids[hadm_id_idx]
#             pooled_output, _, _ = model(input_id, segment_id, input_mask, label_id)
#             dictionary = {}
#             dictionary.update({'HADM_ID': hadm_id, 'clinicalbert_embedding': pooled_output})
#             rows_list.append(dictionary)
#             hadm_id_idx+=1

#         test_df = pd.DataFrame(rows_list, columns=['HADM_ID', 'clinicalbert_embedding'])
#         test_df.to_csv('../data/extended/discharge/test_precomputed_text.csv', columns=['HADM_ID', 'clinicalbert_embedding'])

    
main()
