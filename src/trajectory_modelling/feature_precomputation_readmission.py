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

from modeling_readmission import BertForSequenceClassification
from data_processor_precomputation_readmission import convert_examples_to_features, readmissionProcessorNoText



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
                        required=True,
                        type=str,
                        choices=["readmission", "diagnosis_prediction"],
                        help="The name of the task to run. Please select one of the following predictive tasks: [readmission, diagnosis_prediction].")    
    parser.add_argument("--codes_to_predict",
                        default=None,
                        required=False,
                        type=str,
                        choices=["small_diag_icd9", "diag_ccs"],
                        help="The type of code to predict in code predicting tasks. Please select one of the following predictive tasks: [small_diag_icd9, diag_ccs].")    
    
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
                        help="Whether to run model training.")
    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run a testing phase on the test set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--test_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for test.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
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
    parser.add_argument('-feat','--features',
                        default=None,
                        nargs="*",
                        type=str,
                        choices=["clinical_text", "admittime", "daystonextadmit", "daystoprevadmit", "duration", "diag_ccs", "proc_ccs", "small_diag_icd9",
                                "small_proc_icd9", "cui"], #"proc_icd9", "diag_icd9", "ndc"],
                        help='Features to use as model input. Please select one or more of the following inputs: [clinical_text, admittime, daystonextadmit, daystoprevadmit, duration, diag_ccs, proc_ccs, small_diag_icd9, small_proc_icd9, cui]')
    parser.add_argument('--small_icd9_ccs_maxlength',
                        type=int,
                        default=39,
                        help="max length for icd9 and ccs tensors")
    parser.add_argument('--cui_maxlength',
                        type=int,
                        default=47,
                        help="max length for cui tensors")
    parser.add_argument('--freeze_bert',
                        default=False,
                        action='store_true',
                        help="Whether to freeze parameters from BERT layers or not. When frozen, these are not updated during model training.")
    parser.add_argument('--early_stop',
                        default=False,
                        action='store_true',
                        help="Save a model checkpoint using early stopping to prevent the saving of overfiting models.")
    parser.add_argument('--subsampling',
                        default=False,
                        action='store_true',
                        help="Subsample the training datasets to equalize the distribution of positive vs negative classes. Useful for readmission prediction.")

    args = parser.parse_args()
    
    processors = {
        "readmission": readmissionProcessorNoText
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
    maxLenDict={"small_icd9_ccs_maxlen": args.small_icd9_ccs_maxlength, "cui_maxlen": args.cui_maxlength, }

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    numLabels = 1
    numFolds = 10
    
    for i in range(numFolds):
        modelToLoad = args.bert_model + "fold" + str(i) + "/"    
        model = BertForSequenceClassification.from_pretrained(modelToLoad, numLabels, args.features)

        model.to(device)
        model.eval()

        with torch.no_grad():
            test_batch_size = 1
            filename = "fold" + str(i) + "_allvisits_readmission_notext.csv"
            filepath = os.path.join(args.data_dir, filename)
            examples = processor.get_examples(filepath, args.features, fold=i)
            # for example in examples:
            #     print(example.daystoprevadmit)

            print("***** Loading Fold {} *****".format(i))
            print("  Num examples = %d", len(examples))
            features = convert_examples_to_features(examples, label_list, args.features, maxLenDict)
            hadm_ids  = [f.hadm_id for f in features]

            featurePositionDict = {}
            positionIdx=0
            tensors = []

            if "daystoprevadmit" in args.features:
                days = [f.daystoprevadmit for f in features]
                tensors.append(torch.tensor([f.daystoprevadmit for f in features], dtype=torch.float))
                featurePositionDict["daystoprevadmit"] = positionIdx
                positionIdx+=1            
            if "duration"  in args.features:
                tensors.append(torch.tensor([f.duration for f in features], dtype=torch.float))
                featurePositionDict["duration"] = positionIdx
                positionIdx+=1

            data = TensorDataset(*tensors)
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=test_batch_size)

            rows_list = []
            hadm_id_idx = 0
            for batch in tqdm(dataloader, desc="Fold "+str(i)):
                hadm_id = hadm_ids[hadm_id_idx]
                features = [feature.to(device) for feature in batch]
                featureRepresentation = model(features_name=args.features, features_tensors=features, feature_position_dict=featurePositionDict)
                featureRepresentation = featureRepresentation.detach().cpu().numpy()

                dictionary = {}
                dictionary.update({'HADM_ID': hadm_id, 'featureRepresentation': featureRepresentation})
                rows_list.append(dictionary)
                hadm_id_idx+=1

            array = np.asarray(rows_list, dtype=object)
            # array_filepath = "../data/extended_folds/" + args.readmission_mode + "/fold" + str(i) + "_text_precomputed.npy"
            array_filepath = "../../data/extended_folds/discharge_subjectsplit_trajectory" + "/fold" + str(i) + "_allvisits_readmission_precomputed.npy"
            np.save(array_filepath, array)

    
main()

                