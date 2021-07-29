# coding=utf-8
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import trange, tqdm
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from scipy import interp

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
#from sklearn.utils.fixes import signature
from funcsigs import signature

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import nn

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
#important
from modeling_readmission import BertForSequenceClassification

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def processString(string, charsToRemove):
    for char in charsToRemove: string = string.replace(char, "")
    return string

def convertToIds(feature, idsMappingDict):
    featureIds=[]
    # for key in idsMappingDict.keys():
    #     print(key.__class__)
    # keys = sorted(idsMappingDict.keys())
    # for key in keys:
    #     if key.startswith("6410"):
    #         print(key)
        # if int(key) > int(641040025):
        #     print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    for entry in feature:
        # print(entry)
        featureIds.append(idsMappingDict[str(entry)])
    return featureIds


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, features, text_b=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            features: dictionary containing the following features: ADMITTIME,DURATION,DIAG_ICD9,DIAG_CCS,PROC_ICD9,PROC_CCS,NDC,Label,TEXT.
            TEXT contains the untokenized text of the first sequence. For single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.guid = guid
        self.text_a = features["text_a"]
        self.label = features["label"]
        self.text_b = text_b
        self.admittime = None
        self.duration = None
        self.diag_icd9 = None
        self.diag_ccs = None
        self.proc_icd9 = None
        self.proc_ccs = None
        self.ndc = None

        if features["admittime"] is not None:
            self.admittime = features["admittime"]
        if features["duration"] is not None:
            self.duration = features["duration"]
        if features["diag_icd9"] is not None:
            self.diag_icd9 = features["diag_icd9"]
        if features["diag_ccs"] is not None:
            self.diag_ccs = features["diag_ccs"]
        if features["proc_icd9"] is not None:
            self.proc_icd9 = features["proc_icd9"]
        if features["proc_ccs"] is not None:
            self.proc_ccs = features["proc_ccs"]
        if features["ndc"] is not None:
            self.ndc = features["ndc"]


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, features):
        self.input_ids = features["input_ids"]
        self.input_mask = features["input_mask"]
        self.segment_ids = features["segment_ids"]
        self.label_id = features["label_id"]

        if "admittime" in features.keys():
            self.admittime = features["admittime"]
        if "duration" in features.keys():
            self.duration = features["duration"]
        if "diag_icd9" in features.keys():
            self.diag_icd9 = features["diag_icd9"]
        if "diag_ccs" in features.keys():
            self.diag_ccs = features["diag_ccs"]
        if "proc_icd9" in features.keys():
            self.proc_icd9 = features["proc_icd9"]
        if "proc_ccs" in features.keys():
            self.proc_ccs = features["proc_ccs"]
        if "ndc" in features.keys():
            self.ndc = features["ndc"]


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set. This can vary according to the predictive goal in mind (binary, multiclass, regression)"""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        
    @classmethod
    def _read_csv(cls, input_file):
        """Reads a comma separated value file."""
        file=pd.read_csv(input_file)
        lines=zip(file.SUBJECT_ID,file.HADM_ID,file.ADMITTIME,file.DURATION,file.DIAG_ICD9,file.DIAG_CCS,file.PROC_ICD9,file.PROC_CCS,file.NDC,file.Label,file.TEXT)
        return lines

class readmissionProcessor(DataProcessor):
    def get_train_examples(self, data_dir, additional_features=None):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.csv")))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train", additional_features)
    
    def get_dev_examples(self, data_dir, additional_features=None):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "val.csv")), "val", additional_features)
    
    def get_test_examples(self, data_dir, additional_features=None):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test", additional_features)
    
    def get_labels(self):
        """This is only 0 or 1 for readmission prediction. Other predictive goals may need different labels"""
        return ["0", "1"]
    
    def _create_examples(self, lines, set_type, additionalFeatures=None):
        """Creates examples for the training and dev sets.
        @param additionalFeatures is a list with additional variables to be used"""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            features = dict()
            if additionalFeatures is not None:
                if "admittime" in additionalFeatures: features["admittime"] = line[2]
                else: features["admittime"] = None

                if "duration"  in additionalFeatures: features["duration"] = int(line[3])
                else: features["duration"] = None

                if "diag_icd9" in additionalFeatures:
                    features["diag_icd9"] = line[4]
                    if pd.isna(features["diag_icd9"]): features["diag_icd9"] = [-1]
                    else: features["diag_icd9"] = [int(x) for x in processString(line[4],charsToRemove = "[]\',").split(' ')]
                else: features["diag_icd9"] = None

                if "diag_ccs"  in additionalFeatures:
                    features["diag_ccs"] = line[5]
                    if pd.isna(features["diag_ccs"]): features["diag_ccs"] = [-1]
                    else: features["diag_ccs"] = [int(x) for x in processString(line[5],charsToRemove = "[]\' ").split(',')]
                else: features["diag_ccs"] = None

                if "proc_icd9" in additionalFeatures:
                    features["proc_icd9"] = line[6]
                    if pd.isna(features["proc_icd9"]): features["proc_icd9"] = [-1]
                    else: features["proc_icd9"] = [int(x) for x in processString(line[6],charsToRemove = "[]\',").split(' ')]
                else: features["proc_icd9"] = None

                if "proc_ccs"  in additionalFeatures:
                    features["proc_ccs"] = line[7]
                    if pd.isna(features["proc_ccs"]): features["proc_ccs"] = [-1]
                    else: features["proc_ccs"] = [int(x) for x in processString(line[7],charsToRemove = "[]\' ").split(',')]
                else: features["proc_ccs"] = None

                if "ndc" in additionalFeatures:
                    features["ndc"] = line[8]
                    if pd.isna(features["ndc"]): features["ndc"] = [-1]
                    else: features["ndc"] = [int(x) for x in processString(line[8],charsToRemove = "[]\' ").split(',')]
                else: features["ndc"] = None

            features["label"] = str(int(line[9]))
            features["text_a"] = line[10]

            examples.append(
                InputExample(guid=guid, features=features, text_b=None))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, additionalFeatures, maxLenDict):
    """Loads a data file into a list of `InputBatch`s."""

    """Load the mapping dictionaries for additional features, to be used later to convert to ids"""
    if "diag_icd9" in additionalFeatures or "proc_icd9" in additionalFeatures:
        with open("./data/extended/Icd9ToIdx.json","r") as file:
            icd9MappingDict = json.load(file)

    if "diag_ccs" in additionalFeatures or "proc_ccs" in additionalFeatures:
        with open("./data/extended/CCSToIdx.json","r") as file:
            ccsMappingDict = json.load(file)

    if "ndc" in additionalFeatures:
        with open("./data/extended/NDCToIdx.json","r") as file:
            ndcMappingDict = json.load(file)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        #print (example.label)
        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        feature = dict()
        feature["input_ids"]=input_ids
        feature["input_mask"]=input_mask
        feature["segment_ids"]=segment_ids
        feature["label_id"]=label_id
        # if additionalFeatures is not None:
        #     if "admittime" in additionalFeatures: feature["admittime"] = example.admittime
        #     if "duration"  in additionalFeatures: feature["duration"] = example.duration
        #     if "diag_icd9" in additionalFeatures: feature["diag_icd9"] = convertToIds(example.diag_icd9, icd9MappingDict)
        #     if "diag_ccs"  in additionalFeatures: feature["diag_ccs"] = convertToIds(example.diag_ccs, ccsMappingDict)
        #     if "proc_icd9" in additionalFeatures: feature["proc_icd9"] = convertToIds(example.proc_icd9, icd9MappingDict)
        #     if "proc_ccs"  in additionalFeatures: feature["proc_ccs"] = convertToIds(example.proc_ccs, ccsMappingDict)
        #     if "ndc"       in additionalFeatures: feature["ndc"] = convertToIds(example.ndc, ndcMappingDict)

        if additionalFeatures is not None:
            if "admittime" in additionalFeatures: feature["admittime"] = example.admittime
            if "duration"  in additionalFeatures: feature["duration"] = example.duration
            if "diag_icd9" in additionalFeatures:
                feature["diag_icd9"] = convertToIds(example.diag_icd9, icd9MappingDict)
                while len(feature["diag_icd9"]) < maxLenDict["icd9_ccs_maxlen"]:
                    feature["diag_icd9"].append(0)
            if "diag_ccs"  in additionalFeatures:
                feature["diag_ccs"] = convertToIds(example.diag_ccs, ccsMappingDict)
                while len(feature["diag_ccs"]) < maxLenDict["icd9_ccs_maxlen"]:
                    feature["diag_ccs"].append(0)
            if "proc_icd9" in additionalFeatures:
                feature["proc_icd9"] = convertToIds(example.proc_icd9, icd9MappingDict)
                while len(feature["proc_icd9"]) < maxLenDict["icd9_ccs_maxlen"]:
                    feature["proc_icd9"].append(0)
            if "proc_ccs"  in additionalFeatures:
                feature["proc_ccs"] = convertToIds(example.proc_ccs, ccsMappingDict)
                while len(feature["proc_ccs"]) < maxLenDict["icd9_ccs_maxlen"]:
                    feature["proc_ccs"].append(0)
            if "ndc" in additionalFeatures:
                feature["ndc"] = convertToIds(example.ndc, ndcMappingDict)
                while len(feature["ndc"]) < maxLenDict["ndc_maxlen"]:
                    feature["ndc"].append(0)

        features.append(InputFeatures(feature))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def vote_score(df, score, args):
    df['pred_score'] = score
    df_sort = df.sort_values(by=['ID'])
    #score 
    temp = (df_sort.groupby(['ID'])['pred_score'].agg(max)+df_sort.groupby(['ID'])['pred_score'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['pred_score'].agg(len)/2)
    x = df_sort.groupby(['ID'])['Label'].agg(np.min).values
    df_out = pd.DataFrame({'logits': temp.values, 'ID': x})

    fpr, tpr, thresholds = roc_curve(x, temp.values)
    auc_score = auc(fpr, tpr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Val (area = {:.3f})'.format(auc_score))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    string = 'auroc_clinicalbert_'+args.readmission_mode+'.png'
    plt.savefig(os.path.join(args.output_dir, string))

    return fpr, tpr, df_out

def pr_curve_plot(y, y_score, args):
    precision, recall, _ = precision_recall_curve(y, y_score)
    area = auc(recall,precision)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    
    plt.figure(2)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(
              area))
    
    string = 'auprc_clinicalbert_'+args.readmission_mode+'.png'

    plt.savefig(os.path.join(args.output_dir, string))


def vote_pr_curve(df, score, args):
    df['pred_score'] = score
    df_sort = df.sort_values(by=['ID'])
    #score 
    temp = (df_sort.groupby(['ID'])['pred_score'].agg(max)+df_sort.groupby(['ID'])['pred_score'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['pred_score'].agg(len)/2)
    y = df_sort.groupby(['ID'])['Label'].agg(np.min).values
    
    precision, recall, thres = precision_recall_curve(y, temp)
    pr_thres = pd.DataFrame(data =  list(zip(precision, recall, thres)), columns = ['prec','recall','thres'])
    vote_df = pd.DataFrame(data =  list(zip(temp, y)), columns = ['score','label'])
    
    pr_curve_plot(y, temp, args)
    
    temp = pr_thres[pr_thres.prec > 0.799999].reset_index()
    
    rp80 = 0
    if temp.size == 0:
        print('Test Sample too small or RP80=0')
    else:
        rp80 = temp.iloc[0].recall
        print('Recall at Precision of 80 is {}', rp80)

    return rp80


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
                        help="Number of updates steps to accumualte before performing a backward/update pass.")                       
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
                        choices=["admittime", "duration", "diag_icd9", "diag_ccs", "proc_icd9", "proc_ccs", "ndc"],
                        help='Additional features to use as model input. Please select one or more of the following inputs: [admittime, duration, diag_icd9, diag_ccs, proc_icd9, proc_ccs, ndc]')
    parser.add_argument('--icd9_ccs_maxlength',
                        type=int,
                        default=40,
                        help="max length for icd9 and ccs tensors")
    parser.add_argument('--ndc_maxlength',
                        type=int,
                        default=200,
                        help="max length for ndc tensors")


    args = parser.parse_args()

    processors = {
        "readmission": readmissionProcessor
    }

    maxLenDict={"icd9_ccs_maxlen": args.icd9_ccs_maxlength, "ndc_maxlen": args.ndc_maxlength}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, args.additional_features)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    model = BertForSequenceClassification.from_pretrained(args.bert_model, 1, args.additional_features)
    # print(list(model.named_parameters()))
# SEE THIS AND CHANGE MODEL NAMED PARAMETERS TO SEE WHAT PARAMETERS ARE APPEARING, WITHOUT THE GIANT TENSORS IN THE OUTPUT

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare optimizer
        if args.fp16:
            param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                                for n, param in model.named_parameters()]
        elif args.optimize_on_cpu:
            param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                                for n, param in model.named_parameters()]
        else:
            param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)

    global_step = 0
    train_loss=100000
    number_training_steps=1
    global_step_check=0
    train_loss_history=[]
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, args.additional_features, maxLenDict)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids   = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask  = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids   = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        tensors = [all_input_ids, all_input_mask, all_segment_ids, all_label_ids]

        featurePositionDict = {}
        positionIdx=0
        # additionalFeatureOrder = [feature for feature in args.additional_features]
        # if "admittime" in args.additional_features:
        #     tensors.append(torch.tensor([f.admittime for f in eval_features], dtype=torch.long))
        #     featurePositionDict["admittime"] = positionIdx
        #     positionIdx+=1
        if "duration"  in args.additional_features:
            tensors.append(torch.tensor([f.duration for f in train_features], dtype=torch.long))
            featurePositionDict["duration"] = positionIdx
            positionIdx+=1
        if "diag_icd9" in args.additional_features:
            tensors.append(torch.tensor([f.diag_icd9 for f in train_features], dtype=torch.long))
            featurePositionDict["diag_icd9"] = positionIdx
            positionIdx+=1
        if "diag_ccs"  in args.additional_features:
            tensors.append(torch.tensor([f.diag_ccs for f in train_features], dtype=torch.long))
            featurePositionDict["diag_ccs"] = positionIdx
            positionIdx+=1
        if "proc_icd9" in args.additional_features:
            tensors.append(torch.tensor([f.proc_icd9 for f in train_features], dtype=torch.long))
            featurePositionDict["proc_icd9"] = positionIdx
            positionIdx+=1
        if "proc_ccs"  in args.additional_features:
            tensors.append(torch.tensor([f.proc_ccs for f in train_features], dtype=torch.long))
            featurePositionDict["proc_ccs"] = positionIdx
            positionIdx+=1
        if "ndc"       in args.additional_features:
            tensors.append(torch.tensor([f.ndc for f in train_features], dtype=torch.long))
            featurePositionDict["ndc"] = positionIdx
            positionIdx+=1

        train_data = TensorDataset(*tensors)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        model.train()
        for epo in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, *extraFeatures = batch
                loss, logits = model(input_ids, segment_ids, input_mask, label_ids, args.additional_features, extraFeatures, featurePositionDict)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                train_loss_history.append(loss.item())
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1
                
                if (step+1) % 200 == 0:
                    string = 'step '+str(step+1)
                    print (string)

            train_loss=tr_loss
            global_step_check=global_step
            number_training_steps=nb_tr_steps
            
        string = './pytorch_model_new_'+args.readmission_mode+'.bin'
        torch.save(model.state_dict(), string)

        fig1 = plt.figure()
        plt.plot(train_loss_history)
        fig1.savefig('loss_history.png', dpi=fig1.dpi)
    
    m = nn.Sigmoid()
    if args.do_eval:
        eval_examples = processor.get_test_examples(args.data_dir, args.additional_features)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, args.additional_features, maxLenDict)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        tensors = [all_input_ids, all_input_mask, all_segment_ids, all_label_ids]

        featurePositionDict = {}
        positionIdx=0
        # additionalFeatureOrder = [feature for feature in args.additional_features]
        # if "admittime" in args.additional_features:
        #     tensors.append(torch.tensor([f.admittime for f in eval_features], dtype=torch.long))
        #     featurePositionDict["admittime"] = positionIdx
        #     positionIdx+=1
        if "duration"  in args.additional_features:
            tensors.append(torch.tensor([f.duration for f in eval_features], dtype=torch.long))
            featurePositionDict["duration"] = positionIdx
            positionIdx+=1
        if "diag_icd9" in args.additional_features:
            tensors.append(torch.tensor([f.diag_icd9 for f in eval_features], dtype=torch.long))
            featurePositionDict["diag_icd9"] = positionIdx
            positionIdx+=1
        if "diag_ccs"  in args.additional_features:
            tensors.append(torch.tensor([f.diag_ccs for f in eval_features], dtype=torch.long))
            featurePositionDict["diag_ccs"] = positionIdx
            positionIdx+=1
        if "proc_icd9" in args.additional_features:
            tensors.append(torch.tensor([f.proc_icd9 for f in eval_features], dtype=torch.long))
            featurePositionDict["proc_icd9"] = positionIdx
            positionIdx+=1
        if "proc_ccs"  in args.additional_features:
            tensors.append(torch.tensor([f.proc_ccs for f in eval_features], dtype=torch.long))
            featurePositionDict["proc_ccs"] = positionIdx
            positionIdx+=1
        if "ndc"       in args.additional_features:
            tensors.append(torch.tensor([f.ndc for f in eval_features], dtype=torch.long))
            featurePositionDict["ndc"] = positionIdx
            positionIdx+=1

        eval_data = TensorDataset(*tensors)

        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        true_labels=[]
        pred_labels=[]
        logits_history=[]

        for input_ids, input_mask, segment_ids, label_ids, *extraFeatures in tqdm(eval_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            extraFeatures = [feature.to(device) for feature in extraFeatures]

            with torch.no_grad():
                tmp_eval_loss, temp_logits = model(input_ids, segment_ids, input_mask, label_ids, args.additional_features, extraFeatures, featurePositionDict)
                logits = model(input_ids,segment_ids,input_mask, additional_features_name=args.additional_features, additional_features_tensors=extraFeatures, feature_position_dict=featurePositionDict)
            
            logits = torch.squeeze(m(logits)).detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            outputs = np.asarray([1 if i else 0 for i in (logits.flatten()>=0.5)])
            tmp_eval_accuracy=np.sum(outputs == label_ids)
            
            true_labels = true_labels + label_ids.flatten().tolist()
            pred_labels = pred_labels + outputs.flatten().tolist()
            logits_history = logits_history + logits.flatten().tolist()
       
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
            
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        df = pd.DataFrame({'logits':logits_history, 'pred_label': pred_labels, 'label':true_labels})
        
        string = 'logits_clinicalbert_'+args.readmission_mode+'_chunks.csv'
        df.to_csv(os.path.join(args.output_dir, string))
        
        df_test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

        fpr, tpr, df_out = vote_score(df_test, logits_history, args)
        
        string = 'logits_clinicalbert_'+args.readmission_mode+'_readmissions.csv'
        df_out.to_csv(os.path.join(args.output_dir,string))
        
        rp80 = vote_pr_curve(df_test, logits_history, args)
        
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,                 
                  'global_step': global_step_check,
                  'training loss': train_loss/number_training_steps,
                  'RP80': rp80}
        
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
                
                      
        
if __name__ == "__main__":
    main()
