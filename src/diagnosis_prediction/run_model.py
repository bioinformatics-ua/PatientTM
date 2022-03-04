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
import json
import time
import shutil
import logging
import random
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import trange, tqdm
from datetime import datetime
from operator import itemgetter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RepeatedKFold, KFold

# from scipy import interp

import wandb
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from ranger21 import Ranger21 as RangerOptimizer
#important

from diagnosis_prediction.modeling_diagnosis import BertForSequenceClassification, BertForSequenceClassificationOriginal
from diagnosis_prediction.data_processor import convert_examples_to_features, processorText, processorNoText
from diagnosis_prediction.evaluation import compute_metrics

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


def get_indices_perclass_and_sample(feature_list):
    indices_0, indices_1 = [], []
    for i, feature in enumerate(feature_list):
        if feature.label_id == 1:
            indices_1.append(i)
        else:
            indices_0.append(i)
    #Sub sample negative class to equalize distribution        
    indices_0 = random.sample(indices_0, len(indices_1))
    return [*indices_0, *indices_1]


def runDiagnosisPrediction(args):
    
    try:
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
        logger = logging.getLogger(__name__)

        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S_")

        ##Initialize wandb to upload run data
        wandb.init(project="PatientTM_Admission")#,config={"epochs": 4})
        run_name = wandb.run.name
        run_dir  = os.path.split(wandb.run.dir)[0]
        
        data_type = os.path.normpath(args.data_dir).split(os.path.sep)
        data_type = "".join([data_type[-2],"/",data_type[-1]])

        # config = wandb.config
        # wandb.config.update({"lr": 0.1, "channels": 16})
        wandb.config.features = args.features
        wandb.config.dataset = data_type
        wandb.config.subsampling = args.subsampling

        if args.task_name:
            wandb.config.task_name = args.task_name
        if args.codes_to_predict:
            wandb.config.codes_to_predict = args.codes_to_predict
        if args.readmission_mode:
            wandb.config.readmission_mode = args.readmission_mode
        if args.learning_rate:
            wandb.config.learning_rate = args.learning_rate
        if args.num_train_epochs:
            wandb.config.num_train_epochs = args.num_train_epochs
        if args.train_batch_size:
            wandb.config.train_batch_size = args.train_batch_size
        if args.early_stop:
            wandb.config.early_stop = args.early_stop
        
        if "clinical_text" in args.features:
            file_ending = "_codeprediction_text.csv"
            processors = {"diagnosis_prediction": processorText}
        else:
            file_ending = "_codeprediction_notext.csv"
            processors = {"diagnosis_prediction": processorNoText}

        if args.codes_to_predict == "small_diag_icd9":
            with open("../data/extended_folds/preprocessing/smallICDdiagMapping.json", "r") as file:    
                icd9DiagCodeToIdx = json.load(file)
            numLabels = len(icd9DiagCodeToIdx)
            LabelToPredict = "LABEL_NEXT_SMALL_DIAG_ICD9"
            
        elif args.codes_to_predict == "diag_ccs":
            with open("../data/extended_folds/preprocessing/CCSdiagMapping.json", "r") as file:    
                ccsDiagCodeToIdx = json.load(file)
            numLabels = len(ccsDiagCodeToIdx)
            LabelToPredict = "LABEL_NEXT_DIAG_CCS"
                                
        maxLenDict={"small_icd9_ccs_maxlen": args.small_icd9_ccs_maxlength, "cui_maxlen": args.cui_maxlength, }

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

        args.output_dir = os.path.join(args.output_dir,current_time+run_name)

        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        os.makedirs(args.output_dir, exist_ok=True)

        task_name = args.task_name.lower()

        # Copying the config file to the output dir to save the used model configurations there
        config_file = os.path.join(args.bert_model, "bert_config.json")
        shutil.copy(config_file, args.output_dir)

        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

        processor = processors[task_name]()

        if args.do_train:

            specialVal = False
            specialTest = False

            cvFolds = 10
            folds = [i for i in range(cvFolds)]
            folds_dataset = []
            for i in folds:
                dataset = processor.get_examples(args.data_dir, LabelToPredict, features=args.features, fold=i)
                folds_dataset.append(convert_examples_to_features(dataset, args.features, maxLenDict))
            print("All dataset folds loaded.\n")

            history_training_loss, history_val_loss, history_test_loss = [], [], []
            history_val_microprecision, history_val_microrecall, history_val_microf1, history_val_microAUC, history_val_micro_rp80 = [], [], [], [], []
            history_val_microAvgPrecision, history_val_recallat10, history_val_recallat20, history_val_recallat30, history_val_macro_rp80 = [], [], [], [], []    
            history_test_microprecision, history_test_microrecall, history_test_microf1, history_test_microAUC, history_test_micro_rp80 = [], [], [], [], []
            history_test_microAvgPrecision, history_test_recallat10, history_test_recallat20, history_test_recallat30, history_test_macro_rp80 = [], [], [], [], []

    # Beginning the folding process here
            # repeatedkFold = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)

            kFold = KFold(n_splits = cvFolds)
            for i, (train_index, test_index) in enumerate(kFold.split(folds)):
                val_index = [train_index[-1]] #We select the last partition from the train set for validation
                train_index = train_index[:-1]

                train_features = [feature for fold_index in train_index for feature in folds_dataset[fold_index]]
                val_features   = [feature for fold_index in val_index   for feature in folds_dataset[fold_index]]
                test_features  = [feature for fold_index in test_index  for feature in folds_dataset[fold_index]]

        ## Special test with original test dataset
                if specialVal:
                    val_dataset = processor.get_examples("/clinicalBERT/data/extended/discharge/test.csv", features=args.features,
                                                          fold=i, numpy_dir="/clinicalBERT/data/extended/discharge/test_precomputed_text.npy")
                    val_features = convert_examples_to_features(val_dataset, label_list, args.features, maxLenDict)
                if specialTest:
                    test_dataset = processor.get_examples("/clinicalBERT/data/extended/discharge/test.csv", features=args.features,
                                                          fold=i, numpy_dir="/clinicalBERT/data/extended/discharge/test_precomputed_text.npy")
                    test_features = convert_examples_to_features(test_dataset, label_list, args.features, maxLenDict)

                train_idx = [i for i in range(len(train_features))]
                val_idx = [i for i in range(len(val_features))]
                test_idx = [i for i in range(len(test_features))]
                
                if args.subsampling:
        #   Special sampling to downsample negative class in a random manner, and have equal distribution in the dataset
                    train_idx = get_indices_perclass_and_sample(train_features)
                    val_idx   = get_indices_perclass_and_sample(val_features)
                    test_idx  = get_indices_perclass_and_sample(test_features)

                    train_features = list(itemgetter(*train_idx)(train_features))
                    val_features   = list(itemgetter(*val_idx)(val_features))
                    test_features  = list(itemgetter(*test_idx)(test_features))


                num_train_steps = int(len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
                logger.info("***** Running training in configuration %d *****", i)
                logger.info("  Num examples = %d", len(train_features))
                logger.info("  Batch size = %d", args.train_batch_size)
                logger.info("  Num steps = %d", num_train_steps)

                if "clinical_text" in args.features:
                    train_precomputed_text = torch.tensor([f.clinical_text for f in train_features], dtype=torch.float)
                    train_all_labels       = torch.tensor([f.label for f in train_features], dtype=torch.long)
                    train_tensors          = [train_precomputed_text, train_all_labels]
                    val_precomputed_text   = torch.tensor([f.clinical_text for f in val_features], dtype=torch.float)
                    val_all_labels         = torch.tensor([f.label for f in val_features], dtype=torch.long)
                    val_tensors            = [val_precomputed_text, val_all_labels]
                    test_precomputed_texts = torch.tensor([f.clinical_text for f in test_features], dtype=torch.float)
                    test_all_labels        = torch.tensor([f.label for f in test_features], dtype=torch.long)
                    test_tensors           = [test_precomputed_texts, test_all_labels]
                else:
                    train_all_labels    = torch.tensor([f.label for f in train_features], dtype=torch.long)
                    train_tensors       = [train_all_labels]
                    val_all_labels      = torch.tensor([f.label for f in val_features], dtype=torch.long)
                    val_tensors         = [val_all_labels]
                    test_labels         = torch.tensor([f.label for f in test_features], dtype=torch.long)
                    test_tensors        = [test_labels]

                featurePositionDict = {}
                positionIdx=0

                if "admittime" in args.features:
                    train_tensors.append(torch.tensor([f.admittime for f in train_features], dtype=torch.long))  #long, float, or other??!
                    val_tensors.append(torch.tensor([f.admittime for f in val_features], dtype=torch.long))       #long, float, or other??!
                    test_tensors.append(torch.tensor([f.admittime for f in test_features], dtype=torch.long)) #long, float, or other??!
                    featurePositionDict["admittime"] = positionIdx
                    positionIdx+=1
                if "daystonextadmit" in args.features:
                    train_tensors.append(torch.tensor([f.daystonextadmit for f in train_features], dtype=torch.float))
                    val_tensors.append(torch.tensor([f.daystonextadmit for f in val_features], dtype=torch.float))
                    test_tensors.append(torch.tensor([f.daystonextadmit for f in test_features], dtype=torch.float))
                    featurePositionDict["daystonextadmit"] = positionIdx
                    positionIdx+=1
                if "daystoprevadmit" in args.features:
                    train_tensors.append(torch.tensor([f.daystoprevadmit for f in train_features], dtype=torch.float))
                    val_tensors.append(torch.tensor([f.daystoprevadmit for f in val_features], dtype=torch.float))
                    test_tensors.append(torch.tensor([f.daystoprevadmit for f in test_features], dtype=torch.float))
                    featurePositionDict["daystoprevadmit"] = positionIdx
                    positionIdx+=1            
                if "duration"  in args.features:
                    train_tensors.append(torch.tensor([f.duration for f in train_features], dtype=torch.float))
                    val_tensors.append(torch.tensor([f.duration for f in val_features], dtype=torch.float))
                    test_tensors.append(torch.tensor([f.duration for f in test_features], dtype=torch.float))
                    featurePositionDict["duration"] = positionIdx
                    positionIdx+=1
                if "diag_ccs"  in args.features:
                    train_tensors.append(torch.tensor([f.diag_ccs for f in train_features], dtype=torch.long))
                    val_tensors.append(torch.tensor([f.diag_ccs for f in val_features], dtype=torch.long))
                    test_tensors.append(torch.tensor([f.diag_ccs for f in test_features], dtype=torch.long))
                    featurePositionDict["diag_ccs"] = positionIdx
                    positionIdx+=1
                if "proc_ccs"  in args.features:
                    train_tensors.append(torch.tensor([f.proc_ccs for f in train_features], dtype=torch.long))
                    val_tensors.append(torch.tensor([f.proc_ccs for f in val_features], dtype=torch.long))
                    test_tensors.append(torch.tensor([f.proc_ccs for f in test_features], dtype=torch.long))
                    featurePositionDict["proc_ccs"] = positionIdx
                    positionIdx+=1
                if "small_diag_icd9" in args.features:
                    train_tensors.append(torch.tensor([f.small_diag_icd9 for f in train_features], dtype=torch.long))
                    val_tensors.append(torch.tensor([f.small_diag_icd9 for f in val_features], dtype=torch.long))
                    test_tensors.append(torch.tensor([f.small_diag_icd9 for f in test_features], dtype=torch.long))
                    featurePositionDict["small_diag_icd9"] = positionIdx
                    positionIdx+=1
                if "small_proc_icd9" in args.features:
                    train_tensors.append(torch.tensor([f.small_proc_icd9 for f in train_features], dtype=torch.long))
                    val_tensors.append(torch.tensor([f.small_proc_icd9 for f in val_features], dtype=torch.long))
                    test_tensors.append(torch.tensor([f.small_proc_icd9 for f in test_features], dtype=torch.long))
                    featurePositionDict["small_proc_icd9"] = positionIdx
                    positionIdx+=1    
                if "cui" in args.features:
                    train_tensors.append(torch.tensor([f.cui for f in train_features], dtype=torch.long))
                    val_tensors.append(torch.tensor([f.cui for f in val_features], dtype=torch.long))
                    test_tensors.append(torch.tensor([f.cui for f in test_features], dtype=torch.long))
                    featurePositionDict["cui"] = positionIdx
                    positionIdx+=1

                train_data = TensorDataset(*train_tensors)
                val_data   = TensorDataset(*val_tensors)
                test_data  = TensorDataset(*test_tensors)


                if args.local_rank == -1:
#SAMPLING IS NOT IMPLEMENTED FOR THIS MULTICLASS MULTILABEL PROBLEM                 
#                     if not args.subsampling:
#                         # print(f'target train 0/1: {len(np.where(train_all_label_ids == 0)[0])}/{len(np.where(train_all_label_ids == 1)[0])}')
#                         class_sample_count = np.unique(train_all_label_ids, return_counts=True)[1]
#                         print(f'Target train class distribution 0/1: {class_sample_count}')
#                         weight = 1. / class_sample_count
#                         samples_weight = weight[train_all_label_ids]

#                         # new_majority_proportion = 3
#                         # class_sample_count[0] /= new_majority_proportion
#                         # weight = 1. / class_sample_count
#                         # samples_weight = weight[train_all_label_ids]

#                         samples_weight = torch.from_numpy(samples_weight)       
#                         train_sampler  = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)    
#                     elif args.subsampling:
#                         train_sampler  = RandomSampler(train_data)
                        
                    train_sampler  = RandomSampler(train_data)    
                    val_sampler    = SequentialSampler(val_data)
                    test_sampler   = SequentialSampler(test_data)
                else:
                    train_sampler = DistributedSampler(train_data)
                    val_sampler   = DistributedSampler(val_data)
                    test_sampler  = DistributedSampler(test_data)

                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
                val_dataloader   = DataLoader(val_data, sampler=val_sampler, batch_size=args.train_batch_size)
                test_dataloader  = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)


           ## Initialize the model before every fold training
                model = BertForSequenceClassification.from_pretrained(args.bert_model, numLabels, args.features)

                ## Setting WandBI to log gradients and model parameters
                wandb.watch(model)

                if args.fp16:
                    model.half()
                model.to(device)
                if args.local_rank != -1:
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                                      output_device=args.local_rank)
                elif n_gpu > 1:
                    model = torch.nn.DataParallel(model)

                train_loss, min_validation_loss = 100000, 100000
                min_val_rp80 = 0
                number_training_steps = 1
                train_loss_history, val_loss_history = [], []

                ##if freeze_bert:
                for name, param in model.named_parameters():
                    if name.startswith("bert"): # classifier.weight; classifier.bias
                        # print(name, param.type())            
                        param.requires_grad = False
                        param = param.cpu() # Force the unused parameters to the cpu, to free GPU space for other things
                        # print(name, param.type())  

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

                optimizer = RangerOptimizer(params=optimizer_grouped_parameters,
                                            lr=args.learning_rate,
                                            # warmup_pct_default=args.warmup_proportion,
                                            num_epochs=args.num_train_epochs,
                                            num_batches_per_epoch=len(train_dataloader))

                model.train()
                for epo in trange(int(args.num_train_epochs), desc="Epoch"):
                    train_loss = 0
                    nb_tr_steps = 0
                    for step, batch in enumerate(tqdm(train_dataloader, desc="Training Step")):
                        # batch = tuple(t.to(device) for t in batch)
                        #
                        if "clinical_text" in args.features:
                            precomputed_texts, labels, *nonTextFeatures = batch
                            precomputed_texts = precomputed_texts.to(device)
                            labels = labels.to(device)     
                            if nonTextFeatures:
                                nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]
                                loss, logits = model(precomputed_texts, labels, features_name=args.features, features_tensors=nonTextFeatures,
                                                     feature_position_dict=featurePositionDict) 
                            else:
                                loss, logits = model(precomputed_texts, labels, features_name=args.features)
                        else:
                            labels, *nonTextFeatures = batch
                            labels = labels.to(device) 
                            nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]
                            loss, logits = model(labels=labels, features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)


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
                        train_loss += loss.item()
                        wandb.log({"Training step loss": loss.item()})

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

                    number_training_steps=nb_tr_steps
                    wandb.log({"Training loss": train_loss/number_training_steps})

                    ## Proceeding with a validation stage on the validation dataset
                    # Setting model in evaluation mode:
                    model.eval()
                    m = nn.Sigmoid()            
                    val_loss, val_accuracy = 0, 0
                    nb_val_steps, nb_val_examples = 0, 0
                    true_labels=[]
                    pred_labels=[]
                    logits_history = []
                    for step, batch in enumerate(tqdm(val_dataloader, desc="Validation Step")):
                        ## Turning off gradient computation for safety
                        with torch.no_grad():
                            if "clinical_text" in args.features:
                                precomputed_texts, labels, *nonTextFeatures = batch
                                precomputed_texts = precomputed_texts.to(device)
                                labels = labels.to(device)    
                                if nonTextFeatures:
                                    nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]
                                    loss, logits = model(precomputed_texts, labels, features_name=args.features, features_tensors=nonTextFeatures,
                                                         feature_position_dict=featurePositionDict) 
                                else:
                                    loss, logits = model(precomputed_texts, labels, features_name=args.features)
                            else:
                                labels, *nonTextFeatures = batch
                                labels = labels.to(device) 
                                nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]
                                loss, logits = model(labels=labels, features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)

               
                        logits = m(logits).detach().cpu().numpy() #torch.squeeze(m(logits)).detach().cpu().numpy()
                        logits_history += logits.tolist()                    
                        labels = labels.to('cpu').numpy()

                        val_loss += loss.mean().item()
                        nb_val_steps += 1
                        nb_val_examples += labels.size                
                    val_loss = val_loss/nb_val_steps

                    if specialTest:
                        df_val = pd.read_csv("/clinicalBERT/data/extended/discharge/test.csv")
                    else:
                        df_val = pd.read_csv(os.path.join(args.data_dir, "fold" + str(val_index[0]) + file_ending))
                        df_val = df_val.iloc[val_idx, :]
                    df_val['pred_label_scores'] = logits_history
                    
                    val_metrics = compute_metrics(df_val, numLabels, args, label=LabelToPredict, threshold=0.5)

                    wandb.log({"Val Loss":              val_loss,
                               "Val MicroRP80":         val_metrics["micro_recall@p80"],
                               "Val MacroRP80":         val_metrics["macro_recall@p80"],
                               "Val MicroPrecision":    val_metrics["micro_precision"],
                               "Val MacroPrecision":    val_metrics["macro_precision"],
                               "Val MicroRecall":       val_metrics["micro_recall"],
                               "Val MacroRecall":       val_metrics["macro_recall"],
                               "Val MicroF1":           val_metrics["micro_f1"],
                               "Val MacroF1":           val_metrics["macro_f1"],
                               "Val MicroAUC":          val_metrics["micro_auc"],
                               "Val MacroAUC":          val_metrics["macro_auc"],
                               "Val MicroAvgPrecision": val_metrics["micro_avg_precision"],
                               "Val Recall@10":         val_metrics["recall@10"],
                               "Val Recall@20":         val_metrics["recall@20"],
                               "Val Recall@30":         val_metrics["recall@30"]})


                    ## "Early stopping" mechanism where validation loss is used to save model checkpoints
                    if args.early_stop:
                        if val_loss < min_validation_loss:
                            min_validation_loss = val_loss
                        # if val_rp80 > min_val_rp80:
                        #     min_val_rp80 = val_rp80
                            checkpoint_model_state = deepcopy(model.state_dict()) #must save a deepcopy, otherwise this is a reference to the state dict that keeps getting updated

                    ## Setting model back in training mode:
                    model.train()

                string = os.path.join(args.output_dir, 'pytorch_model_fold' + str(i) + ".bin") #'pytorch_model_new_'+args.readmission_mode+'.bin')
                if not args.early_stop: checkpoint_model_state = model.state_dict()
                torch.save(checkpoint_model_state, string)

                fig1 = plt.figure()
                plt.plot(train_loss_history)
                fig_name = os.path.join(args.output_dir, 'train_loss_history_fold' + str(i) + '.png')
                fig1.savefig(fig_name, dpi=fig1.dpi)

    #    Going to perform a test run on the test split now
                # Load the best model on train/val if early stopping was used
                if args.early_stop:
                    model.load_state_dict(checkpoint_model_state)

                m = nn.Sigmoid()
                logger.info("***** Running testing *****")
                logger.info("  Num examples = %d", len(test_features))
                logger.info("  Batch size = %d", args.test_batch_size)

                model.eval()
                test_loss, test_accuracy = 0, 0
                nb_test_steps, nb_test_examples = 0, 0
                true_labels=[]
                pred_labels=[]
                logits_history=[]

                if "clinical_text" in args.features:
                    for precomputed_texts, labels, *nonTextFeatures in tqdm(test_dataloader, desc="Test Step"):
                        precomputed_texts = precomputed_texts.to(device)
                        labels = labels.to(device)

                        with torch.no_grad():
                            if nonTextFeatures:
                                nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]
                                tmp_test_loss, logits = model(precomputed_texts, labels, features_name=args.features, features_tensors=nonTextFeatures,
                                                              feature_position_dict=featurePositionDict)
                                # logits = model(input_ids,segment_ids,input_mask, features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)
                            else:
                                tmp_test_loss, logits = model(precomputed_texts, labels, features_name=args.features)
                                # logits = model(input_ids, segment_ids, input_mask, features_name=args.features,)

                        logits = m(logits).detach().cpu().numpy() #torch.squeeze(m(logits)).detach().cpu().numpy()
                        logits_history += logits.tolist()                    
                        labels = labels.to('cpu').numpy()

                        test_loss += tmp_test_loss.mean().item()
                        nb_test_steps += 1
                        nb_test_examples += labels.size                
                    
                else:
                    for labels, *nonTextFeatures in tqdm(test_dataloader, desc="Test Step"):
                        labels = labels.to(device)
                        nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]

                        with torch.no_grad():
                            tmp_test_loss, logits = model(labels=labels, features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)
                            # logits = model(features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)       
                                                                           
                        logits = m(logits).detach().cpu().numpy() #torch.squeeze(m(logits)).detach().cpu().numpy()
                        logits_history += logits.tolist()                    
                        labels = labels.to('cpu').numpy()

                        test_loss += tmp_test_loss.mean().item()
                        nb_test_steps += 1
                        nb_test_examples += labels.size  

                test_loss = test_loss / nb_test_steps

        ## Special test with original test dataset
                if specialTest:
                    df_test = pd.read_csv("/clinicalBERT/data/extended/discharge/test.csv")
                else:
                    df_test = pd.read_csv(os.path.join(args.data_dir, "fold" + str(test_index[0]) + file_ending))
                    df_test = df_test.iloc[test_idx, :]
                df_test['pred_label_scores'] = logits_history
                
                test_metrics = compute_metrics(df_test, numLabels, args, label=LabelToPredict, threshold=0.5)

                wandb.log({"Test Loss":              test_loss,
                           "Test MicroRP80":         test_metrics["micro_recall@p80"],
                           "Test MacroRP80":         test_metrics["macro_recall@p80"],
                           "Test MicroPrecision":    test_metrics["micro_precision"],
                           "Test MacroPrecision":    test_metrics["macro_precision"],
                           "Test MicroRecall":       test_metrics["micro_recall"],
                           "Test MacroRecall":       test_metrics["macro_recall"],
                           "Test MicroF1":           test_metrics["micro_f1"],
                           "Test MacroF1":           test_metrics["macro_f1"],
                           "Test MicroAUC":          test_metrics["micro_auc"],
                           "Test MacroAUC":          test_metrics["macro_auc"],
                           "Test MicroAvgPrecision": test_metrics["micro_avg_precision"],
                           "Test Recall@10":         test_metrics["recall@10"],
                           "Test Recall@20":         test_metrics["recall@20"],
                           "Test Recall@30":         test_metrics["recall@30"]})


                result = {'Training loss':           train_loss/number_training_steps,
                          'Val Loss':                val_loss,
                          "Val MicroRP80":           val_metrics["micro_recall@p80"],
                          "Val MacroRP80":           val_metrics["macro_recall@p80"],
                          "Val MicroPrecision":      val_metrics["micro_precision"],
                          "Val MicroRecall":         val_metrics["micro_recall"],
                          "Val MicroF1":             val_metrics["micro_f1"],
                          "Val MicroAUC":            val_metrics["micro_auc"],
                          "Val MicroAvgPrecision":   val_metrics["micro_avg_precision"],
                          "Val Recall@10":           val_metrics["recall@10"],
                          "Val Recall@20":           val_metrics["recall@20"],
                          "Val Recall@30":           val_metrics["recall@30"],
                          'Test Loss':               test_loss,
                          "Test MicroRP80":          test_metrics["micro_recall@p80"],
                          "Test MacroRP80":          test_metrics["macro_recall@p80"],
                          "Test MicroPrecision":     test_metrics["micro_precision"],
                          "Test MicroRecall":        test_metrics["micro_recall"],
                          "Test MicroF1":            test_metrics["micro_f1"],
                          "Test MicroAUC":           test_metrics["micro_auc"],
                          "Test MicroAvgPrecision":  test_metrics["micro_avg_precision"],
                          "Test Recall@10":          test_metrics["recall@10"],
                          "Test Recall@20":          test_metrics["recall@20"],
                          "Test Recall@30":          test_metrics["recall@30"],
                         }
                          
                history_training_loss.append(train_loss/number_training_steps)
                history_val_loss.append(val_loss)
                history_test_loss.append(test_loss)
                history_val_micro_rp80.append(val_metrics["micro_recall@p80"])
                history_val_macro_rp80.append(val_metrics["macro_recall@p80"])
                history_val_microprecision.append(val_metrics["micro_precision"])
                history_val_microrecall.append(val_metrics["micro_recall"])
                history_val_microf1.append(val_metrics["micro_f1"])
                history_val_microAUC.append(val_metrics["micro_auc"])         
                history_val_microAvgPrecision.append(val_metrics["micro_avg_precision"])
                history_val_recallat10.append(val_metrics["recall@10"])      
                history_val_recallat20.append(val_metrics["recall@20"])     
                history_val_recallat30.append(val_metrics["recall@30"])   
                
                history_test_micro_rp80.append(test_metrics["micro_recall@p80"])
                history_test_macro_rp80.append(test_metrics["macro_recall@p80"])
                history_test_microprecision.append(test_metrics["micro_precision"])
                history_test_microrecall.append(test_metrics["micro_recall"])
                history_test_microf1.append(test_metrics["micro_f1"])
                history_test_microAUC.append(test_metrics["micro_auc"])         
                history_test_microAvgPrecision.append(test_metrics["micro_avg_precision"])
                history_test_recallat10.append(test_metrics["recall@10"])      
                history_test_recallat20.append(test_metrics["recall@20"])     
                history_test_recallat30.append(test_metrics["recall@30"])  
                

                output_results_file = os.path.join(args.output_dir, "validation_and_test_results.txt")
                if not os.path.exists(output_results_file):
                    with open(output_results_file, "w") as writer:
                        writer.write("***** Logging validation and test results for cross validation *****\n")
                else:
                    with open(output_results_file, "a") as writer:
                        logger.info("***** Results fold %d *****\n", i)
                        writer.write("***** Results fold %d *****\n" % (i))
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key]))) 
                            
                            
            mean_result = {'Training loss':       sum(history_training_loss)/len(history_training_loss),
                           "Validation loss":           sum(history_val_loss)/len(history_val_loss),
                           "Validation Micro RP80":     sum(history_val_micro_rp80)/len(history_val_micro_rp80),
                           "Validation Macro RP80":     sum(history_val_macro_rp80)/len(history_val_macro_rp80),
                           "Validation Micro F1":       sum(history_val_microf1)/len(history_val_microf1),                   
                           "Validation Micro AUC":      sum(history_val_microAUC)/len(history_val_microAUC),
                           "Validation Micro Avg Precision":       sum(history_val_microAvgPrecision)/len(history_val_microAvgPrecision),                
                           "Validation Recall@10":      sum(history_val_recallat10)/len(history_val_recallat10),
                           "Validation Recall@20":      sum(history_val_recallat20)/len(history_val_recallat20),
                           "Validation Recall@30":      sum(history_val_recallat30)/len(history_val_recallat30),
                           "Test loss":                 sum(history_test_loss)/len(history_test_loss),
                           "Test Micro RP80":           sum(history_test_micro_rp80)/len(history_test_micro_rp80),
                           "Test Macro RP80":           sum(history_test_macro_rp80)/len(history_test_macro_rp80),
                           "Test Micro F1":             sum(history_test_microf1)/len(history_test_microf1),                   
                           "Test Micro AUC":            sum(history_test_microAUC)/len(history_test_microAUC),
                           "Test Micro Avg Precision":       sum(history_test_microAvgPrecision)/len(history_test_microAvgPrecision),                
                           "Test Recall@10":            sum(history_test_recallat10)/len(history_test_recallat10),
                           "Test Recall@20":            sum(history_test_recallat20)/len(history_test_recallat20),
                           "Test Recall@30":            sum(history_test_recallat30)/len(history_test_recallat30)
                          }  
                          
            
            wandb.log({"KFold Avg Test loss":      sum(history_test_loss)/len(history_test_loss),
                       "KFold Avg Micro RP80":     sum(history_test_micro_rp80)/len(history_test_micro_rp80),
                       "KFold Avg Macro RP80":     sum(history_test_macro_rp80)/len(history_test_macro_rp80),
                       "KFold Avg Micro F1":       sum(history_test_microf1)/len(history_test_microf1),                   
                       "KFold Avg Micro AUC":      sum(history_test_microAUC)/len(history_test_microAUC),
                       "KFold Avg Micro Avg Precision":       sum(history_test_microAvgPrecision)/len(history_test_microAvgPrecision),                
                       "KFold Avg Recall@10":      sum(history_test_recallat10)/len(history_test_recallat10),
                       "KFold Avg Recall@20":      sum(history_test_recallat20)/len(history_test_recallat20),
                       "KFold Avg Recall@30":      sum(history_test_recallat30)/len(history_test_recallat30)})  
            


            print("Writing mean performances across all folds")
            with open(output_results_file, "a") as writer:
                logger.info("***** Average performance across all folds *****\n")
                writer.write("***** Average performance across all folds *****\n")
                for key in sorted(mean_result.keys()):
                    logger.info("  %s = %s", key, str(mean_result[key]))
                    writer.write("%s = %s\n" % (key, str(mean_result[key])))





#     ##
#     ## ATTENTION!!!!: The test method is working with hardcoded filepaths as it was only used to evaluate the problems with the original datasets vs folded dataset
#     ##
#         if args.do_test:

#             csv_filepath = "/clinicalBERT/data/extended/discharge/test.csv"
#             npy_precomputed_filepath = "/clinicalBERT/data/extended/discharge/test_precomputed_text.npy"

#             test_dataset = processor.get_examples(csv_filepath, features=args.features, fold=0, numpy_dir=npy_precomputed_filepath)
#             test_features = convert_examples_to_features(test_dataset, label_list, args.features, maxLenDict)

#             if "clinical_text" in args.features:
#                 test_precomputed_texts = torch.tensor([f.clinical_text for f in test_features], dtype=torch.float)
#                 test_all_label_ids     = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
#                 print(np.unique(test_all_label_ids, return_counts=True))
#                 test_tensors           = [test_precomputed_texts, test_all_label_ids]
#             else:
#                 test_label_ids      = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
#                 test_tensors        = [test_label_ids]

#             featurePositionDict = {}
#             positionIdx=0

#             if "admittime" in args.features:
#                 test_tensors.append(torch.tensor([f.admittime for f in test_features], dtype=torch.long)) #long, float, or other??!
#                 featurePositionDict["admittime"] = positionIdx
#                 positionIdx+=1
#             if "daystonextadmit" in args.features:
#                 test_tensors.append(torch.tensor([f.daystonextadmit for f in test_features], dtype=torch.float))
#                 featurePositionDict["daystonextadmit"] = positionIdx
#                 positionIdx+=1
#             if "daystoprevadmit" in args.features:
#                 test_tensors.append(torch.tensor([f.daystoprevadmit for f in test_features], dtype=torch.float))
#                 featurePositionDict["daystoprevadmit"] = positionIdx
#                 positionIdx+=1            
#             if "duration"  in args.features:
#                 test_tensors.append(torch.tensor([f.duration for f in test_features], dtype=torch.float))
#                 featurePositionDict["duration"] = positionIdx
#                 positionIdx+=1
#             if "diag_ccs"  in args.features:
#                 test_tensors.append(torch.tensor([f.diag_ccs for f in test_features], dtype=torch.long))
#                 featurePositionDict["diag_ccs"] = positionIdx
#                 positionIdx+=1
#             if "proc_ccs"  in args.features:
#                 test_tensors.append(torch.tensor([f.proc_ccs for f in test_features], dtype=torch.long))
#                 featurePositionDict["proc_ccs"] = positionIdx
#                 positionIdx+=1
#             if "small_diag_icd9" in args.features:
#                 test_tensors.append(torch.tensor([f.small_diag_icd9 for f in test_features], dtype=torch.long))
#                 featurePositionDict["small_diag_icd9"] = positionIdx
#                 positionIdx+=1
#             if "small_proc_icd9" in args.features:
#                 test_tensors.append(torch.tensor([f.small_proc_icd9 for f in test_features], dtype=torch.long))
#                 featurePositionDict["small_proc_icd9"] = positionIdx
#                 positionIdx+=1    
#             if "cui" in args.features:
#                 test_tensors.append(torch.tensor([f.cui for f in test_features], dtype=torch.long))
#                 featurePositionDict["cui"] = positionIdx
#                 positionIdx+=1

#             test_data  = TensorDataset(*test_tensors)

#             if args.local_rank == -1:
#                 test_sampler   = SequentialSampler(test_data)
#             else:
#                 test_sampler  = DistributedSampler(test_data)

#             test_dataloader  = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)


#             model = BertForSequenceClassification.from_pretrained(args.bert_model, 1, args.features)

#             ## Setting WandBI to log gradients and model parameters
#             wandb.watch(model)

#             if args.fp16:
#                 model.half()
#             model.to(device)
#             if args.local_rank != -1:
#                 model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
#                                                                   output_device=args.local_rank)
#             elif n_gpu > 1:
#                 model = torch.nn.DataParallel(model)

#             ##if freeze_bert:
#             for name, param in model.named_parameters():
#                 if name.startswith("bert"): # classifier.weight; classifier.bias
#                     # print(name, param.type())            
#                     param.requires_grad = False
#                     param = param.cpu() # Force the unused parameters to the cpu, to free GPU space for other things
#                     # print(name, param.type())  

#             m = nn.Sigmoid()
#             logger.info("***** Running testing *****")
#             logger.info("  Num examples = %d", len(test_features))
#             logger.info("  Batch size = %d", args.test_batch_size)

#             model.eval()
#             test_loss, test_accuracy = 0, 0
#             nb_test_steps, nb_test_examples = 0, 0
#             true_labels=[]
#             pred_labels=[]
#             logits_history=[]

#             if "clinical_text" in args.features:
#                 for precomputed_texts, label_ids, *nonTextFeatures in tqdm(test_dataloader, desc="Test Step"):
#                     precomputed_texts = precomputed_texts.to(device)
#                     label_ids = label_ids.to(device)

#                     with torch.no_grad():
#                         if nonTextFeatures:
#                             nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]
#                             tmp_test_loss, logits = model(precomputed_texts, label_ids, features_name=args.features, features_tensors=nonTextFeatures,
#                                                           feature_position_dict=featurePositionDict)
#                             # logits = model(input_ids,segment_ids,input_mask, features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)
#                         else:
#                             tmp_test_loss, logits = model(precomputed_texts, label_ids, features_name=args.features)
#                             # logits = model(input_ids, segment_ids, input_mask, features_name=args.features,)

#                     logits = torch.squeeze(m(logits)).detach().cpu().numpy()
#                     label_ids = label_ids.to('cpu').numpy()
#                     outputs = np.asarray([1 if i else 0 for i in (logits.flatten()>=0.5)])
#                     true_labels = true_labels + label_ids.flatten().tolist()
#                     pred_labels = pred_labels + outputs.flatten().tolist()
#                     logits_history = logits_history + logits.flatten().tolist()

#                     test_accuracy += np.sum(outputs == label_ids)
#                     test_loss += tmp_test_loss.mean().item()
#                     nb_test_steps += 1
#                     nb_test_examples += label_ids.size

#             else:
#                 for label_ids, *nonTextFeatures in tqdm(test_dataloader, desc="Test Step"):
#                     label_ids = label_ids.to(device)
#                     nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]

#                     with torch.no_grad():
#                         tmp_test_loss, logits = model(labels=label_ids, features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)
#                         # logits = model(features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)                    

#                     logits = torch.squeeze(m(logits)).detach().cpu().numpy()
#                     label_ids = label_ids.to('cpu').numpy()
#                     outputs = np.asarray([1 if i else 0 for i in (logits.flatten()>=0.5)])
#                     true_labels = true_labels + label_ids.flatten().tolist()
#                     pred_labels = pred_labels + outputs.flatten().tolist()
#                     logits_history = logits_history + logits.flatten().tolist()

#                     test_accuracy += np.sum(outputs == label_ids)
#                     test_loss += tmp_test_loss.mean().item()
#                     nb_test_steps += 1
#                     nb_test_examples += label_ids.size


#             test_accuracy /= nb_test_examples
#             test_loss = test_loss / nb_test_steps

#             # df = pd.DataFrame({'logits':logits_history, 'pred_label': pred_labels, 'label':true_labels})
#             # string = 'test_logits_fold' + str(i) + '_' + args.readmission_mode + '_chunks.csv'
#             # df.to_csv(os.path.join(args.output_dir, string))

#             df_test = pd.read_csv(csv_filepath)
#             fpr, tpr, df_out = vote_score(df_test, logits_history, args)

#             # string = 'test_logits_fold' + str(test_index[0]) + '_' + args.readmission_mode + '_readmissions.csv'
#             # df_out.to_csv(os.path.join(args.output_dir,string))

#             test_precision, test_recall, test_f1score, test_support_matrix = precision_recall_fscore_support(true_labels, pred_labels)
#             test_rp80 = vote_pr_curve(df_test, logits_history, args)

#             wandb.log({"Test loss": test_loss, "Test accuracy": test_accuracy, "Recall at Precision 80 (RP80)": test_rp80})          

#             result = {'Test loss':            test_loss,
#                       'Test accuracy':        test_accuracy,
#                       'Test RP80':            test_rp80,
#                       'Test Precision':       test_precision[1],
#                       'Test Recall':          test_recall[1],
#                       'Test F1-Score':        test_f1score[1],
#                      }

#             logger.info("***** Results %d *****\n")
#             for key in sorted(result.keys()):
#                 logger.info("  %s = %s", key, str(result[key]))



    #   Close the run by finishing it
        wandb.finish()
        
    # If the run is interrupted, delete junk directories
    except KeyboardInterrupt:
        wandb.finish()
        shutil.rmtree(run_dir)
        shutil.rmtree(args.output_dir)
