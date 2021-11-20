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
import shutil
import logging
import random
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import trange, tqdm
from datetime import datetime

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

from modeling_readmission import BertForSequenceClassification, BertForSequenceClassificationOriginal
from data_processor import convert_examples_to_features, readmissionProcessorText, readmissionProcessorNoText
from evaluation import vote_score, vote_pr_curve

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


def runReadmission(args):
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
    logger = logging.getLogger(__name__)
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S_")

    ##Initialize wandb to upload run data
    wandb.init(project="clinicalBERT")#,config={"epochs": 4})
    run_name = wandb.run.name
    
    config = wandb.config
    # wandb.config.update({"lr": 0.1, "channels": 16})
    # config.learning_rate = 0.01
    
    if "clinical_text" in args.features:
        file_ending = "_text.csv"
        processors = {"readmission": readmissionProcessorText}
    else:
        file_ending = "_notext.csv"
        processors = {"readmission": readmissionProcessorNoText}


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
    label_list = processor.get_labels()

    model = BertForSequenceClassification.from_pretrained(args.bert_model, 1, args.features)
    # model = BertForSequenceClassificationOriginal.from_pretrained(args.bert_model, 1)
    
    ## Setting WandBI to log gradients and model parameters
    wandb.watch(model)

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

    global_step = 0
    train_loss=100000
    number_training_steps=1
    global_step_check=0
    train_loss_history=[]
    
    min_validation_loss=100000
    val_loss_history=[]
        
    if args.do_train:

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
        
        folds = [i for i in range(10)]
        folds_dataset = []
        for i in folds:
            dataset = processor.get_examples(args.data_dir, features=args.features, fold=i)
            folds_dataset.append(convert_examples_to_features(dataset, label_list, args.features, maxLenDict))
        print("All dataset folds loaded.\n")
        
        history_training_loss, history_validation_loss, history_test_loss = [], [], []
        history_val_accuracy, history_val_RP80, history_val_precision, history_val_recall, history_val_f1score = [], [], [], [], []
        history_test_accuracy, history_test_RP80, history_test_precision, history_test_recall, history_test_f1score = [], [], [], [], []

# Beginning the folding process here
        # repeatedkFold = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)
        kFold = KFold(n_splits=10)
        for i, (train_index, test_index) in enumerate(kFold.split(folds)):
            val_index = [train_index[-1]] #We select the last partition from the train set for validation
            train_index = train_index[:-1]
        
            train_features = [fold_dataset for fold_index in train_index for fold_dataset in folds_dataset[fold_index]]
            val_features   = [fold_dataset for fold_index in val_index   for fold_dataset in folds_dataset[fold_index]]
            test_features  = [fold_dataset for fold_index in test_index  for fold_dataset in folds_dataset[fold_index]]
            
            num_train_steps = int(len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
            logger.info("***** Running training in configuration %d *****", i)
            logger.info("  Num examples = %d", len(train_features))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
   
            if "clinical_text" in args.features:
                train_precomputed_text = torch.tensor([f.clinical_text for f in train_features], dtype=torch.float)
                train_all_label_ids    = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
                train_tensors          = [train_precomputed_text, train_all_label_ids]
                val_precomputed_text   = torch.tensor([f.clinical_text for f in val_features], dtype=torch.float)
                val_all_label_ids      = torch.tensor([f.label_id for f in val_features], dtype=torch.long)
                val_tensors            = [val_precomputed_text, val_all_label_ids]
                test_precomputed_texts = torch.tensor([f.clinical_text for f in test_features], dtype=torch.float)
                test_all_label_ids     = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
                test_tensors           = [test_precomputed_texts, test_all_label_ids]
            else:
                train_all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
                train_tensors       = [train_all_label_ids]
                val_all_label_ids   = torch.tensor([f.label_id for f in val_features], dtype=torch.long)
                val_tensors         = [val_all_label_ids]
                test_label_ids      = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
                test_tensors        = [test_label_ids]

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
                # print(f'target train 0/1: {len(np.where(train_all_label_ids == 0)[0])}/{len(np.where(train_all_label_ids == 1)[0])}')
                class_sample_count = np.unique(train_all_label_ids, return_counts=True)[1]
                print(f'Target train class distribution 0/1: {class_sample_count}')
                weight = 1. / class_sample_count
                samples_weight = weight[train_all_label_ids]

                new_majority_proportion = 3
                class_sample_count[0] /= new_majority_proportion
                weight = 1. / class_sample_count
                samples_weight = weight[train_all_label_ids]

                samples_weight = torch.from_numpy(samples_weight)       
                train_sampler  = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)    
                val_sampler    = SequentialSampler(val_data)
                test_sampler   = SequentialSampler(test_data)
            else:
                train_sampler = DistributedSampler(train_data)
                val_sampler   = DistributedSampler(val_data)
                test_sampler  = DistributedSampler(test_data)

            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
            val_dataloader   = DataLoader(val_data, sampler=val_sampler, batch_size=args.train_batch_size)
            test_dataloader  = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)
                  
                
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
                        precomputed_texts, label_ids, *nonTextFeatures = batch
                        precomputed_texts = precomputed_texts.to(device)
                        label_ids = label_ids.to(device)     
                        if nonTextFeatures:
                            nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]
                            loss, logits = model(precomputed_texts, label_ids, features_name=args.features, features_tensors=nonTextFeatures,
                                                 feature_position_dict=featurePositionDict) 
                        else:
                            loss, logits = model(precomputed_texts, label_ids, features_name=args.features)
                    else:
                        label_ids, *nonTextFeatures = batch
                        label_ids = label_ids.to(device) 
                        nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]
                        loss, logits = model(labels=label_ids, features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)


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
                            precomputed_texts, label_ids, *nonTextFeatures = batch
                            precomputed_texts = precomputed_texts.to(device)
                            label_ids = label_ids.to(device)    
                            if nonTextFeatures:
                                nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]
                                loss, logits = model(precomputed_texts, label_ids, features_name=args.features, features_tensors=nonTextFeatures,
                                                     feature_position_dict=featurePositionDict) 
                            else:
                                loss, logits = model(precomputed_texts, label_ids, features_name=args.features)
                        else:
                            label_ids, *nonTextFeatures = batch
                            label_ids = label_ids.to(device) 
                            nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]
                            loss, logits = model(labels=label_ids, features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)

                    logits = torch.squeeze(m(logits)).detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.asarray([1 if i else 0 for i in (logits.flatten()>=0.5)])
                    true_labels = true_labels + label_ids.flatten().tolist()
                    pred_labels = pred_labels + outputs.flatten().tolist()
                    logits_history = logits_history + logits.flatten().tolist()

                    val_accuracy += np.sum(outputs == label_ids)
                    val_loss += loss.mean().item()
                    nb_val_steps += 1
                    nb_val_examples += label_ids.size

                val_accuracy /= nb_val_examples
                val_loss = val_loss/nb_val_steps

                df_val = pd.read_csv(os.path.join(args.data_dir, "fold" + str(val_index[0]) + file_ending))
                fpr, tpr, df_out = vote_score(df_val, logits_history, args)
                # string = 'validation_logits_fold' + str(val_index[0]) + '_' + args.readmission_mode + '_readmissions.csv'
                # df_out.to_csv(os.path.join(args.output_dir,string))

                val_precision, val_recall, val_f1score, support_matrix = precision_recall_fscore_support(true_labels, pred_labels)
                print(f'Precision {val_precision[1]}, Recall {val_recall[1]}, F1-Score {val_f1score[1]}') #The first entry is for class 0, the second for class 1
                val_rp80 = vote_pr_curve(df_val, logits_history, args)

                wandb.log({"Validation loss": val_loss, "Validation accuracy": val_accuracy, "Recall at Precision 80 (RP80)": val_rp80}) 

                ## "Early stopping" mechanism where validation loss is used to save model checkpoints
                if args.early_stop:
                    if val_loss < min_validation_loss:
                        min_validation_loss = val_loss
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
                for precomputed_texts, label_ids, *nonTextFeatures in tqdm(test_dataloader, desc="Test Step"):
                    precomputed_texts = precomputed_texts.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        if nonTextFeatures:
                            nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]
                            tmp_test_loss, logits = model(precomputed_texts, label_ids, features_name=args.features, features_tensors=nonTextFeatures,
                                                          feature_position_dict=featurePositionDict)
                            # logits = model(input_ids,segment_ids,input_mask, features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)
                        else:
                            tmp_test_loss, logits = model(precomputed_texts, label_ids, features_name=args.features)
                            # logits = model(input_ids, segment_ids, input_mask, features_name=args.features,)

                    logits = torch.squeeze(m(logits)).detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.asarray([1 if i else 0 for i in (logits.flatten()>=0.5)])
                    true_labels = true_labels + label_ids.flatten().tolist()
                    pred_labels = pred_labels + outputs.flatten().tolist()
                    logits_history = logits_history + logits.flatten().tolist()

                    test_accuracy += np.sum(outputs == label_ids)
                    test_loss += tmp_test_loss.mean().item()
                    nb_test_steps += 1
                    nb_test_examples += label_ids.size

            else:
                for label_ids, *nonTextFeatures in tqdm(test_dataloader, desc="Test Step"):
                    label_ids = label_ids.to(device)
                    nonTextFeatures = [feature.to(device) for feature in nonTextFeatures]

                    with torch.no_grad():
                        tmp_test_loss, logits = model(labels=label_ids, features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)
                        # logits = model(features_name=args.features, features_tensors=nonTextFeatures, feature_position_dict=featurePositionDict)                    

                    logits = torch.squeeze(m(logits)).detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.asarray([1 if i else 0 for i in (logits.flatten()>=0.5)])
                    true_labels = true_labels + label_ids.flatten().tolist()
                    pred_labels = pred_labels + outputs.flatten().tolist()
                    logits_history = logits_history + logits.flatten().tolist()

                    test_accuracy += np.sum(outputs == label_ids)
                    test_loss += tmp_test_loss.mean().item()
                    nb_test_steps += 1
                    nb_test_examples += label_ids.size


            test_accuracy /= nb_test_examples
            test_loss = test_loss / nb_test_steps

            # df = pd.DataFrame({'logits':logits_history, 'pred_label': pred_labels, 'label':true_labels})
            # string = 'test_logits_fold' + str(i) + '_' + args.readmission_mode + '_chunks.csv'
            # df.to_csv(os.path.join(args.output_dir, string))
            
            df_test = pd.read_csv(os.path.join(args.data_dir, "fold" + str(test_index[0]) + file_ending))
            fpr, tpr, df_out = vote_score(df_test, logits_history, args)

            # string = 'test_logits_fold' + str(test_index[0]) + '_' + args.readmission_mode + '_readmissions.csv'
            # df_out.to_csv(os.path.join(args.output_dir,string))

            test_precision, test_recall, test_f1score, test_support_matrix = precision_recall_fscore_support(true_labels, pred_labels)
            test_rp80 = vote_pr_curve(df_test, logits_history, args)

            wandb.log({"Test loss": test_loss, "Test accuracy": test_accuracy, "Recall at Precision 80 (RP80)": test_rp80})          

            result = {'Training loss':        train_loss/number_training_steps,
                      'Validation loss':      val_loss,
                      'Validation accuracy':  val_accuracy,
                      'Validation RP80':      val_rp80,
                      'Validation Precision': val_precision[1],
                      'Validation Recall':    val_recall[1],
                      'Validation F1-Score':  val_f1score[1],
                      'Test accuracy':        test_accuracy,
                      'Test RP80':            test_rp80,
                      'Test Precision':       test_precision[1],
                      'Test Recall':          test_recall[1],
                      'Test F1-Score':        test_f1score[1],
                     }
            
            history_training_loss.append(train_loss/number_training_steps)
            history_validation_loss.append(val_loss)
            history_test_loss.append(test_loss)
            history_val_accuracy.append(val_accuracy)
            history_val_RP80.append(val_rp80)
            history_val_precision.append(val_precision[1])
            history_val_recall.append(val_recall[1])
            history_val_f1score.append(val_f1score[1])
            history_test_accuracy.append(test_accuracy)
            history_test_RP80.append(test_rp80)
            history_test_precision.append(test_precision[1])
            history_test_recall.append(test_recall[1])
            history_test_f1score.append(test_f1score[1])
            
            output_results_file = os.path.join(args.output_dir, "validation_and_test_results.txt")
            if not os.path.exists(output_results_file):
                with open(output_results_file, "w") as writer:
                    writer.write("***** Logging validation and test results for cross validation *****\n")
            else:
                with open(output_results_file, "a") as writer:
                    logger.info("***** Results fold %d *****", i)
                    writer.write("***** Results fold %d *****" % (i))
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
        
        mean_result = {'Training loss':       sum(history_training_loss)/len(history_training_loss),
                      'Validation loss':      sum(history_validation_loss)/len(history_validation_loss),
                      'Validation accuracy':  sum(history_val_accuracy)/len(history_val_accuracy),               
                      'Validation RP80':      sum(history_val_RP80)/len(history_val_RP80),
                      'Validation Precision': sum(history_val_precision)/len(history_val_precision),
                      'Validation Recall':    sum(history_val_recall)/len(history_val_recall),
                      'Validation F1-Score':  sum(history_val_f1score)/len(history_val_f1score),
                      'Test accuracy':        sum(history_test_accuracy)/len(history_test_accuracy),               
                      'Test RP80':            sum(history_test_RP80)/len(history_test_RP80),
                      'Test Precision':       sum(history_test_precision)/len(history_test_precision),
                      'Test Recall':          sum(history_test_recall)/len(history_test_recall),
                      'Test F1-Score':        sum(history_test_f1score)/len(history_test_f1score),
                      }
        
        print("Writing mean performances across all folds")
        with open(output_results_file, "a") as writer:
            logger.info("***** Average performance across all folds *****")
            writer.write("***** Average performance across all folds *****")
            for key in sorted(mean_result.keys()):
                logger.info("  %s = %s", key, str(mean_result[key]))
                writer.write("%s = %s\n" % (key, str(mean_result[key])))
        
    
    
    
#   Close the run by finishing it
    wandb.finish()
        