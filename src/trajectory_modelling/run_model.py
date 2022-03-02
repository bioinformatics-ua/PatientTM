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
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence

from ranger21 import Ranger21 as RangerOptimizer
#important

from trajectory_modelling.data_processor import readmissionProcessor, diagnosisProcessor, multihotDiagnosisProcessor
from trajectory_modelling.modeling_trajectory import GRU, LSTM
from trajectory_modelling.evaluation import compute_metrics_readmission, compute_metrics_diagnosis

def get_indices_perclass_and_sample(label_list):
    indices_0, indices_1 = [], []
    for i, label in enumerate(label_list):
        if label == 1:
            indices_1.append(i)
        else:
            indices_0.append(i)
    #Sub sample negative class to equalize distribution        
    indices_0 = random.sample(indices_0, len(indices_1))
    return [*indices_0, *indices_1]


def runTrajectoryModelling(args):
    
    try:
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
        logger = logging.getLogger(__name__)

        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S_")

        ##Initialize wandb to upload run data
        wandb.init(project="PatientTM_Trajectory")#,config={"epochs": 4})
        run_name = wandb.run.name

        run_dir  = os.path.split(wandb.run.dir)[0]
        
        data_type = os.path.normpath(args.data_dir).split(os.path.sep)
        data_type = "".join([data_type[-2],"/",data_type[-1]])

        # config = wandb.config
        # wandb.config.update({"lr": 0.1, "channels": 16})
        # wandb.config.features = args.features
        wandb.config.dataset = data_type

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
            
        if args.recurrent_hidden_size:
            wandb.config.recurrent_hidden_size = args.recurrent_hidden_size
        if args.recurrent_num_layers:
            wandb.config.recurrent_num_layers = args.recurrent_num_layers
        if args.recurrent_network:
            wandb.config.recurrent_network = args.recurrent_network
        if args.bidirectional:
            wandb.config.bidirectional = args.bidirectional  
        if args.visit_sliding_window:
            wandb.config.visit_sliding_window = args.visit_sliding_window
        if args.weight_decay:
            wandb.config.weight_decay = args.weight_decay  
        if args.multi_hot_diag:
            wandb.config.multi_hot_diag = args.multi_hot_diag  
            
        if args.multi_hot_diag:
            processors = {"diagnosis": multihotDiagnosisProcessor,
                          "readmission": readmissionProcessor}
        else:
            processors = {"diagnosis": diagnosisProcessor,
                          "readmission": readmissionProcessor}
        
        if args.trajectory_subtask_name.lower() == "readmission":
            numLabels = 1
            LabelToPredict = "Label"
            file_ending = "_sliding" + str(args.visit_sliding_window) + "trajectory_readmission_notext.csv"
            
        elif args.codes_to_predict == "small_diag_icd9":
            with open("../data/extended_folds/preprocessing/smallICDdiagMapping.json", "r") as file:    
                icd9DiagCodeToIdx = json.load(file)
            numLabels = len(icd9DiagCodeToIdx)
            LabelToPredict = "LABEL_NEXT_SMALL_DIAG_ICD9"
            file_ending = "_sliding" + str(args.visit_sliding_window) + "trajectory_diagnosis_notext.csv"
            
        elif args.codes_to_predict == "diag_ccs":
            with open("../data/extended_folds/preprocessing/CCSdiagMapping.json", "r") as file:    
                ccsDiagCodeToIdx = json.load(file)
            numLabels = len(ccsDiagCodeToIdx)
            LabelToPredict = "LABEL_NEXT_DIAG_CCS"
            file_ending = "_sliding" + str(args.visit_sliding_window) + "trajectory_diagnosis_notext.csv"
        
        if args.bidirectional:
            bidirectionalMultiplier = 2
        else:
            bidirectionalMultiplier = 1
            
        args.recurrent_network = args.recurrent_network.lower()
                                
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
        subtask_name = args.trajectory_subtask_name.lower()

        if subtask_name not in processors:
            raise ValueError("Subtask not found: %s" % (subtask_name))

        processor = processors[subtask_name]()

        if args.do_train:

            cvFolds = 10
            folds = [i for i in range(cvFolds)]
            folds_examples, folds_lengths, folds_labels = [], [], []
            for i in folds:
                if subtask_name == "readmission":
                    examples, lengths, labels, featureDimension = processor.get_examples(args.data_dir, fold=i, visit_sliding_window=args.visit_sliding_window)
                    folds_examples.append(examples)
                    folds_lengths.append(lengths)
                    folds_labels.append(labels)
                elif subtask_name == "diagnosis":
                    examples, lengths, labels, featureDimension = processor.get_examples(args.data_dir, LabelToPredict, fold=i,
                                                                                         visit_sliding_window=args.visit_sliding_window)
                    folds_examples.append(examples)
                    folds_lengths.append(lengths)
                    folds_labels.append(labels)
            print("All dataset folds loaded.\n")

            history_training_loss, history_val_loss, history_test_loss = [], [], []
            history_val_microprecision, history_val_microrecall, history_val_microf1, history_val_microAUC, history_val_micro_rp80 = [], [], [], [], []
            history_val_microAvgPrecision, history_val_recallat10, history_val_recallat20, history_val_recallat30, history_val_macro_rp80 = [], [], [], [], []    
            history_test_microprecision, history_test_microrecall, history_test_microf1, history_test_microAUC, history_test_micro_rp80 = [], [], [], [], []
            history_test_microAvgPrecision, history_test_recallat10, history_test_recallat20, history_test_recallat30, history_test_macro_rp80 = [], [], [], [], []
            history_val_RP80, history_val_auc, history_val_avgprecision, history_val_precision, history_val_recall, history_val_f1score = [], [], [], [], [], []
            history_val_micro_rp50, history_val_micro_rp60, history_val_micro_rp70, history_test_micro_rp50, history_test_micro_rp60, history_test_micro_rp70 = [], [], [], [], [], []
            history_test_RP80, history_test_auc, history_test_avgprecision, history_test_precision, history_test_recall, history_test_f1score = [], [], [], [], [], []
            
    # Beginning the folding process here

            kFold = KFold(n_splits = cvFolds)
            for i, (train_index, test_index) in enumerate(kFold.split(folds)):
                val_index = [train_index[0]] #We select the first partition from the train set for validation
                train_index = train_index[1:]

                train_features = [feature for fold_index in train_index for feature in folds_examples[fold_index]]
                train_lengths  = [length for fold_index in train_index for length in folds_lengths[fold_index]]
                train_labels   = [label for fold_index in train_index for label in folds_labels[fold_index]]
                
                val_features  = [feature for fold_index in val_index for feature in folds_examples[fold_index]]
                val_lengths   = [length for fold_index in val_index for length in folds_lengths[fold_index]]
                val_labels    = [label for fold_index in val_index for label in folds_labels[fold_index]]
                
                test_features = [feature for fold_index in test_index for feature in folds_examples[fold_index]]
                test_lengths  = [length for fold_index in test_index for length in folds_lengths[fold_index]]
                test_labels   = [label for fold_index in test_index for label in folds_labels[fold_index]]
   
                train_idx = [i for i in range(len(train_features))]
                val_idx = [i for i in range(len(val_features))]
                test_idx = [i for i in range(len(test_features))]
                
                
                num_train_steps = int(len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
                logger.info("***** Running training in configuration %d *****", i)
                logger.info("  Num examples = %d", len(train_features))
                logger.info("  Batch size = %d", args.train_batch_size)
                logger.info("  Num steps = %d", num_train_steps)
                
                if args.subsampling:
        #   Special sampling to downsample negative class in a random manner, and have equal distribution in the dataset
                    train_idx = get_indices_perclass_and_sample(train_labels)
                    val_idx   = get_indices_perclass_and_sample(val_labels)

                    train_features = list(itemgetter(*train_idx)(train_features))
                    train_lengths = list(itemgetter(*train_idx)(train_lengths))
                    train_labels = list(itemgetter(*train_idx)(train_labels))
                    val_features   = list(itemgetter(*val_idx)(val_features))
                    val_lengths   = list(itemgetter(*val_idx)(val_lengths))
                    val_labels   = list(itemgetter(*val_idx)(val_labels))
                    
                    test_idx = get_indices_perclass_and_sample(test_labels)
                    test_features   = list(itemgetter(*test_idx)(test_features))
                    test_lengths   = list(itemgetter(*test_idx)(test_lengths))
                    test_labels   = list(itemgetter(*test_idx)(test_labels))

                train_all_features     = torch.tensor([feature for feature in train_features], dtype=torch.float)
                train_all_lengths      = torch.tensor([length for length in train_lengths], dtype=torch.long)
                train_all_labels       = torch.tensor([label for label in train_labels], dtype=torch.long)
                train_tensors          = [train_all_features, train_all_lengths, train_all_labels]
                
                val_all_features     = torch.tensor([feature for feature in val_features], dtype=torch.float)
                val_all_lengths      = torch.tensor([length for length in val_lengths], dtype=torch.long)
                val_all_labels       = torch.tensor([label for label in val_labels], dtype=torch.long)
                val_tensors          = [val_all_features, val_all_lengths, val_all_labels]
                
                test_all_features     = torch.tensor([feature for feature in test_features], dtype=torch.float)
                test_all_lengths      = torch.tensor([length for length in test_lengths], dtype=torch.long)
                test_all_labels       = torch.tensor([label for label in test_labels], dtype=torch.long)
                test_tensors          = [test_all_features, test_all_lengths, test_all_labels]
                
                train_data = TensorDataset(*train_tensors)
                val_data   = TensorDataset(*val_tensors)
                test_data  = TensorDataset(*test_tensors)

                if args.local_rank == -1:
                    train_sampler  = RandomSampler(train_data)    
                    val_sampler    = SequentialSampler(val_data)
                    test_sampler   = SequentialSampler(test_data)

                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
                val_dataloader   = DataLoader(val_data, sampler=val_sampler, batch_size=args.train_batch_size)
                test_dataloader  = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)

             ## Initialize the model before every fold training 
                if args.recurrent_network == "lstm":
                    model = LSTM(args, numLabels, input_size=featureDimension, hidden_size=args.recurrent_hidden_size, num_layers=args.recurrent_num_layers)
                elif args.recurrent_network == "gru":
                    model = GRU(args, numLabels, input_size=featureDimension, hidden_size=args.recurrent_hidden_size, num_layers=args.recurrent_num_layers)

                model.to(device)
                ## Setting WandBI to log gradients and model parameters
                wandb.watch(model)

                train_loss, min_validation_loss = 100000, 100000
                min_val_rp80 = 0
                patience_counter = 0
                number_training_steps = 1
                train_loss_history, val_loss_history = [], []
                

                # labels_1_counter = torch.sum(train_all_labels,0)
                # labels_0_counter = train_all_labels.shape[0]-labels_1_counter
                # pos_weights = labels_0_counter/labels_1_counter
                # pos_weights[pos_weights == float("Inf")] = 0
                # max_weight = torch.max(pos_weights)
                # pos_weights[pos_weights == 0] = max_weight
                # loss_criterion = BCEWithLogitsLoss(pos_weight=pos_weights)
                
                loss_criterion = BCEWithLogitsLoss() #BCEWithLogitsLoss has the sigmoid activation integrated, does not require sigmoid(logits) as input
                optimizer = RangerOptimizer(params=model.parameters(),
                                            lr=args.learning_rate,
                                            weight_decay=args.weight_decay,
                                            num_epochs=args.num_train_epochs,
                                            num_batches_per_epoch=len(train_dataloader))
                model.train()

                for epo in trange(int(args.num_train_epochs), desc="Epoch"):
                    train_loss = 0
                    nb_tr_steps = 0
                    for step, batch in enumerate(tqdm(train_dataloader, desc="Training Step")):
                        features, lengths, labels = batch
                        features = features.to(device)    
                        labels = labels.to(device) 
                        features = features.to(device) 
                        packed_sequences = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
                        optimizer.zero_grad()  
                        
                        if args.recurrent_network == "lstm":
                            cell_state_0   = torch.zeros(bidirectionalMultiplier*args.recurrent_num_layers, len(features), args.recurrent_hidden_size).to(device)
                            hidden_state_0 = torch.zeros(bidirectionalMultiplier*args.recurrent_num_layers, len(features), args.recurrent_hidden_size).to(device) 
                            logits = model(packed_sequences, hidden_state_0, cell_state_0)
                        elif args.recurrent_network == "gru":
                            hidden_state_0 = torch.zeros(bidirectionalMultiplier*args.recurrent_num_layers, len(features), args.recurrent_hidden_size).to(device)
                            logits = model(packed_sequences, hidden_state_0)
                    
                        logits = logits.to(device)
                        loss = loss_criterion(logits, labels.float())
                        loss.backward()
                        optimizer.step()
                        model.zero_grad() 
                          
                        train_loss_history.append(loss.item())
                        train_loss += loss.item()
                        wandb.log({"Training step loss": loss.item()})
                        nb_tr_steps += 1                
                         

                    number_training_steps=nb_tr_steps
                    print(f"Training loss in epoch {epo}: {train_loss/number_training_steps}")
                    wandb.log({"Training loss": train_loss/number_training_steps})
                    
                    

                    ## Proceeding with a validation stage on the validation dataset
                    # Setting model in evaluation mode:
                    model.eval()         
                    val_loss, val_accuracy = 0, 0
                    nb_val_steps, nb_val_examples = 0, 0
                    true_labels=[]
                    pred_labels=[]
                    logits_history = []
                    sigmoid = nn.Sigmoid()
                    
                    for step, batch in enumerate(tqdm(val_dataloader, desc="Validation Step")):
                        ## Turning off gradient computation for safety
                        with torch.no_grad():
                            features, lengths, labels = batch
                            features = features.to(device)    
                            labels = labels.to(device) 
                            features = features.to(device) 
                            packed_sequences = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
   
                            if args.recurrent_network == "lstm":
                                cell_state_0   = torch.zeros(bidirectionalMultiplier*args.recurrent_num_layers, len(features), args.recurrent_hidden_size).to(device)
                                hidden_state_0 = torch.zeros(bidirectionalMultiplier*args.recurrent_num_layers, len(features), args.recurrent_hidden_size).to(device) 
                                logits = model(packed_sequences, hidden_state_0, cell_state_0)
                            elif args.recurrent_network == "gru":
                                hidden_state_0 = torch.zeros(bidirectionalMultiplier*args.recurrent_num_layers, len(features), args.recurrent_hidden_size).to(device)
                                logits = model(packed_sequences, hidden_state_0)
                            
                            logits = logits.to(device)
                            loss = loss_criterion(logits, labels.float())
                    
                        logits = sigmoid(logits).detach().cpu().numpy() #torch.squeeze(m(logits)).detach().cpu().numpy()
                        logits_history += logits.tolist()                    
                        labels = labels.to('cpu').numpy()

                        val_loss += loss.mean().item()
                        nb_val_steps += 1
                        nb_val_examples += labels.size                
                    val_loss = val_loss/nb_val_steps

                    df_val = pd.read_csv(os.path.join(args.data_dir, "fold" + str(val_index[0]) + file_ending))
                    df_val = df_val.iloc[val_idx, :]
                    df_val['pred_label_scores'] = logits_history
                    
                    if args.trajectory_subtask_name.lower() == "readmission":
                        val_auc_roc, val_auc_pr, val_rp80, val_precision, val_recall, val_f1score = compute_metrics_readmission(df_val, args)
                        # print(f"AUC {val_auc_roc}, Avg Precision {val_auc_pr}, RP80 {val_rp80}")
                        wandb.log({"Val Loss": val_loss,
                                   "Val AUC":  val_auc_roc,
                                   "Val AvgPrecision": val_auc_pr,
                                   "Val RP80": val_rp80,
                                   "Val Precision": val_precision,
                                   "Val Recall": val_recall,
                                   "Val F1": val_f1score,})
                    
                    elif args.trajectory_subtask_name.lower() == "diagnosis":
                        val_metrics = compute_metrics_diagnosis(df_val, numLabels, args, label=LabelToPredict, threshold=0.5)
                        wandb.log({"Val Loss":              val_loss,
                                   "Val MicroRP50":         val_metrics["micro_recall@p50"],
                                   "Val MicroRP60":         val_metrics["micro_recall@p60"],
                                   "Val MicroRP70":         val_metrics["micro_recall@p70"],
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
                        else:
                            patience_counter += 1
                            if patience_counter == args.patience:
                                patience_counter = 0
                                break
                                
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

                sigmoid = nn.Sigmoid()
                logger.info("***** Running testing *****")
                logger.info("  Num examples = %d", len(test_features))
                logger.info("  Batch size = %d", args.test_batch_size)

                model.eval()
                test_loss, test_accuracy = 0, 0
                nb_test_steps, nb_test_examples = 0, 0
                true_labels=[]
                pred_labels=[]
                logits_history=[]
                
                for batch in tqdm(test_dataloader, desc="Test Step"):
                    ## Turning off gradient computation for safety
                    with torch.no_grad():
                        features, lengths, labels = batch
                        features = features.to(device)    
                        labels = labels.to(device) 
                        features = features.to(device) 
                        packed_sequences = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)

                        if args.recurrent_network == "lstm":
                            cell_state_0   = torch.zeros(bidirectionalMultiplier*args.recurrent_num_layers, len(features), args.recurrent_hidden_size).to(device)
                            hidden_state_0 = torch.zeros(bidirectionalMultiplier*args.recurrent_num_layers, len(features), args.recurrent_hidden_size).to(device) 
                            logits = model(packed_sequences, hidden_state_0, cell_state_0)
                        elif args.recurrent_network == "gru":
                            hidden_state_0 = torch.zeros(bidirectionalMultiplier*args.recurrent_num_layers, len(features), args.recurrent_hidden_size).to(device)
                            logits = model(packed_sequences, hidden_state_0)
                        
                        logits = logits.to(device)
                        loss = loss_criterion(logits, labels.float())

                    logits = sigmoid(logits).detach().cpu().numpy() #torch.squeeze(m(logits)).detach().cpu().numpy()
                    logits_history += logits.tolist()                    
                    labels = labels.to('cpu').numpy()

                    test_loss += loss.mean().item()
                    nb_test_steps += 1
                    nb_test_examples += labels.size  

                test_loss = test_loss / nb_test_steps

       
                df_test = pd.read_csv(os.path.join(args.data_dir, "fold" + str(test_index[0]) + file_ending))
                df_test = df_test.iloc[test_idx, :]
                df_test['pred_label_scores'] = logits_history
                
                if args.trajectory_subtask_name.lower() == "readmission":
                    test_auc_roc, test_auc_pr, test_rp80, test_precision, test_recall, test_f1score = compute_metrics_readmission(df_test, args)
                    # print(f"AUC {test_auc_roc}, Avg Precision {test_auc_pr}, RP80 {test_rp80}")
                    wandb.log({"Test Loss": test_loss,
                               "Test AUC":  test_auc_roc,
                               "Test AvgPrecision": test_auc_pr,
                               "Test RP80": test_rp80,
                               "Test Precision": test_precision,
                               "Test Recall": test_recall,
                               "Test F1": test_f1score,})
                    
                    result = {'Training loss':        train_loss/number_training_steps,
                              'Validation loss':      val_loss,
                              'Validation AUC':       val_auc_roc, 
                              'Validation AvgPrecision': val_auc_pr,
                              'Validation RP80':      val_rp80,
                              'Validation Precision': val_precision[1],
                              'Validation Recall':    val_recall[1],
                              'Validation F1-Score':  val_f1score[1],
                              'Test Loss':            test_loss,
                              'Test AUC':             test_auc_roc,
                              'Test AvgPrecision':    test_auc_pr,
                              'Test RP80':            test_rp80,
                              'Test Precision':       test_precision[1],
                              'Test Recall':          test_recall[1],
                              'Test F1-Score':        test_f1score[1],
                             }
                    
                    history_training_loss.append(train_loss/number_training_steps)
                    history_val_loss.append(val_loss)
                    history_test_loss.append(test_loss)
                    history_val_auc.append(val_auc_roc)
                    history_val_avgprecision.append(val_auc_pr)
                    history_val_RP80.append(val_rp80)
                    history_val_precision.append(val_precision[1])
                    history_val_recall.append(val_recall[1])
                    history_val_f1score.append(val_f1score[1])
                    history_test_auc.append(test_auc_roc)
                    history_test_avgprecision.append(test_auc_pr)
                    history_test_RP80.append(test_rp80)
                    history_test_precision.append(test_precision[1])
                    history_test_recall.append(test_recall[1])
                    history_test_f1score.append(test_f1score[1])

                elif args.trajectory_subtask_name.lower() == "diagnosis":
                    test_metrics = compute_metrics_diagnosis(df_test, numLabels, args, label=LabelToPredict, threshold=0.5)
                    wandb.log({"Test Loss":              test_loss,
                               "Test MicroRP50":         test_metrics["micro_recall@p50"],
                               "Test MicroRP60":         test_metrics["micro_recall@p60"],
                               "Test MicroRP70":         test_metrics["micro_recall@p70"],
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
                              "Val MicroRP50":           val_metrics["micro_recall@p50"],
                              "Val MicroRP60":           val_metrics["micro_recall@p60"],
                              "Val MicroRP70":           val_metrics["micro_recall@p70"],
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
                              "Test MicroRP50":          test_metrics["micro_recall@p50"],
                              "Test MicroRP60":          test_metrics["micro_recall@p60"],
                              "Test MicroRP70":          test_metrics["micro_recall@p70"],
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
                    history_val_micro_rp50.append(val_metrics["micro_recall@p50"])
                    history_val_micro_rp60.append(val_metrics["micro_recall@p60"])
                    history_val_micro_rp70.append(val_metrics["micro_recall@p70"])
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

                    history_test_micro_rp50.append(test_metrics["micro_recall@p50"])
                    history_test_micro_rp60.append(test_metrics["micro_recall@p60"])
                    history_test_micro_rp70.append(test_metrics["micro_recall@p70"])
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
                            
            if args.trajectory_subtask_name.lower() == "readmission":
                mean_result = {"KFold Avg Test loss":      sum(history_test_loss)/len(history_test_loss),
                               "KFold Avg RP80":           sum(history_test_RP80)/len(history_test_RP80),
                               "KFold Avg F1":             sum(history_test_f1score)/len(history_test_f1score),                   
                               "KFold Avg AUC":            sum(history_test_auc)/len(history_test_auc),
                               "KFold Avg Avg Precision":  sum(history_test_avgprecision)/len(history_test_avgprecision)}
                
                wandb.log({"KFold Avg Test loss":      sum(history_test_loss)/len(history_test_loss),
                           "KFold Avg RP80":           sum(history_test_RP80)/len(history_test_RP80),
                           "KFold Avg F1":             sum(history_test_f1score)/len(history_test_f1score),                   
                           "KFold Avg AUC":            sum(history_test_auc)/len(history_test_auc),
                           "KFold Avg Avg Precision":  sum(history_test_avgprecision)/len(history_test_avgprecision)})  

                
            elif args.trajectory_subtask_name.lower() == "diagnosis":
                mean_result = {"KFold Avg Test loss":      sum(history_test_loss)/len(history_test_loss),
                               "KFold Avg Micro RP50":     sum(history_test_micro_rp80)/len(history_test_micro_rp50),
                               "KFold Avg Micro RP60":     sum(history_test_micro_rp80)/len(history_test_micro_rp60),
                               "KFold Avg Micro RP70":     sum(history_test_micro_rp80)/len(history_test_micro_rp70),
                               "KFold Avg Micro RP80":     sum(history_test_micro_rp80)/len(history_test_micro_rp80),
                               "KFold Avg Macro RP80":     sum(history_test_macro_rp80)/len(history_test_macro_rp80),
                               "KFold Avg Micro F1":       sum(history_test_microf1)/len(history_test_microf1),                   
                               "KFold Avg Micro AUC":      sum(history_test_microAUC)/len(history_test_microAUC),
                               "KFold Avg Micro Avg Precision":       sum(history_test_microAvgPrecision)/len(history_test_microAvgPrecision),                
                               "KFold Avg Recall@10":      sum(history_test_recallat10)/len(history_test_recallat10),
                               "KFold Avg Recall@20":      sum(history_test_recallat20)/len(history_test_recallat20),
                               "KFold Avg Recall@30":      sum(history_test_recallat30)/len(history_test_recallat30)}
                
                wandb.log({"KFold Avg Test loss":      sum(history_test_loss)/len(history_test_loss),
                           "KFold Avg Micro RP50":     sum(history_test_micro_rp50)/len(history_test_micro_rp50),
                           "KFold Avg Micro RP60":     sum(history_test_micro_rp60)/len(history_test_micro_rp60),
                           "KFold Avg Micro RP70":     sum(history_test_micro_rp70)/len(history_test_micro_rp70),
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






#         if args.do_test:



    # # Close the run by finishing it
        wandb.finish()
        
    # If the run is interrupted, delete junk directories
    except KeyboardInterrupt:
        wandb.finish()
        shutil.rmtree(run_dir)
        shutil.rmtree(args.output_dir)
