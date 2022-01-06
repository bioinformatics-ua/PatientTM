import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from torchmetrics.functional import auc, auroc, precision_recall_curve
#from sklearn.utils.fixes import signature
from funcsigs import signature

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from diagnosis_prediction.data_processor import processString

def convert_string_to_array(pandasDataFrame, columnName):
    labels = []
    for index in pandasDataFrame.index:
        entry = pandasDataFrame.at[index, columnName]
        entry = [int(x) for x in processString(entry, charsToRemove = "[] ").split(',')]
        entry = np.array(entry, dtype=np.float)
        labels.append(entry)
    return labels


def convert_scores_to_labels(scores, threshold):
    labels = []
    for entry in scores:
        entry = np.array([1. if i else 0. for i in (entry>=threshold)])
        labels.append(entry)
    return labels


def get_values_to_list(values):
    outputList = []
    for entry in values:
        outputList.append(entry)
    return outputList


def get_values_from_dataframe_to_list(dataframeValues):
    outputList = []
    for entry in dataframeValues:
        outputList.append(np.array(entry, dtype=np.float))
    return outputList


def recall_at_k(yTrueLabels, yPredScores, k=10):
    recallList = []
    for trueLabel, predScore in zip(yTrueLabels, yPredScores):
        idx = np.argpartition(predScore, -k)[-k:]
        correctPreds, realLabelCount = count_positive_labels_at_idx(trueLabel, idx)
        recallList.append(correctPreds/realLabelCount)     
    return sum(recallList)/len(recallList)
        

def count_positive_labels_at_idx(trueLabels, indices):
    realLabels = np.sum(trueLabels)
    predictedLabelsMatching = 0
    for idx in indices:
        if trueLabels[idx] == 1:
            predictedLabelsMatching += 1
    return predictedLabelsMatching, realLabels




def compute_metrics(dataFrame, numLabels, args, label, threshold=0.5):
    metrics = dict()
        
    dataFrameSort = dataFrame.sort_values(by=['HADM_ID'])
    scalingFactor = 2
   
    if "clinical_text" in args.features:
        dataFrameMetrics = dataFrameSort.groupby(['HADM_ID']).agg(numNotes=('HADM_ID', lambda x: len(x)),
                                                                  maxScore=('pred_label_scores', lambda x: np.vstack(x).max(axis=0).tolist()),
                                                                  sumScore=('pred_label_scores', lambda x: np.vstack(x).sum(axis=0).tolist()))

        dataFrameMetrics['maxScore'] = dataFrameMetrics['maxScore'].apply(np.array)
        dataFrameMetrics['sumScore'] = dataFrameMetrics['sumScore'].apply(np.array)
        yPredScores = (dataFrameMetrics['maxScore'] + dataFrameMetrics['sumScore'] / scalingFactor) / (1 + dataFrameMetrics['numNotes'] / scalingFactor)
        yPredScores = get_values_to_list(yPredScores)
        yPredLabels = convert_scores_to_labels(yPredScores, threshold)
                
        yTrueLabel = dataFrameSort.drop(["TEXT"], axis=1)
        yTrueLabel = yTrueLabel.drop_duplicates(subset=["HADM_ID"])
        yTrueLabel = convert_string_to_array(yTrueLabel, label)  
        
    else:

        yPredScores = get_values_from_dataframe_to_list(dataFrameSort['pred_label_scores'])
        yPredLabels = convert_scores_to_labels(yPredScores, threshold)
        yTrueLabel = convert_string_to_array(dataFrameSort, label)
    
    # Micro and Macro precision/recall/F1
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(yTrueLabel, yPredLabels, average='micro')
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(yTrueLabel, yPredLabels, average='macro')
    
    # Micro average precision score instead of micro AU-PR
    micro_avg_precision_score = average_precision_score(yTrueLabel, yPredScores, average='micro')
    
    # Recall@k as in the Lig-Doctor paper
    recall_at_10 = recall_at_k(yTrueLabel, yPredScores, k=10)
    recall_at_20 = recall_at_k(yTrueLabel, yPredScores, k=20)
    recall_at_30 = recall_at_k(yTrueLabel, yPredScores, k=30)
        
    yPredScores = torch.tensor(yPredScores)
    yTrueLabel  = torch.tensor(yTrueLabel, dtype=torch.long)
        
    # Micro and Macro AU-ROC
    micro_auc = auroc(yPredScores, yTrueLabel, num_classes=numLabels, average='micro')
    macro_auc = auroc(yPredScores, yTrueLabel, num_classes=numLabels, average='macro')
    
    # Recall@precision80 and AUC
    precision, recall, thresholds = precision_recall_curve(yPredScores, yTrueLabel, num_classes=numLabels)
    # aupr = auc(recall, precision) # Using average precision score instead of AU-PR which can be too optimistic

#talvez tenha de tirar varios rp80 e fazer uma media
    print(precision.shape)
    print(len(precision))
    


    pr_thres = pd.DataFrame(data = list(zip(precision, recall, thresholds)), columns = ['prec','recall','thres'])
    temp = pr_thres[pr_thres.prec > 0.799999].reset_index()
    rp80 = 0
    if temp.size == 0:
        print('Sample too small or RP80=0')
    else:
        rp80 = temp.iloc[0].recall
        print('Recall at Precision of 80 is {}', rp80)
    

    ### plot roc and pr curves? not for now
    
    
    metrics["micro_precision"] = micro_precision
    metrics["macro_precision"] = macro_precision
    metrics["micro_recall"] = micro_recall
    metrics["macro_recall"] = macro_recall
    metrics["micro_f1"] = micro_f1
    metrics["macro_f1"] = macro_f1
    metrics["micro_auc"] = micro_auc
    metrics["macro_auc"] = macro_auc
    metrics["micro_avg_precision_score"] = micro_avg_precision_score
    metrics["recall@p80"] = rp80
    metrics["recall@10"] = recall_at_10
    metrics["recall@20"] = recall_at_20
    metrics["recall@30"] = recall_at_30
    
    
    return metrics

