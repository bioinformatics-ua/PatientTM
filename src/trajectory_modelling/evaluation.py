import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score, roc_curve, confusion_matrix, classification_report
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import precision_recall_curve as sklearn_precision_recall_curve

from torchmetrics.functional import auc, auroc, precision_recall_curve
#from sklearn.utils.fixes import signature
from funcsigs import signature

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt




def processString(string, charsToRemove):
    for char in charsToRemove: string = string.replace(char, "")
    return string


# def accuracy(out, labels):
#     outputs = np.argmax(out, axis=1)
#     return np.sum(outputs == labels)

# def compute_accuracy_noclinicaltext(true_labels, pred_labels, hadm_ids):
#     unique_entries = 0
#     correct_predictions = 0
#     incorrect_predictions = 0
#     prev_hadm_id = ""
#     for idx, hadm_id in enumerate(hadm_ids):
#         if hadm_id != prev_hadm_id:
#             prev_hadm_id = hadm_id
#             unique_entries+=1
#             if true_labels[idx] == pred_labels[idx]: correct_predictions+=1
#             else:
#                 # print(f'Incorrect prediction: true label {true_labels[idx]}, pred label {pred_labels[idx]}, hadm_id {hadm_id}')
#                 incorrect_predictions+=1
#     accuracy = correct_predictions / unique_entries 
#     print(f'Correct predictions {correct_predictions} vs Incorrect predictions {incorrect_predictions}')
#     return accuracy

# def vote_score(df, score, args):
#     df['pred_score'] = score
#     df_sort = df.sort_values(by=['HADM_ID'])
#     #score 
#     scaling_factor = 2
#     if "clinical_text" in args.features:
#         temp = (df_sort.groupby(['HADM_ID'])['pred_score'].agg(max) + df_sort.groupby(['HADM_ID'])['pred_score'].agg(sum)/
#                 scaling_factor)/(1+df_sort.groupby(['HADM_ID'])['pred_score'].agg(len)/scaling_factor)
#         y = df_sort.groupby(['HADM_ID'])['Label'].agg(np.min).values
#     else:
#         temp = df_sort['pred_score']
#         y = df_sort['Label'].values
        
#     df_out = pd.DataFrame({'logits': temp.values, 'Label': y})
#     fpr, tpr, thresholds = roc_curve(y, temp.values)
#     auc_score = auc(fpr, tpr)

#     plt.figure(1)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.plot(fpr, tpr, label='Val (area = {:.3f})'.format(auc_score))
#     plt.xlabel('False positive rate')
#     plt.ylabel('True positive rate')
#     plt.title('ROC curve')
#     plt.legend(loc='best')
#     plt.show()
#     string = 'auroc_clinicalbert_'+args.readmission_mode+'.png'
#     plt.savefig(os.path.join(args.output_dir, string))

#     return fpr, tpr, df_out

def pr_curve_plot(y, y_score, args):
    precision, recall, thresholds = sklearn_precision_recall_curve(y, y_score)
    area_pr = sklearn_auc(recall,precision)
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
              area_pr))
    
    string = 'auprc_clinicalbert_'+args.readmission_mode+'.png'

    plt.savefig(os.path.join(args.output_dir, string))
    
    return area_pr, precision, recall, thresholds


def compute_metrics_readmission(df, args):
    y_true = df["Label"].values
    y_pred = df["pred_label_scores"].values
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_roc = sklearn_auc(fpr, tpr)
           
    auc_pr, precision, recall, thres = pr_curve_plot(y_true, y_pred, args)
    #  = sklearn_precision_recall_curve(y_true, y_pred)
    # auc_pr = sklearn_auc(recall,precision)
    
    pr_thres = pd.DataFrame(data = list(zip(precision, recall, thres)), columns = ['prec','recall','thres'])
    print(pr_thres)
    
    temp = pr_thres[pr_thres.prec > 0.799999].reset_index()
    
    rp80 = 0
    if temp.size == 0:
        print('Sample too small or RP80=0')
    else:
        rp80 = temp.iloc[0].recall
        print('Recall at Precision of 80 is {}', rp80)
    
    y_pred = [1 if i else 0 for i in (y_pred.flatten()>=0.5)]
    precision, recall, f1score, _ = precision_recall_fscore_support(y_true, y_pred)

    return auc_roc, auc_pr, rp80, precision, recall, f1score



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
        #top_k = np.partition(predScore, -k)[-k:] # This retrieves the values instead of the indices
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


def compute_metrics_diagnosis(dataFrame, numLabels, args, label, threshold=0.5):
    metrics = dict()
        
#     dataFrameSort = dataFrame.sort_values(by=['HADM_ID'])
#     scalingFactor = 2
   
#     if "clinical_text" in args.features:
#         dataFrameMetrics = dataFrameSort.groupby(['HADM_ID']).agg(numNotes=('HADM_ID', lambda x: len(x)),
#                                                                   maxScore=('pred_label_scores', lambda x: np.vstack(x).max(axis=0).tolist()),
#                                                                   sumScore=('pred_label_scores', lambda x: np.vstack(x).sum(axis=0).tolist()))

#         dataFrameMetrics['maxScore'] = dataFrameMetrics['maxScore'].apply(np.array)
#         dataFrameMetrics['sumScore'] = dataFrameMetrics['sumScore'].apply(np.array)
#         yPredScores = (dataFrameMetrics['maxScore'] + dataFrameMetrics['sumScore'] / scalingFactor) / (1 + dataFrameMetrics['numNotes'] / scalingFactor)
#         yPredScores = get_values_to_list(yPredScores)
#         yPredLabels = convert_scores_to_labels(yPredScores, threshold)
                
#         yTrueLabel = dataFrameSort.drop(["TEXT"], axis=1)
#         yTrueLabel = yTrueLabel.drop_duplicates(subset=["HADM_ID"])
#         yTrueLabel = convert_string_to_array(yTrueLabel, label)  
        
#     else:


    yPredScores = get_values_from_dataframe_to_list(dataFrame['pred_label_scores'])
    yPredLabels = convert_scores_to_labels(yPredScores, threshold)
    yTrueLabel = convert_string_to_array(dataFrame, label)
    
    # Micro and Macro precision/recall/F1
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(yTrueLabel, yPredLabels, average='micro', zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(yTrueLabel, yPredLabels, average='macro', zero_division=0)
    
    # Micro average precision instead of micro AU-PR
    micro_avg_precision = average_precision_score(yTrueLabel, yPredScores, average='micro')
    
    # Recall@k as in the Lig-Doctor paper
    recall_at_10 = recall_at_k(yTrueLabel, yPredScores, k=10)
    recall_at_20 = recall_at_k(yTrueLabel, yPredScores, k=20)
    recall_at_30 = recall_at_k(yTrueLabel, yPredScores, k=30)
    
    # Micro Recall@precision80
    precision, recall, thresholds = sklearn_precision_recall_curve(np.asarray(yTrueLabel).ravel(), np.asarray(yPredScores).ravel())
    pr_thres = pd.DataFrame(data = list(zip(precision, recall, thresholds)), columns = ['prec','recall','thres'])
    print(pr_thres)
    temp50 = pr_thres[pr_thres.prec > 0.499999].reset_index()
    temp60 = pr_thres[pr_thres.prec > 0.599999].reset_index()
    temp70 = pr_thres[pr_thres.prec > 0.699999].reset_index()
    temp80 = pr_thres[pr_thres.prec > 0.799999].reset_index()
    micro_rp50 = 0
    micro_rp60 = 0
    micro_rp70 = 0
    micro_rp80 = 0
    if temp50.size == 0:
        pass
    else:
        micro_rp50 = temp50.iloc[0].recall  
        
    if temp60.size == 0:
        pass
    else:
        micro_rp60 = temp60.iloc[0].recall  
        
    if temp70.size == 0:
        pass
    else:
        micro_rp70 = temp70.iloc[0].recall  
        
    if temp80.size == 0:
        pass
    else:
        micro_rp80 = temp80.iloc[0].recall  
        
        
    yPredScores = torch.tensor(yPredScores)
    yTrueLabel  = torch.tensor(yTrueLabel, dtype=torch.long)
        
    # Micro and Macro AU-ROC
    micro_auc = auroc(yPredScores, yTrueLabel, num_classes=numLabels, average='micro')
    macro_auc = auroc(yPredScores, yTrueLabel, num_classes=numLabels, average='macro')
    
    # Macro Recall@precision80
    precision, recall, thresholds = precision_recall_curve(yPredScores, yTrueLabel, num_classes=numLabels)
    # aupr = auc(recall, precision) # Using average precision above instead of AU-PR here as AU-PR uses interpolation and can be too optimistic
    rp80_list = []
    for prec, rec, threshold in zip(precision, recall, thresholds):
        pr_thres = pd.DataFrame(data = list(zip(prec, rec, threshold)), columns = ['prec','recall','thres'])
        temp = pr_thres[pr_thres.prec > 0.799999].reset_index()
        rp80 = 0
        if temp.size == 0:
            pass
            # print('Sample too small or RP80=0')
        else:
            rp80 = temp.iloc[0].recall
            # print('Recall at Precision of 80 is {}', rp80)
            
        rp80_list.append(rp80) # colocar isto dentro do else? favorece muito os resultados
    average_macro_rp_80 = sum(rp80_list)/len(rp80_list)
    
    # # Micro Recall@precision80
    # precision, recall, thresholds = precision_recall_curve(yPredScores.ravel(), yTrueLabel.ravel())
    # pr_thres = pd.DataFrame(data = list(zip(precision, recall, thresholds)), columns = ['prec','recall','thres'])
    # temp = pr_thres[pr_thres.prec > 0.799999].reset_index()
    # rp80 = 0
    # if temp.size == 0:
    #     pass
    #     # print('Sample too small or RP80=0')
    # else:
    #     rp80 = temp.iloc[0].recall
    #     # print('Recall at Precision of 80 is {}', rp80)             


    ### plot roc and pr curves? not for now  
    
    metrics["micro_precision"] = micro_precision
    metrics["macro_precision"] = macro_precision
    metrics["micro_recall"] = micro_recall
    metrics["macro_recall"] = macro_recall
    metrics["micro_f1"] = micro_f1
    metrics["macro_f1"] = macro_f1
    metrics["micro_auc"] = micro_auc
    metrics["macro_auc"] = macro_auc
    metrics["micro_avg_precision"] = micro_avg_precision
    metrics["macro_recall@p80"] = average_macro_rp_80
    metrics["micro_recall@p80"] = micro_rp80
    metrics["micro_recall@p70"] = micro_rp70
    metrics["micro_recall@p60"] = micro_rp60
    metrics["micro_recall@p50"] = micro_rp50
    metrics["recall@10"] = recall_at_10
    metrics["recall@20"] = recall_at_20
    metrics["recall@30"] = recall_at_30
    
    return metrics


