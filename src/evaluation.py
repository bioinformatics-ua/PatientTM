import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
#from sklearn.utils.fixes import signature
from funcsigs import signature

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def compute_accuracy_noclinicaltext(true_labels, pred_labels, hadm_ids):
    unique_entries = 0
    correct_predictions = 0
    incorrect_predictions = 0
    prev_hadm_id = ""
    for idx, hadm_id in enumerate(hadm_ids):
        if hadm_id != prev_hadm_id:
            prev_hadm_id = hadm_id
            unique_entries+=1
            if true_labels[idx] == pred_labels[idx]: correct_predictions+=1
            else:
                # print(f'Incorrect prediction: true label {true_labels[idx]}, pred label {pred_labels[idx]}, hadm_id {hadm_id}')
                incorrect_predictions+=1
    accuracy = correct_predictions / unique_entries 
    print(f'Correct predictions {correct_predictions} vs Incorrect predictions {incorrect_predictions}')
    return accuracy

def vote_score(df, score, args):
    df['pred_score'] = score
    df_sort = df.sort_values(by=['HADM_ID'])
    #score 
    scaling_factor = 2
    if "clinical_text" in args.features:
        temp = (df_sort.groupby(['HADM_ID'])['pred_score'].agg(max) + df_sort.groupby(['HADM_ID'])['pred_score'].agg(sum)/
                scaling_factor)/(1+df_sort.groupby(['HADM_ID'])['pred_score'].agg(len)/scaling_factor)
        y = df_sort.groupby(['HADM_ID'])['Label'].agg(np.min).values
    else:
        temp = df_sort['pred_score']
        y = df_sort['Label'].values
        
    df_out = pd.DataFrame({'logits': temp.values, 'Label': y})
    fpr, tpr, thresholds = roc_curve(y, temp.values)
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
    df_sort = df.sort_values(by=['HADM_ID'])
    #score
    scaling_factor = 2
    if "clinical_text" in args.features:
        temp = (df_sort.groupby(['HADM_ID'])['pred_score'].agg(max)+df_sort.groupby(['HADM_ID'])['pred_score'].agg(sum)/
                scaling_factor)/(1+df_sort.groupby(['HADM_ID'])['pred_score'].agg(len)/scaling_factor)
        y = df_sort.groupby(['HADM_ID'])['Label'].agg(np.min).values
    else:
        temp = df_sort['pred_score']
        y = df_sort['Label'].values
        
    precision, recall, thres = precision_recall_curve(y, temp)
    pr_thres = pd.DataFrame(data = list(zip(precision, recall, thres)), columns = ['prec','recall','thres'])
    vote_df = pd.DataFrame(data =  list(zip(temp, y)), columns = ['score','label'])
    
    pr_curve_plot(y, temp, args)
    temp = pr_thres[pr_thres.prec > 0.799999].reset_index()
    
    rp80 = 0
    if temp.size == 0:
        print('Sample too small or RP80=0')
    else:
        rp80 = temp.iloc[0].recall
        print('Recall at Precision of 80 is {}', rp80)

    return rp80