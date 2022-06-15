import pandas as pd
import numpy as np
import pickle
import os
import csv
import sys
import matplotlib.pyplot as plt
import json



# def featureToIdx(features):
#     feature2idx = {}
#     feature2idx[0] = 0
#     idx=0
#     for i, entry in enumerate(features):
#         if isinstance(entry, str):
#             entry = int(entry.strip())
#         print(i, entry)
#         feature2idx[entry] = i
#     return feature2idx

# def featureToIdx_original(features):
#     feature2idx = {}
#     feature2idx[int(0)] = 0 #will be used to mask padding "codes" in the model
#     idx=1
#     for entry in features:
#         print(idx, entry)
#         feature2idx[int(entry)] = idx
#         idx+=1
#     return feature2idx

def featureToIdx(features):
    feature2idx = {}
    feature2idx["0"] = 0 #will be used to mask padding "codes" in the model
    idx=1
    for entry in features:
        if entry in feature2idx.keys():
            pass
        else:# print(idx, entry)
            feature2idx[entry] = idx
            idx+=1
    return feature2idx


def getICDlevel1(icd9_code):
    """
    This method extracts the first level of hierarchy of an ICD code:
        - Procedure codes start with P_xxxx and only have 2 digits in the first level so we extract P_xx
        - Diagnoses codes start with D_ and can have the following formats (1) Exxx (2) Vxx (3) xxx 
    """
    if icd9_code.startswith("P"):
        return icd9_code[:4]
    elif icd9_code.startswith("D"):
        if icd9_code.startswith("D_E"):
            return icd9_code[:6]
        else:
            return icd9_code[:5]


def writeToJSON(content, filepath):
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            baseContent = json.load(file)
        content.update(baseContent)
        print("Added content to previously existing file at {}!!".format(filepath))
    with open(filepath, 'w') as file:
        json.dump(content, file)


def countCodesPerType(Icd9CcsMap):
    diagDict = dict()
    procDict = dict()
    for icd9, ccs in Icd9CcsMap.items():
        if icd9[0] == "D":
            diagDict[icd9] = ccs
        elif icd9[0] == "P":
            procDict[icd9] = ccs
    print("Number of Diagnosis ICD9 codes in map: {}.".format(len(diagDict.keys())))
    print("Number of Diagnosis CCS codes in map: {}.".format(len(set(diagDict.values()))))
    print("Number of Procedure ICD9 codes in map: {}.".format(len(procDict.keys())))
    print("Number of Procedure CCS codes in map: {}.".format(len(set(procDict.values()))))

# def map_ICD9_to_CCS(pandasDataFrame):
#     with open('../data/extended/ICDandCCSmappings/merged_icdccs_codes.json','r') as file:
#         icd9TOCCS_Map = json.load(file)
#     mappedCCSList = []
#     unmapped=0
#     mapped=0
#     dfDict = pandasDataFrame.to_dict('records')
#     for row in dfDict:
#         hadm_id = row["HADM_ID"]
#         ICD9s_List = row["ICD9_CODE"]
#         tempCodeList = []
#         for ICD9 in ICD9s_List:
#             try:
#                 CCS_code = icd9TOCCS_Map[ICD9]
#                 tempCodeList.append(CCS_code)
#                 mapped+=1
#             except KeyError:
#                 tempCodeList.append("-1") #Used for NaN entries
#                 unmapped+=1

#         mappedCCSList.append(tempCodeList)

#     # This "-1" is necessary because in proc_icd9 there exist empty entries (where no proc_icd9 codes were assigned to the patient)
#     # These empty entries will be processed as "-1" entries later on, before being fed to the model as input
#     icd9Set=set(icd9TOCCS_Map.keys())
#     icd9Set.add("-1")
#     if not os.path.isfile("../data/extended/Icd9ToIdx.json"):
#         icd9Set = sorted({entry.strip() for entry in icd9Set})
#         icd9ToIdx = featureToIdx(icd9Set)
#         writeToJSON(icd9ToIdx, "../data/extended/Icd9ToIdx.json")

#     ccsSet = set(icd9TOCCS_Map.values())
#     ccsSet.add("-1")
#     if not os.path.isfile("../data/extended/CCSToIdx.json"):
#         ccsSet = sorted({entry.strip() for entry in ccsSet})
#         ccsToIdx = featureToIdx(ccsSet)
#         writeToJSON(ccsToIdx, "../data/extended/CCSToIdx.json")

#     print('-Total number (complete set) of ICD9 codes (diag + proc): {}'.format(len(icd9Set)))
#     print('-Total number (complete set) of CCS codes (diag + proc): {}'.format(len(ccsSet)))
#     print("-Total of mapped/unmapped entries {}/{}".format(mapped,unmapped))
#     return mappedCCSList

def map_ICD9_to_CCS(pandasDataFrame):
    with open('../data/extended/ICDandCCSmappings/merged_icdccs_codes.json','r') as file:
        icd9TOCCS_Map = json.load(file)
    mappedSmallICDList = []
    mappedCCSList = []
    unmapped=0
    mapped=0  
    for row in pandasDataFrame.itertuples():
        tempSmallICDCodeList = []
        tempCCSCodeList = []
        for ICD9 in row.ICD9_CODE:
            smallICD = getICDlevel1(ICD9)
            if smallICD not in tempSmallICDCodeList: tempSmallICDCodeList.append(smallICD)
            try:
                CCS_code = icd9TOCCS_Map[ICD9]
                tempCCSCodeList.append(CCS_code)
                mapped+=1
            except KeyError:
    ## This was previously added but we decided to simply not introduce more noise if the map is unsuccessful
                # tempCCSCodeList.append("0") #Used for NaN entries
                unmapped+=1
        mappedSmallICDList.append(tempSmallICDCodeList)
        mappedCCSList.append(tempCCSCodeList)
    with open('../data/extended/ICDandCCSmappings/merged_icd_text.json','r') as file:
        icd9map = json.load(file)

    if not os.path.isfile("../data/extended/smallIcd9ToIdx.json"):
        smallICDset = set()
        for code in icd9map.keys():
            smallICDset.add(getICDlevel1(code))
        smallIcd9ToIdx = featureToIdx(smallICDset)
        writeToJSON(smallIcd9ToIdx, "../data/extended/smallIcd9ToIdx.json")

    # A "0" is added during featureToIdx because in proc_icd9 there exist empty entries (where no proc_icd9 codes were assigned to the patient)
    # These empty entries will be processed as "0" entries later on, before being fed to the model as input
    if not os.path.isfile("../data/extended/Icd9ToIdx.json"):
        icd9ToIdx = featureToIdx(set(icd9map.keys()))
        writeToJSON(icd9ToIdx, "../data/extended/Icd9ToIdx.json")

    ccsSet = set(icd9TOCCS_Map.values())
    if not os.path.isfile("../data/extended/CCSToIdx.json"):
        ccsSet = {entry.strip() for entry in ccsSet}
        ccsToIdx = featureToIdx(ccsSet)
        writeToJSON(ccsToIdx, "../data/extended/CCSToIdx.json")
    print('-Total number (complete set) of ICD9 codes (diag + proc): {}'.format(len(set(icd9map.keys()))))
    print('-Total number (complete set) of CCS codes (diag + proc): {}'.format(len(ccsSet)))
    print("-Total of mapped/unmapped entries {}/{}".format(mapped,unmapped))
    return mappedCCSList, mappedSmallICDList


def get_unique_ordered_medication(pandasDataFrame):
    with open("../data/extended/NDCmappings/ndc_cui_map.json", "r") as file:
        ndcCuiMap = json.load(file)
    RxNormNdcs = set(ndcCuiMap.keys())
    mapped=0
    unmapped=0
    if "NDC" in pandasDataFrame.columns.values:
        column = "NDC"
    elif "DRUG" in pandasDataFrame.columns.values:
        column = "DRUG"
    # counter = 0
    unique_cuis_list = []
    for index in pandasDataFrame.index:
        used_medications = set()
        #unique_medication = [x for x in pandasDataFrame.loc[index, column] if x not in used_medications and (used_medications.add(x) or True)]
        unique_medication = []
        temp_cuis_list = []
        for value in pandasDataFrame.loc[index, column]:
            if value not in used_medications and (used_medications.add(value) or True):
                if pd.isna(value) or value==0.0 or str(int(value)) not in RxNormNdcs: #Attention, the last clause leads to codes not appearing in the set
                    # if (not pd.isna(value) and not value==0.0):
                    #     print(value)
                    #     counter+=1
                    value = 0 # Swapping NaNs to a default numerical number that is not used elsewhere
                if value is not 0: unique_medication.append(str(int(value)))
            if (int(value) is not 0) and (str(int(value)) in RxNormNdcs):
                mapped+=1
            else:
                unmapped+=1
            try:
                cui = ndcCuiMap[str(int(value))]
                if cui not in temp_cuis_list: temp_cuis_list.append(cui)
            except KeyError:
                pass
        unique_cuis_list.append(temp_cuis_list)
        pandasDataFrame.at[index, column] = unique_medication
    ndcsToIdx = featureToIdx(RxNormNdcs)
    cuisToIdx = featureToIdx(set(ndcCuiMap.values()))
    if not os.path.isfile("../data/extended/NDCToIdx.json"): writeToJSON(ndcsToIdx, "../data/extended/NDCToIdx.json")
    if not os.path.isfile("../data/extended/cui_NDCToIdx.json"): writeToJSON(cuisToIdx, "../data/extended/cui_NDCToIdx.json")
    # print("mapped: {}, unmapped: {}".format(mapped, unmapped))
    # print(counter)
    return pandasDataFrame, unique_cuis_list



df_adm = pd.read_csv('/backup/mimiciii/ADMISSIONS.csv.gz', compression="gzip")
df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

## To count numbers of admissions in the dataset
visit_distribution = visit_distribution.groupby(['SUBJECT_ID'])['HADM_ID'].agg(['nunique']).reset_index(drop=False)
visit_distribution = visit_distribution.groupby(['nunique'])['SUBJECT_ID'].agg('nunique')


df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])
df_adm = df_adm.reset_index(drop = True)
df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)

df_adm['PREV_DISCHTIME'] = df_adm.groupby('SUBJECT_ID').DISCHTIME.shift(1)
df_adm['PREV_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(1)

rows = df_adm.NEXT_ADMISSION_TYPE == 'ELECTIVE'
df_adm.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
df_adm.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

# rows = df_adm.PREV_ADMISSION_TYPE == 'ELECTIVE'
# df_adm.loc[rows,'PREV_DISCHTIME'] = pd.NaT
# df_adm.loc[rows,'PREV_ADMISSION_TYPE'] = np.NaN

df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])

#When we filter out the "ELECTIVE", we need to correct the next admit time for these admissions since there might be 'emergency' next admit after "ELECTIVE"
df_adm[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')
df_adm['DAYS_NEXT_ADMIT'] = (df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME).dt.total_seconds()/(24*60*60)
df_adm[['PREV_DISCHTIME','PREV_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['PREV_DISCHTIME','PREV_ADMISSION_TYPE']].fillna(method = 'bfill')
df_adm['DAYS_PREV_ADMIT'] = (df_adm.ADMITTIME - df_adm.PREV_DISCHTIME).dt.total_seconds()/(24*60*60)

df_adm['OUTPUT_LABEL'] = (df_adm.DAYS_NEXT_ADMIT < 30).astype('int')
### filter out newborn and death
df_adm = df_adm[df_adm['ADMISSION_TYPE']!='NEWBORN']
df_adm = df_adm[df_adm.DEATHTIME.isnull()]
df_adm['DURATION']  = (df_adm['DISCHTIME']-df_adm['ADMITTIME']).dt.total_seconds()/(24*60*60)


df_diagnoses = pd.read_csv('/backup/mimiciii/DIAGNOSES_ICD.csv.gz', compression="gzip")
df_diagnoses = df_diagnoses[df_diagnoses.ICD9_CODE.notna()]
df_diagnoses = df_diagnoses.sort_values(['HADM_ID','SEQ_NUM'], ascending=True)
df_diagnoses = df_diagnoses.reset_index(drop = True)
df_diagnoses.ICD9_CODE = "D_" + df_diagnoses.ICD9_CODE.astype(str)
df_diag_listing = df_diagnoses.groupby('HADM_ID')['ICD9_CODE'].apply(list)
df_diag_listing = df_diag_listing.reset_index()
diagnosesCCS, smallICDs = map_ICD9_to_CCS(df_diag_listing)
df_diag_listing['SMALL_DIAG_ICD9'] = smallICDs
df_diag_listing['DIAG_CCS'] = diagnosesCCS

df_adm = pd.merge(df_adm,
                  df_diag_listing[['HADM_ID','ICD9_CODE','SMALL_DIAG_ICD9','DIAG_CCS']],
                  on = ['HADM_ID'],
                  how = 'left')

df_adm = df_adm.rename(columns={'ICD9_CODE': 'DIAG_ICD9'})

df_procedures = pd.read_csv('/backup/mimiciii/PROCEDURES_ICD.csv.gz', compression="gzip")
df_procedures = df_procedures.sort_values(['HADM_ID','SEQ_NUM'], ascending=True)
df_procedures = df_procedures.reset_index(drop = True)
df_procedures.ICD9_CODE = "P_" + df_procedures.ICD9_CODE.astype(str)
df_proc_listing = df_procedures.groupby('HADM_ID')['ICD9_CODE'].apply(list)
df_proc_listing = df_proc_listing.reset_index()
proceduresCCS, smallICDs = map_ICD9_to_CCS(df_proc_listing)
df_proc_listing['SMALL_PROC_ICD9'] = smallICDs
df_proc_listing['PROC_CCS'] = proceduresCCS

df_adm = pd.merge(df_adm,
                  df_proc_listing[['HADM_ID','ICD9_CODE','SMALL_PROC_ICD9','PROC_CCS']],
                  on = ['HADM_ID'],
                  how = 'left')

df_adm = df_adm.rename(columns={'ICD9_CODE': 'PROC_ICD9'})


df_medication = pd.read_csv('/backup/mimiciii/PRESCRIPTIONS.csv.gz', compression="gzip")
df_medication = df_medication.sort_values(['HADM_ID','STARTDATE'], ascending=True)
df_medication = df_medication.reset_index(drop = True)
# df_med_listing1 = df_medication.groupby('HADM_ID')['DRUG'].apply(list)
# df_med_listing1 = df_med_listing1.reset_index()
df_med_listing = df_medication.groupby('HADM_ID')['NDC'].apply(list)
df_med_listing = df_med_listing.reset_index()
df_med_listing, cuis = get_unique_ordered_medication(df_med_listing) #now the list of medication only contains unique medications, not a long list of many repeated meds
df_med_listing['CUI'] = cuis

df_adm = pd.merge(df_adm,
                  df_med_listing[['HADM_ID','NDC','CUI']],
                  on = ['HADM_ID'],
                  how = 'left')

df_adm['NEXT_SMALL_DIAG_ICD9'] = df_adm.groupby('SUBJECT_ID').SMALL_DIAG_ICD9.shift(-1)
df_adm['NEXT_DIAG_CCS'] = df_adm.groupby('SUBJECT_ID').DIAG_CCS.shift(-1)
df_adm['NEXT_SMALL_PROC_ICD9'] = df_adm.groupby('SUBJECT_ID').SMALL_PROC_ICD9.shift(-1)
df_adm['NEXT_PROC_CCS'] = df_adm.groupby('SUBJECT_ID').PROC_CCS.shift(-1)
df_adm['NEXT_CUI'] = df_adm.groupby('SUBJECT_ID').CUI.shift(-1)

df_notes = pd.read_csv('/backup/mimiciii/NOTEEVENTS.csv.gz', compression="gzip")
df_notes = df_notes.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTDATE'])
df_adm_notes = pd.merge(df_adm[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT','DAYS_PREV_ADMIT','NEXT_ADMITTIME','ADMISSION_TYPE',
                                'DEATHTIME','OUTPUT_LABEL','DURATION','DIAG_ICD9','SMALL_DIAG_ICD9','DIAG_CCS','PROC_ICD9','SMALL_PROC_ICD9',
                                'PROC_CCS','NDC', 'CUI','NEXT_SMALL_DIAG_ICD9', 'NEXT_DIAG_CCS', 'NEXT_SMALL_PROC_ICD9', 'NEXT_PROC_CCS', 'NEXT_CUI']],
                        df_notes[['SUBJECT_ID','HADM_ID','CHARTDATE','CHARTTIME','TEXT','CATEGORY']],
                        on = ['SUBJECT_ID','HADM_ID'],
                        how = 'left')

df_adm_notes["ADMITTIME_C"] = df_adm_notes.ADMITTIME.apply(lambda x: str(x).split(' ')[0])
df_adm_notes['ADMITTIME_C'] = pd.to_datetime(df_adm_notes.ADMITTIME_C, format = '%Y-%m-%d', errors = 'coerce')
df_adm_notes['CHARTDATE'] = pd.to_datetime(df_adm_notes.CHARTDATE, format = '%Y-%m-%d', errors = 'coerce')
df_adm_notes['CHARTTIME'] = pd.to_datetime(df_adm_notes.CHARTTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_adm_notes = df_adm_notes.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTDATE','CHARTTIME'])
df_adm_notes = df_adm_notes.reset_index(drop = True)


### If Discharge Summary
df_discharge = df_adm_notes[df_adm_notes['CATEGORY'] == 'Discharge summary']
# multiple discharge summary for one admission -> after examination -> replicated summary -> replace with the last one
df_discharge = (df_discharge.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()
df_discharge=df_discharge[df_discharge['TEXT'].notnull()]


### If Less than n days on admission notes (Early notes)
def less_n_days_data (df_adm_notes, n):
    df_less_n = df_adm_notes[((df_adm_notes['CHARTDATE']-df_adm_notes['ADMITTIME_C']).dt.total_seconds()/(24*60*60))<n]
    df_less_n = df_less_n[df_less_n['TEXT'].notnull()]
    #concatenate first
    df_concat = pd.DataFrame(df_less_n.groupby('HADM_ID')['TEXT'].apply(lambda x: "%s" % ' '.join(x))).reset_index()
    df_concat['OUTPUT_LABEL'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].OUTPUT_LABEL.values[0])
    df_concat['SUBJECT_ID'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].SUBJECT_ID.values[0])
    df_concat['ADMITTIME'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].ADMITTIME.values[0])
    df_concat['DAYS_NEXT_ADMIT'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].DAYS_NEXT_ADMIT.values[0])
    df_concat['DAYS_PREV_ADMIT'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].DAYS_PREV_ADMIT.values[0])
    df_concat['DURATION'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].DURATION.values[0])
    df_concat['DIAG_ICD9'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].DIAG_ICD9.values[0])
    df_concat['SMALL_DIAG_ICD9'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].SMALL_DIAG_ICD9.values[0])
    df_concat['DIAG_CCS'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].DIAG_CCS.values[0])
    df_concat['PROC_ICD9'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].PROC_ICD9.values[0])
    df_concat['SMALL_PROC_ICD9'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].SMALL_PROC_ICD9.values[0])
    df_concat['PROC_CCS'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].PROC_CCS.values[0])
    df_concat['NDC'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].NDC.values[0])
    df_concat['CUI'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].CUI.values[0])
    df_concat['NEXT_SMALL_DIAG_ICD9'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].NEXT_SMALL_DIAG_ICD9.values[0])
    df_concat['NEXT_DIAG_CCS'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].NEXT_DIAG_CCS.values[0])
    df_concat['NEXT_SMALL_PROC_ICD9'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].NEXT_SMALL_PROC_ICD9.values[0])
    df_concat['NEXT_PROC_CCS'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].NEXT_PROC_CCS.values[0])
    df_concat['NEXT_CUI'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].NEXT_CUI.values[0])  
    df_concat = df_concat.sort_values(by=['SUBJECT_ID','HADM_ID'])
    df_concat = df_concat.reset_index(drop = True)
    return df_concat

df_less_2 = less_n_days_data(df_adm_notes, 2)
df_less_3 = less_n_days_data(df_adm_notes, 3)

print("LOADED THE DATA")

import re
def preprocess1(x):
    y=re.sub('\\[(.*?)\\]','',x) #remove de-identified brackets
    y=re.sub('[0-9]+\.','',y) #remove 1.2. since the segmenter segments based on this
    y=re.sub('dr\.','doctor',y)
    y=re.sub('m\.d\.','md',y)
    y=re.sub('admission date:','',y)
    y=re.sub('discharge date:','',y)
    y=re.sub('--|__|==','',y)
    return y

def preprocessing(df_less_n):
    df_less_n['TEXT']=df_less_n['TEXT'].fillna(' ')
    df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\n',' ')
    df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\r',' ')
    df_less_n['TEXT']=df_less_n['TEXT'].apply(str.strip)
    df_less_n['TEXT']=df_less_n['TEXT'].str.lower()
    df_less_n['TEXT']=df_less_n['TEXT'].apply(lambda x: preprocess1(x))
    #to get 318 words chunks for readmission tasks
    from tqdm import tqdm
    df_len = len(df_less_n)
    #want=pd.DataFrame({'ID':[],'DIAG_ICD9':[],'DIAG_CCS':[],'PROC_ICD9':[],'PROC_CCS':[],'NDC':[],'TEXT':[],'Label':[]})
    want=pd.DataFrame({'HADM_ID':[],'SUBJECT_ID':[],'ADMITTIME':[],'DAYS_NEXT_ADMIT':[],'DAYS_PREV_ADMIT':[],'DURATION':[],'DIAG_ICD9':[],
                       'SMALL_DIAG_ICD9':[],'DIAG_CCS':[],'PROC_ICD9':[],'SMALL_PROC_ICD9':[],'PROC_CCS':[],'NDC':[],'CUI':[],'TEXT':[],'Label':[],
                       'NEXT_SMALL_DIAG_ICD9':[],'NEXT_DIAG_CCS':[],'NEXT_SMALL_PROC_ICD9':[],'NEXT_PROC_CCS':[],'NEXT_CUI':[]})
    for i in tqdm(range(df_len)):
        x=df_less_n.TEXT.iloc[i].split()
        n=int(len(x)/318)
        for j in range(n):
            want=want.append({'TEXT':' '.join(x[j*318:(j+1)*318]),'Label':df_less_n.OUTPUT_LABEL.iloc[i],'HADM_ID':df_less_n.HADM_ID.iloc[i],
                              'SUBJECT_ID':df_less_n.SUBJECT_ID.iloc[i],'ADMITTIME':df_less_n.ADMITTIME.iloc[i],
                              'DAYS_NEXT_ADMIT':df_less_n.DAYS_NEXT_ADMIT.iloc[i], 'DAYS_PREV_ADMIT':df_less_n.DAYS_PREV_ADMIT.iloc[i], 
                              'DURATION':df_less_n.DURATION.iloc[i], 'NDC':df_less_n.NDC.iloc[i], 
                              'DIAG_ICD9':df_less_n.DIAG_ICD9.iloc[i],'DIAG_CCS':df_less_n.DIAG_CCS.iloc[i],
                              'PROC_ICD9':df_less_n.PROC_ICD9.iloc[i],'PROC_CCS':df_less_n.PROC_CCS.iloc[i],
                              'SMALL_DIAG_ICD9':df_less_n.SMALL_DIAG_ICD9.iloc[i],'SMALL_PROC_ICD9':df_less_n.SMALL_PROC_ICD9.iloc[i],
                              'CUI':df_less_n.CUI.iloc[i], 'NEXT_CUI':df_less_n.NEXT_CUI.iloc[i],
                              'NEXT_SMALL_DIAG_ICD9':df_less_n.NEXT_SMALL_DIAG_ICD9.iloc[i], 'NEXT_SMALL_PROC_ICD9':df_less_n.NEXT_SMALL_PROC_ICD9.iloc[i],
                              'NEXT_DIAG_CCS':df_less_n.NEXT_DIAG_CCS.iloc[i], 'NEXT_PROC_CCS':df_less_n.NEXT_PROC_CCS.iloc[i],}, ignore_index=True)                           
        if len(x)%318>10:
            want=want.append({'TEXT':' '.join(x[-(len(x)%318):]),'Label':df_less_n.OUTPUT_LABEL.iloc[i],'HADM_ID':df_less_n.HADM_ID.iloc[i],
                              'SUBJECT_ID':df_less_n.SUBJECT_ID.iloc[i],'ADMITTIME':df_less_n.ADMITTIME.iloc[i],
                              'DAYS_NEXT_ADMIT':df_less_n.DAYS_NEXT_ADMIT.iloc[i], 'DAYS_PREV_ADMIT':df_less_n.DAYS_PREV_ADMIT.iloc[i], 
                              'DURATION':df_less_n.DURATION.iloc[i], 'NDC':df_less_n.NDC.iloc[i],
                              'DIAG_ICD9':df_less_n.DIAG_ICD9.iloc[i],'DIAG_CCS':df_less_n.DIAG_CCS.iloc[i],
                              'PROC_ICD9':df_less_n.PROC_ICD9.iloc[i],'PROC_CCS':df_less_n.PROC_CCS.iloc[i],
                              'SMALL_DIAG_ICD9':df_less_n.SMALL_DIAG_ICD9.iloc[i],'SMALL_PROC_ICD9':df_less_n.SMALL_PROC_ICD9.iloc[i],
                              'CUI':df_less_n.CUI.iloc[i], 'NEXT_CUI':df_less_n.NEXT_CUI.iloc[i],
                              'NEXT_SMALL_DIAG_ICD9':df_less_n.NEXT_SMALL_DIAG_ICD9.iloc[i], 'NEXT_SMALL_PROC_ICD9':df_less_n.NEXT_SMALL_PROC_ICD9.iloc[i],
                              'NEXT_DIAG_CCS':df_less_n.NEXT_DIAG_CCS.iloc[i], 'NEXT_PROC_CCS':df_less_n.NEXT_PROC_CCS.iloc[i],}, ignore_index=True)                     
    return want

df_discharge = preprocessing(df_discharge)
df_less_2 = preprocessing(df_less_2)
df_less_3 = preprocessing(df_less_3)

print("DATA PREPROCESSED")



### An example to get the train/test/split with random state:
### note that we divide on patient admission level and share among experiments, instead of notes level.
### This way, since our methods run on the same set of admissions, we can see the
### progression of readmission scores.


df_adm.OUTPUT_LABEL.value_counts() #0 - 42358 / 1 - 2963 | Total: 45322 (93,46% vs 6,54%)

# Writing files for the discharge dataset
all_HADM_IDs = df_adm.HADM_ID
from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits = 10, shuffle=True)
i = 0
with open("../data/extended_folds/distributions_hadmsplit.txt", "w") as file:
    for train_index, test_index in skfold.split(X=all_HADM_IDs, y=df_adm.OUTPUT_LABEL):
        file.write("Fold Number {}:\n".format(i))
        print("Fold Number {}:\n".format(i))
        # print(train_index, test_index)
        # print(len(train_index), len(test_index))
        # print(df_adm.HADM_ID[test_index])
        fold_HADM_IDs = all_HADM_IDs[test_index]
        fold_df = df_adm[df_adm.HADM_ID.isin(fold_HADM_IDs)]
        # print(fold_df)
        file.write("Label count considering HADM_IDs:\n")
        file.write(str(fold_df.OUTPUT_LABEL.value_counts())+"\n")
        # print("Label count considering HADM_IDs:")
        # print(fold_df.OUTPUT_LABEL.value_counts())
        discharge_fold = df_discharge[df_discharge.HADM_ID.isin(fold_HADM_IDs)].reset_index(drop=True)
        file.write("Label count considering discharge with text:\n")
        file.write(str(discharge_fold.Label.value_counts())+"\n")
        # print("Label count considering discharge with text:")
        # print(discharge_fold.Label.value_counts())
        filename = '../data/extended_folds/discharge/fold' + str(i) + '_text.csv'
        discharge_fold.to_csv(filename,
                                    columns=["SUBJECT_ID","HADM_ID","ADMITTIME","DAYS_NEXT_ADMIT","DAYS_PREV_ADMIT","DURATION","DIAG_ICD9","SMALL_DIAG_ICD9","DIAG_CCS","PROC_ICD9","SMALL_PROC_ICD9","PROC_CCS","NDC","CUI","Label","TEXT","NEXT_SMALL_DIAG_ICD9","NEXT_DIAG_CCS","NEXT_SMALL_PROC_ICD9","NEXT_PROC_CCS","NEXT_CUI"])
        discharge_fold_notext = discharge_fold.drop("TEXT", axis=1).groupby("HADM_ID").nth(0).reset_index(drop=False)
        filename = '../data/extended_folds/discharge/fold' + str(i) + '_notext.csv'
        discharge_fold_notext.to_csv(filename,
                                    columns=["SUBJECT_ID","HADM_ID","ADMITTIME","DAYS_NEXT_ADMIT","DAYS_PREV_ADMIT","DURATION","DIAG_ICD9","SMALL_DIAG_ICD9","DIAG_CCS","PROC_ICD9","SMALL_PROC_ICD9","PROC_CCS","NDC","CUI","Label","NEXT_SMALL_DIAG_ICD9","NEXT_DIAG_CCS","NEXT_SMALL_PROC_ICD9","NEXT_PROC_CCS","NEXT_CUI"])
        file.write("Label count considering discharge without text:\n")
        file.write(str(discharge_fold_notext.Label.value_counts())+"\n")
        # print("Label count considering discharge without text:")
        # print(discharge_fold_notext.Label.value_counts())
###
        file.write("------Moving from Discharge to Early -------\n")
###
        ### for Early notes experiment: we only need to find training set for 3 days, then we can test
        ### both 3 days and 2 days. Since we split the data on patient level and experiments share admissions
        ### in order to see the progression, the 2 days training dataset is a subset of 3 days training set.
        ### So we only train 3 days and we can test/val on both 2 & 3days or any time smaller than 3 days. This means
        ### if we train on a dataset with all the notes in n days, we can predict readmissions smaller than n days.
###
        # Writing files for the early dataset
        fold_HADM_IDs = all_HADM_IDs[test_index]
        df_less_3_fold = df_less_3[df_less_3.HADM_ID.isin(fold_HADM_IDs)].reset_index(drop=True)
        file.write("Label count considering all early with text:\n")
        file.write(str(df_less_3_fold.Label.value_counts())+"\n")
        # print("Label count considering all early with text")
        # print(df_less_3_fold.Label.value_counts())
###        
#This mechanism below was used to differentiate which samples can be used during test time
###
        df_less_3_fold["TEST_READY_2"] = "No"  
        valid_HADM_IDs = df_less_3_fold[df_less_3_fold['DURATION'] >= 2].HADM_ID
        file.write("Label count considering TEST 2 early days with text:\n")
        file.write(str(df_less_3_fold[df_less_3_fold.HADM_ID.isin(valid_HADM_IDs)].Label.value_counts())+"\n")
        # print("Label count considering TEST 2 early days with text")
        # print(df_less_3_fold[df_less_3_fold.HADM_ID.isin(valid_HADM_IDs)].Label.value_counts())
        df_less_3_fold.loc[df_less_3_fold["DURATION"]>=2, "TEST_READY_2"] = "Yes"
        file.write("Test ready samples considering 2 early days with text:\n")
        file.write(str(df_less_3_fold.TEST_READY_2.value_counts())+"\n")  
        # print("Test ready samples considering 2 early days with text")
        # print(df_less_3_fold.TEST_READY_2.value_counts())  
###
        df_less_3_fold["TEST_READY_3"] = "No"
        valid_HADM_IDs = df_less_3_fold[df_less_3_fold['DURATION'] >= 3].HADM_ID
        file.write("Label count considering TEST 3 early days with text:\n")
        file.write(str(df_less_3_fold[df_less_3_fold.HADM_ID.isin(valid_HADM_IDs)].Label.value_counts())+"\n")
        # print("Label count considering TEST 3 early days with text")
        # print(df_less_3_fold[df_less_3_fold.HADM_ID.isin(valid_HADM_IDs)].Label.value_counts())
        df_less_3_fold.loc[df_less_3_fold["DURATION"]>=3, "TEST_READY_3"] = "Yes"
        file.write("Test ready samples considering 3 early days with text:\n")
        file.write(str(df_less_3_fold.TEST_READY_3.value_counts())+"\n")
        # print("Test ready samples considering 3 early days with text")
        # print(df_less_3_fold.TEST_READY_3.value_counts())  
        filename = '../data/extended_folds/early/fold' + str(i) + '_text.csv'
        df_less_3_fold.to_csv(filename,
                                    columns=["SUBJECT_ID","HADM_ID","TEST_READY_2","TEST_READY_3","ADMITTIME","DAYS_NEXT_ADMIT","DAYS_PREV_ADMIT","DURATION","DIAG_ICD9","SMALL_DIAG_ICD9","DIAG_CCS","PROC_ICD9","SMALL_PROC_ICD9","PROC_CCS","NDC","CUI","Label","TEXT","NEXT_SMALL_DIAG_ICD9","NEXT_DIAG_CCS","NEXT_SMALL_PROC_ICD9","NEXT_PROC_CCS","NEXT_CUI"])
        df_less_3_fold_notext = df_less_3_fold.drop("TEXT", axis=1).groupby("HADM_ID").nth(0).reset_index(drop=False)
        filename = '../data/extended_folds/early/fold' + str(i) + '_notext.csv'
        df_less_3_fold_notext.to_csv(filename,
                                    columns=["SUBJECT_ID","HADM_ID","TEST_READY_2","TEST_READY_3","ADMITTIME","DAYS_NEXT_ADMIT","DAYS_PREV_ADMIT","DURATION","DIAG_ICD9","SMALL_DIAG_ICD9","DIAG_CCS","PROC_ICD9","SMALL_PROC_ICD9","PROC_CCS","NDC","CUI","Label","NEXT_SMALL_DIAG_ICD9","NEXT_DIAG_CCS","NEXT_SMALL_PROC_ICD9","NEXT_PROC_CCS","NEXT_CUI"])
        file.write("Label count considering early without text:\n")
        file.write(str(df_less_3_fold_notext.Label.value_counts())+"\n")
        file.write("Test ready samples considering 2 early days without text:\n")
        file.write(str(df_less_3_fold_notext.TEST_READY_2.value_counts())+"\n")
        file.write("Test ready samples considering 3 early days without text:\n")
        file.write(str(df_less_3_fold_notext.TEST_READY_3.value_counts())+"\n")
        file.write("\n\n\n")
        # print("Label count considering early without text:")
        # print(df_less_3_fold_notext.Label.value_counts())
        # print("Test ready samples considering 2 early days without text")
        # print(df_less_3_fold_notext.TEST_READY_2.value_counts())
        # print("Test ready samples considering 3 early days without text")
        # print(df_less_3_fold_notext.TEST_READY_3.value_counts())
        # print("\n\n")
        i+=1

        
        
#Now splitting by Subject Id and not by Admission Id       
    
all_SUBJECT_IDs = df_adm.groupby(["SUBJECT_ID"]).size().reset_index(name='COUNTS')
SUBJECT_IDs_1visit = all_SUBJECT_IDs[all_SUBJECT_IDs["COUNTS"]==1]        # 28173 subjects
SUBJECT_IDs_multiplevisits = all_SUBJECT_IDs[all_SUBJECT_IDs["COUNTS"]>1] # 6387  subjects

df_subject_label = (df_adm.groupby(['SUBJECT_ID'])
                          .agg({'OUTPUT_LABEL': 'max'})
                          .rename(columns={'OUTPUT_LABEL': 'MAX_LABEL'})
                          .reset_index()
                          )

readmitted_subjects = df_subject_label[df_subject_label["MAX_LABEL"]==1].reset_index(drop=True).drop(["MAX_LABEL"], axis=1)      # 2263 subjects
non_readmitted_subjects = df_subject_label[df_subject_label["MAX_LABEL"]==0].reset_index(drop=True).drop(["MAX_LABEL"], axis=1)  # 32297 subjects

from sklearn.model_selection import KFold
i = 0
readmitted_folds = []
non_readmitted_folds = []

kfold = KFold(n_splits = 10, shuffle=True)
for train_index, test_index in kfold.split(X=readmitted_subjects.SUBJECT_ID):
    readmitted_folds.append(test_index)

for train_index, test_index in kfold.split(X=non_readmitted_subjects.SUBJECT_ID):
    non_readmitted_folds.append(test_index)

with open("../data/extended_folds/distributions_subjectsplit.txt", "w") as file:
    for readmit_index, non_readmit_index in zip(readmitted_folds, non_readmitted_folds):
        file.write("Fold Number {}:\n".format(i))
        print("Fold Number {}:\n".format(i))
        fold_readmit_SUBJECT_IDS = readmitted_subjects.SUBJECT_ID[readmit_index]
        fold_non_readmit_SUBJECT_IDS = non_readmitted_subjects.SUBJECT_ID[non_readmit_index]
        fold_readmit_HADM_IDs = df_adm[df_adm.SUBJECT_ID.isin(fold_readmit_SUBJECT_IDS)].HADM_ID
        fold_non_readmit_HADM_IDs = df_adm[df_adm.SUBJECT_ID.isin(fold_non_readmit_SUBJECT_IDS)].HADM_ID
        fold_HADM_IDs = pd.concat([fold_readmit_HADM_IDs, fold_non_readmit_HADM_IDs])
        #Shuffle the samples up next
        fold_HADM_IDs = fold_HADM_IDs.sample(frac = 1)
        fold_df = df_adm[df_adm.HADM_ID.isin(fold_HADM_IDs)]
        #print(fold_df)
        file.write("Label count considering HADM_IDs:\n")
        file.write(str(fold_df.OUTPUT_LABEL.value_counts())+"\n")
        #print("Label count considering HADM_IDs:")
        #print(fold_df.OUTPUT_LABEL.value_counts())
        discharge_fold = df_discharge[df_discharge.HADM_ID.isin(fold_HADM_IDs)].reset_index(drop=True)
        file.write("Label count considering discharge with text:\n")
        file.write(str(discharge_fold.Label.value_counts())+"\n")
        # print("Label count considering discharge with text:")
        # print(discharge_fold.Label.value_counts())
        filename = '../data/extended_folds/discharge_subjectsplit/fold' + str(i) + '_text.csv'
        discharge_fold.to_csv(filename,
                                    columns=["SUBJECT_ID","HADM_ID","ADMITTIME","DAYS_NEXT_ADMIT","DAYS_PREV_ADMIT","DURATION","DIAG_ICD9","SMALL_DIAG_ICD9","DIAG_CCS","PROC_ICD9","SMALL_PROC_ICD9","PROC_CCS","NDC","CUI","Label","TEXT","NEXT_SMALL_DIAG_ICD9","NEXT_DIAG_CCS","NEXT_SMALL_PROC_ICD9","NEXT_PROC_CCS","NEXT_CUI"])
        discharge_fold_notext = discharge_fold.drop("TEXT", axis=1).groupby("HADM_ID").nth(0).reset_index(drop=False)
        filename = '../data/extended_folds/discharge_subjectsplit/fold' + str(i) + '_notext.csv'
        discharge_fold_notext.to_csv(filename,
                                    columns=["SUBJECT_ID","HADM_ID","ADMITTIME","DAYS_NEXT_ADMIT","DAYS_PREV_ADMIT","DURATION","DIAG_ICD9","SMALL_DIAG_ICD9","DIAG_CCS","PROC_ICD9","SMALL_PROC_ICD9","PROC_CCS","NDC","CUI","Label","NEXT_SMALL_DIAG_ICD9","NEXT_DIAG_CCS","NEXT_SMALL_PROC_ICD9","NEXT_PROC_CCS","NEXT_CUI"])
        file.write("Label count considering discharge without text:\n")
        file.write(str(discharge_fold_notext.Label.value_counts())+"\n")
        # print("Label count considering discharge without text:")
        # print(discharge_fold_notext.Label.value_counts())
    ###
        file.write("------Moving from Discharge to Early -------\n")
    ###
        ### for Early notes experiment: we only need to find training set for 3 days, then we can test
        ### both 3 days and 2 days. Since we split the data on patient level and experiments share admissions
        ### in order to see the progression, the 2 days training dataset is a subset of 3 days training set.
        ### So we only train 3 days and we can test/val on both 2 & 3days or any time smaller than 3 days. This means
        ### if we train on a dataset with all the notes in n days, we can predict readmissions smaller than n days.
    ###
        # Writing files for the early dataset
        df_less_3_fold = df_less_3[df_less_3.HADM_ID.isin(fold_HADM_IDs)].reset_index(drop=True)
        file.write("Label count considering all early with text:\n")
        file.write(str(df_less_3_fold.Label.value_counts())+"\n")
        # print("Label count considering all early with text")
        # print(df_less_3_fold.Label.value_counts())
    ###        
    #This mechanism below was used to differentiate which samples can be used during test time
    ###
        df_less_3_fold["TEST_READY_2"] = "No"  
        valid_HADM_IDs = df_less_3_fold[df_less_3_fold['DURATION'] >= 2].HADM_ID
        file.write("Label count considering TEST 2 early days with text:\n")
        file.write(str(df_less_3_fold[df_less_3_fold.HADM_ID.isin(valid_HADM_IDs)].Label.value_counts())+"\n")
        # print("Label count considering TEST 2 early days with text")
        # print(df_less_3_fold[df_less_3_fold.HADM_ID.isin(valid_HADM_IDs)].Label.value_counts())
        df_less_3_fold.loc[df_less_3_fold["DURATION"]>=2, "TEST_READY_2"] = "Yes"
        file.write("Test ready samples considering 2 early days with text:\n")
        file.write(str(df_less_3_fold.TEST_READY_2.value_counts())+"\n")  
        # print("Test ready samples considering 2 early days with text")
        # print(df_less_3_fold.TEST_READY_2.value_counts())  
    ###
        df_less_3_fold["TEST_READY_3"] = "No"
        valid_HADM_IDs = df_less_3_fold[df_less_3_fold['DURATION'] >= 3].HADM_ID
        file.write("Label count considering TEST 3 early days with text:\n")
        file.write(str(df_less_3_fold[df_less_3_fold.HADM_ID.isin(valid_HADM_IDs)].Label.value_counts())+"\n")
        # print("Label count considering TEST 3 early days with text")
        # print(df_less_3_fold[df_less_3_fold.HADM_ID.isin(valid_HADM_IDs)].Label.value_counts())
        df_less_3_fold.loc[df_less_3_fold["DURATION"]>=3, "TEST_READY_3"] = "Yes"
        file.write("Test ready samples considering 3 early days with text:\n")
        file.write(str(df_less_3_fold.TEST_READY_3.value_counts())+"\n")
        # print("Test ready samples considering 3 early days with text")
        # print(df_less_3_fold.TEST_READY_3.value_counts())  
        filename = '../data/extended_folds/early_subjectsplit/fold' + str(i) + '_text.csv'
        df_less_3_fold.to_csv(filename,
                                    columns=["SUBJECT_ID","HADM_ID","TEST_READY_2","TEST_READY_3","ADMITTIME","DAYS_NEXT_ADMIT","DAYS_PREV_ADMIT","DURATION","DIAG_ICD9","SMALL_DIAG_ICD9","DIAG_CCS","PROC_ICD9","SMALL_PROC_ICD9","PROC_CCS","NDC","CUI","Label","TEXT","NEXT_SMALL_DIAG_ICD9","NEXT_DIAG_CCS","NEXT_SMALL_PROC_ICD9","NEXT_PROC_CCS","NEXT_CUI"])
        df_less_3_fold_notext = df_less_3_fold.drop("TEXT", axis=1).groupby("HADM_ID").nth(0).reset_index(drop=False)
        filename = '../data/extended_folds/early_subjectsplit/fold' + str(i) + '_notext.csv'
        df_less_3_fold_notext.to_csv(filename,
                                    columns=["SUBJECT_ID","HADM_ID","TEST_READY_2","TEST_READY_3","ADMITTIME","DAYS_NEXT_ADMIT","DAYS_PREV_ADMIT","DURATION","DIAG_ICD9","SMALL_DIAG_ICD9","DIAG_CCS","PROC_ICD9","SMALL_PROC_ICD9","PROC_CCS","NDC","CUI","Label","NEXT_SMALL_DIAG_ICD9","NEXT_DIAG_CCS","NEXT_SMALL_PROC_ICD9","NEXT_PROC_CCS","NEXT_CUI"])
        file.write("Label count considering early without text:\n")
        file.write(str(df_less_3_fold_notext.Label.value_counts())+"\n")
        file.write("Test ready samples considering 2 early days without text:\n")
        file.write(str(df_less_3_fold_notext.TEST_READY_2.value_counts())+"\n")
        file.write("Test ready samples considering 3 early days without text:\n")
        file.write(str(df_less_3_fold_notext.TEST_READY_3.value_counts())+"\n")
        file.write("\n\n\n")
        i+=1

