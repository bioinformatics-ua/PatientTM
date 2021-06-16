import pandas as pd
import numpy as np
import pickle as pickle
import sys
import matplotlib.pyplot as plt


df_adm = pd.read_csv('/backup/mimiciii/ADMISSIONS.csv.gz', compression="gzip")
df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])
df_adm = df_adm.reset_index(drop = True)
df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)

rows = df_adm.NEXT_ADMISSION_TYPE == 'ELECTIVE'
df_adm.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
df_adm.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])

#When we filter out the "ELECTIVE", we need to correct the next admit time for these admissions since there might be 'emergency' next admit after "ELECTIVE"
df_adm[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')
df_adm['DAYS_NEXT_ADMIT']=  (df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME).dt.total_seconds()/(24*60*60)
df_adm['OUTPUT_LABEL'] = (df_adm.DAYS_NEXT_ADMIT < 30).astype('int')
### filter out newborn and death
df_adm = df_adm[df_adm['ADMISSION_TYPE']!='NEWBORN']
df_adm = df_adm[df_adm.DEATHTIME.isnull()]
df_adm['DURATION'] = (df_adm['DISCHTIME']-df_adm['ADMITTIME']).dt.total_seconds()/(24*60*60)

##THIS IS NEW, BRINGS ICD9 CODES, CCS CODES AND MEDICATION

def map_ICD9_to_CCS(pandasDataFrame):
    #icd9TOCCS_Map = pickle.load(open('./icd9_to_css_dictionary','rb'))
    icd9TOCCS_Map = pickle.load(open('./icd9_to_ccs_dict.pkl','rb'))
    mappedCCSList = []
    set_of_used_codes = set()
    unmapped=0
    for (hadm_id, ICD9s_List) in pandasDataFrame.values:
        tempCodeList = []
        for ICD9 in ICD9s_List:
            while (len(ICD9) < 6): ICD9 += ' '  #pad right white spaces because the CCS mapping uses this pattern
            try:
                CCS_code = icd9TOCCS_Map[ICD9]
                tempCodeList.append(CCS_code)
                set_of_used_codes.add(ICD9)
            except KeyError:
                tempCodeList.append("nan")
                print(str(sys.exc_info()[0]) + '  ' + str(ICD9) + ". ICD9 code not found, please verify your ICD9 to CCS mapping before proceeding.")
                unmapped+=1
        mappedCCSList.append(tempCodeList)
    print('-Total number (complete set) of ICD9 codes (diag + proc): ' + str(len(set(icd9TOCCS_Map.keys()))))
    print('-Total number (complete set) of CCS codes (diag + proc): ' + str(len(set(icd9TOCCS_Map.values()))))
    print('-Total number of ICD9 codes actually used: ' + str(len(set_of_used_codes)))
    print("Unmapped codes {}".format(unmapped))
    return mappedCCSList

df_diagnoses = pd.read_csv('/backup/mimiciii/DIAGNOSES_ICD.csv.gz', compression="gzip")
df_diagnoses = df_diagnoses.sort_values(['HADM_ID','SEQ_NUM'], ascending=True)
df_diagnoses = df_diagnoses.reset_index(drop = True)
df_diag_listing = df_diagnoses[['HADM_ID', 'ICD9_CODE']]
df_diag_listing['ICD9_CODE'] = 'D' + df_diag_listing['ICD9_CODE'].astype(str)
df_diag_listing = df_diag_listing.groupby('HADM_ID')['ICD9_CODE'].apply(list)
df_diag_listing = df_diag_listing.reset_index()
diagnosesCCS = map_ICD9_to_CCS(df_diag_listing)
df_diag_listing['DIAG_CCS'] = diagnosesCCS

df_adm = pd.merge(df_adm,
                  df_diag_listing[['HADM_ID','ICD9_CODE','DIAG_CCS']],
                  on = ['HADM_ID'],
                  how = 'left')

df_adm = df_adm.rename(columns={'ICD9_CODE': 'DIAG_ICD9'})

df_procedures = pd.read_csv('/backup/mimiciii/PROCEDURES_ICD.csv.gz', compression="gzip")
df_procedures = df_procedures.sort_values(['HADM_ID','SEQ_NUM'], ascending=True)
df_procedures = df_procedures.reset_index(drop = True)
df_proc_listing = df_procedures[['HADM_ID', 'ICD9_CODE']]
df_proc_listing['ICD9_CODE'] = 'P' + df_proc_listing['ICD9_CODE'].astype(str)
df_proc_listing = df_proc_listing.groupby('HADM_ID')['ICD9_CODE'].apply(list)
df_proc_listing = df_proc_listing.reset_index()
proceduresCCS = map_ICD9_to_CCS(df_proc_listing)
df_proc_listing['PROC_CCS'] = proceduresCCS

df_adm = pd.merge(df_adm,
                  df_proc_listing[['HADM_ID','ICD9_CODE','PROC_CCS']],
                  on = ['HADM_ID'],
                  how = 'left')

df_adm = df_adm.rename(columns={'ICD9_CODE': 'PROC_ICD9'})

df_medication = pd.read_csv('/backup/mimiciii/PRESCRIPTIONS.csv.gz', compression="gzip")
df_medication = df_medication.sort_values(['HADM_ID'], ascending=True)
df_medication = df_medication.reset_index(drop = True)
df_med_listing = df_medication[['HADM_ID','DRUG']]
df_med_listing = df_med_listing.groupby('HADM_ID')['DRUG'].apply(list)
df_med_listing = df_med_listing.reset_index()

df_adm = pd.merge(df_adm,
                  df_med_listing[['HADM_ID','DRUG']],
                  on = ['HADM_ID'],
                  how = 'left')

df_notes = pd.read_csv('/backup/mimiciii/NOTEEVENTS.csv.gz', compression="gzip")
df_notes = df_notes.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTDATE'])
df_adm_notes = pd.merge(df_adm[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT','NEXT_ADMITTIME','ADMISSION_TYPE',
                                'DEATHTIME','OUTPUT_LABEL','DURATION','DIAG_ICD9','PROC_ICD9','DRUG']],
                        df_notes[['SUBJECT_ID','HADM_ID','CHARTDATE','TEXT','CATEGORY']],
                        on = ['SUBJECT_ID','HADM_ID'],
                        how = 'left')

df_adm_notes.ADMITTIME_C = df_adm_notes.ADMITTIME.apply(lambda x: str(x).split(' ')[0])
df_adm_notes['ADMITTIME_C'] = pd.to_datetime(df_adm_notes.ADMITTIME_C, format = '%Y-%m-%d', errors = 'coerce')
df_adm_notes['CHARTDATE'] = pd.to_datetime(df_adm_notes.CHARTDATE, format = '%Y-%m-%d', errors = 'coerce')