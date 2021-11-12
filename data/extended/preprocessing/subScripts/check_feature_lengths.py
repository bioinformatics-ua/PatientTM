import argparse
import pandas as pd

def processString(string, charsToRemove):
    for char in charsToRemove: string = string.replace(char, "")
    return string

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input file to check.")
    
    args = parser.parse_args()
    print(args.data_dir)
    
    file=pd.read_csv(args.data_dir)
    lines=zip(file.SUBJECT_ID,file.HADM_ID,file.ADMITTIME,file.DAYS_NEXT_ADMIT,file.DAYS_PREV_ADMIT,file.DURATION,file.DIAG_ICD9,
              file.DIAG_CCS,file.PROC_ICD9,file.PROC_CCS,file.NDC,file.SMALL_DIAG_ICD9,file.SMALL_PROC_ICD9,file.CUI,file.Label,file.TEXT)

    ccs_diag_maxlen=0
    ccs_proc_maxlen=0
    cui_maxlen=0
    small_icd_diag_maxlen=0
    small_icd_proc_maxlen=0

    for line in lines:
        small_icd_diag = line[11]  
        if not pd.isna(small_icd_diag):
            small_icd_diag = [x for x in processString(small_icd_diag, charsToRemove = "[]\' ").split(',')]
            if len(small_icd_diag) > small_icd_diag_maxlen: small_icd_diag_maxlen = len(small_icd_diag)
            
        small_icd_proc = line[12]
        if not pd.isna(small_icd_proc):
            small_icd_proc = [x for x in processString(small_icd_proc, charsToRemove = "[]\' ").split(',')]
            if len(small_icd_proc) > small_icd_proc_maxlen: small_icd_proc_maxlen = len(small_icd_proc)
        
        ccs_diag = line[7]
        if not pd.isna(ccs_diag):
            ccs_diag = [x for x in processString(ccs_diag, charsToRemove = "[]\' ").split(',')]
            if len(ccs_diag) > ccs_diag_maxlen: ccs_diag_maxlen = len(ccs_diag)
        
        ccs_proc = line[9]
        if not pd.isna(ccs_proc):
            ccs_proc = [x for x in processString(ccs_proc, charsToRemove = "[]\' ").split(',')]
            if len(ccs_proc) > ccs_proc_maxlen: ccs_proc_maxlen = len(ccs_proc) 
        
        cui = line[13]
        if not pd.isna(cui):
            cui = [x for x in processString(cui, charsToRemove = "[]\' ").split(',')]
            if len(cui) > cui_maxlen: cui_maxlen = len(cui)

    print("Max length of Small ICD9 DIAG: {}".format(small_icd_diag_maxlen))
    print("Max length of Small ICD9 PROC: {}".format(small_icd_proc_maxlen))
    print("Max length of CCS DIAG: {}".format(ccs_diag_maxlen))
    print("Max length of CCS PROC: {}".format(ccs_proc_maxlen))
    print("Max length of CUI: {}".format(cui_maxlen))


# 0 SUBJECT_ID
# 1 HADM_ID
# 2 ADMITTIME
# 3 DAYS_NEXT_ADMIT
# 4 DAYS_PREV_ADMIT
# 5 DURATION
# 6 DIAG_ICD9
# 7 DIAG_CCS,
# 8 PROC_ICD9,
# 9 PROC_CCS,
# 10 NDC,
# 11 SMALL_DIAG_ICD9,
# 12 SMALL_PROC_ICD9,
# 13 CUI,
# 14 Label,
# 15 TEXT)

if __name__ == "__main__":
    main()
