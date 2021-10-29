#!/bin/bash

echo "Fetching ICD9 codes from UMLS\n"
sh subScripts/1_1-getIcd9FromUMLS.sh
echo "Mapping fetched ICD9 codes to their corresponding description string\n"
python3 subScripts/1_2-getICDtextmappings.py

echo "Mapping ICD9 codes to CCS codes and extracting CCS codes corresponding description string\n"
python3 ../../../external_repos/ccs/ccs/icd9_modified_new.py

echo "Fetching NDC codes from UMLS\n"
sh subScripts/2_1-getNdcsFromUMLSdummy.sh 
echo "Mapping fetched NDC codes to their corresponding description UMLS CUI and compiling a list of the CUIs\n"
python3 subScripts/2_2-getNDC_CUImappings.py
echo "Mapping CUI codes to their corresponding description string\n"
python3 ../../../external_repos/UMLS/umls_map.py

echo "Create embeddings for all description strings. Note!!!: this script must be run in an environment with tensorflow+torch+transformers\n"
python3 subScripts/3-getTextEmbeddings.py