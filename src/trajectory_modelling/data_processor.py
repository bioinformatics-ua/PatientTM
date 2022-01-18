import os
import csv
import json
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def processString(string, charsToRemove):
    for char in charsToRemove: string = string.replace(char, "")
    return string


def processDiagnosisLabel(label_entry):
    label_entry = [x for x in processString(label_entry,charsToRemove = "[] ").split(',')]
    label_entry = np.asarray(label_entry, dtype=np.int)
    return label_entry


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dataset."""
        raise NotImplementedError()
        
    @classmethod
    def _read_csv(cls, input_file):
        """Reads a comma separated value file."""
        raise NotImplementedError()
    
    def _read_precomputed_npy(cls, input_file):
        """Reads a comma separated value file."""
        array = np.load(input_file, allow_pickle=True)
        hadmToFeatureRepresentation = dict()
        for entry in array:
            hadmToFeatureRepresentation[entry["HADM_ID"]] = entry["featureRepresentation"]
        featureDimension = list(hadmToFeatureRepresentation.values())[0].shape[1]
        
        return hadmToFeatureRepresentation, featureDimension

    
class readmissionProcessor(DataProcessor):
    def get_examples(self, data_dir, fold):
        datasetFile = self._read_csv(os.path.join(data_dir, "fold" + str(fold) + "_trajectory_readmission_notext.csv"))
        precomputedFeaturesDict, featureDimension = self._read_precomputed_npy(os.path.join(data_dir, "fold" + str(fold) + "_6visits_readmission_precomputed.npy"))
        return self._create_examples(datasetFile, precomputedFeaturesDict, featureDimension)

    def _read_csv(cls, input_file):
        """Reads a comma separated value file."""
        file=pd.read_csv(input_file)
        lines=zip(file.SUBJECT_ID,file.VISIT_1,file.VISIT_2,file.VISIT_3,file.VISIT_4,file.VISIT_5,file.VISIT_6,file.Label)
        return lines
     
    def _create_examples(self, datasetFile, precomputedFeaturesDict, featureDimension):
        """Creates examples for the training, dev and test sets.
        @param Features is a list with additional variables to be used"""
        examples, lengths, labels = [], [], []
        padding = [0]*featureDimension
                       
        for i, line in enumerate(datasetFile):
            subjectId, *hadmIds, label = line
            numAdmissions = 6 - hadmIds.count(0.0)
            lengths.append(numAdmissions)
            labels.append(label)
            # print(subjectId, hadmIds, label)
            # print(numAdmissions)
            
            example = []
            for idx in range(numAdmissions):
                hadmId = hadmIds[idx]
                featureRepresentation = precomputedFeaturesDict[hadmId][0]
                example.append(featureRepresentation)
# This padding part might not be correct, might need equal dimension 100?
            while len(example) < 6:
                example.append(padding)
            examples.append(example)
        return examples, lengths, labels, featureDimension    
    

class diagnosisProcessor(DataProcessor):        
    def get_examples(self, data_dir, label, fold):
        datasetFile = self._read_csv(os.path.join(data_dir, "fold" + str(fold) + "_trajectory_diagnosis_notext.csv"))
        if label == "LABEL_NEXT_SMALL_DIAG_ICD9":
            numpyDir = os.path.join(data_dir, "fold" + str(fold) + "_6visits_diagnosis_icd_precomputed.npy")
        elif label == "LABEL_NEXT_DIAG_CCS":
            numpyDir = os.path.join(data_dir, "fold" + str(fold) + "_6visits_diagnosis_ccs_precomputed.npy")
        precomputedFeaturesDict, featureDimension = self._read_precomputed_npy(numpyDir)
        return self._create_examples(datasetFile, precomputedFeaturesDict, label, featureDimension)
    
    def _read_csv(cls, input_file):
        """Reads a comma separated value file."""
        file=pd.read_csv(input_file)
        lines=zip(file.SUBJECT_ID,file.VISIT_1,file.VISIT_2,file.VISIT_3,file.VISIT_4,file.VISIT_5,file.VISIT_6,file.LABEL_NEXT_SMALL_DIAG_ICD9, file.LABEL_NEXT_DIAG_CCS)
        return lines
 
     
    def _create_examples(self, datasetFile, precomputedFeaturesDict, label, featureDimension):
        """Creates examples for the training, dev and test sets.
        @param Features is a list with additional variables to be used"""
        examples, lengths, labels = [], [], []
        padding = [0]*featureDimension
                       
        for i, line in enumerate(datasetFile):
            subjectId, *hadmIds, icd_label, ccs_label = line
            numAdmissions = 6 - hadmIds.count(0.0)
            lengths.append(numAdmissions)

            if label == "LABEL_NEXT_SMALL_DIAG_ICD9":
                icd_label = processDiagnosisLabel(icd_label)
                labels.append(icd_label)
            elif label == "LABEL_NEXT_DIAG_CCS":
                ccs_label = processDiagnosisLabel(ccs_label)
                labels.append(ccs_label)
            
            example = []
            for idx in range(numAdmissions):
                hadmId = hadmIds[idx]
                featureRepresentation = precomputedFeaturesDict[hadmId][0]
                example.append(featureRepresentation)
# This padding part might not be correct, might need equal dimension 100?
            while len(example) < 6:
                example.append(padding)
            examples.append(example)
        return examples, lengths, labels, featureDimension   
    
    


    
    


