import os
import csv
import json
import logging
import pandas as pd

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, features, text_b=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            features: dictionary containing the following features: ADMITTIME,DURATION,DIAG_ICD9,DIAG_CCS,PROC_ICD9,PROC_CCS,NDC,Label,TEXT.
            TEXT contains the untokenized text of the first sequence. For single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.guid = guid
        self.text_a = features["text_a"]
        self.label = features["label"]
        self.text_b = text_b
        self.hadm_id = features["hadm_id"]
        self.admittime = None
        self.daystonextadmit = None
        self.daystoprevadmit = None
        self.duration = None
        self.small_diag_icd9 = None
        self.diag_ccs = None
        self.small_proc_icd9 = None
        self.proc_ccs = None
        self.cui = None

        if features["admittime"] is not None:       self.admittime = features["admittime"]
        if features["daystonextadmit"] is not None: self.daystonextadmit = features["daystonextadmit"]
        if features["daystoprevadmit"] is not None: self.daystoprevadmit = features["daystoprevadmit"]
        if features["duration"] is not None:        self.duration = features["duration"]
        if features["small_diag_icd9"] is not None: self.small_diag_icd9 = features["small_diag_icd9"]
        if features["diag_ccs"] is not None:        self.diag_ccs = features["diag_ccs"]
        if features["small_proc_icd9"] is not None: self.small_proc_icd9 = features["small_proc_icd9"]       
        if features["proc_ccs"] is not None:        self.proc_ccs = features["proc_ccs"]
        if features["cui"] is not None:             self.cui = features["cui"]


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, features):
        if "input_ids" in features.keys():
            self.input_ids = features["input_ids"]
            self.input_mask = features["input_mask"]
            self.segment_ids = features["segment_ids"]
        else:
            self.input_ids = None
            self.input_mask = None
            self.segment_ids = None
        
        self.label_id = features["label_id"]
        self.hadm_id = features["hadm_id"]    
        
        if "admittime" in features.keys():       self.admittime = features["admittime"]
        if "daystonextadmit" in features.keys(): self.daystonextadmit = features["daystonextadmit"]
        if "daystoprevadmit" in features.keys(): self.daystoprevadmit = features["daystoprevadmit"]
        if "duration" in features.keys():        self.duration = features["duration"]
        if "small_diag_icd9" in features.keys(): self.small_diag_icd9 = features["small_diag_icd9"]
        if "diag_ccs" in features.keys():        self.diag_ccs = features["diag_ccs"]
        if "small_proc_icd9" in features.keys(): self.small_proc_icd9 = features["small_proc_icd9"]  
        if "proc_ccs" in features.keys():        self.proc_ccs = features["proc_ccs"]
        if "cui" in features.keys():             self.cui = features["cui"]


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set. This can vary according to the predictive goal in mind (binary, multiclass, regression)"""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        
    @classmethod
    def _read_csv(cls, input_file):
        """Reads a comma separated value file."""
        file=pd.read_csv(input_file)
        lines=zip(file.SUBJECT_ID,file.HADM_ID,file.ADMITTIME,file.DAYS_NEXT_ADMIT,file.DAYS_PREV_ADMIT,file.DURATION,file.DIAG_ICD9,\
                  file.DIAG_CCS,file.PROC_ICD9,file.PROC_CCS,file.NDC,file.SMALL_DIAG_ICD9,file.SMALL_PROC_ICD9,file.CUI,file.Label,file.TEXT)
        return lines

class readmissionProcessor(DataProcessor):
    def get_train_examples(self, data_dir, features=None):
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.csv")))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train", features)
    
    def get_dev_examples(self, data_dir, features=None):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "val.csv")), "val", features)
    
    def get_test_examples(self, data_dir, features=None):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test", features)
    
    def get_labels(self):
        """This is only 0 or 1 for readmission prediction. Other predictive goals may need different labels"""
        return ["0", "1"]
    
    def _create_examples(self, lines, set_type, Features=None):
        """Creates examples for the training, dev and test sets.
        @param Features is a list with additional variables to be used"""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            features = dict()
            if Features is not None:
                if "admittime" in Features: features["admittime"] = [line[2]]
                else: features["admittime"] = None

                if "daystonextadmit" in Features:
                    features["daystonextadmit"] = line[3]
                    if pd.isna(features["daystonextadmit"]): features["daystonextadmit"] = [-1]
                    else: features["daystonextadmit"] = [float(line[3])]
                else: features["daystonextadmit"] = None
                
                if "daystoprevadmit" in Features:
                    features["daystoprevadmit"] = line[4]
                    if pd.isna(features["daystoprevadmit"]): features["daystoprevadmit"] = [float(32000)]
                    elif  features["daystoprevadmit"] < 0: features["daystoprevadmit"] = [float(32000)]
                    else: features["daystoprevadmit"] = [float(line[4])]
                else: features["daystoprevadmit"] = None

                if "duration"  in Features: features["duration"] = [float(line[5])]
                else: features["duration"] = None

                if "diag_ccs"  in Features:
                    features["diag_ccs"] = line[7]
                    if pd.isna(features["diag_ccs"]) or features["diag_ccs"] == "[]": features["diag_ccs"] = [0]
                    else: features["diag_ccs"] = [x for x in processString(line[7],charsToRemove = "[]\' ").split(',')]
                else: features["diag_ccs"] = None

                if "proc_ccs"  in Features:
                    features["proc_ccs"] = line[9]
                    if pd.isna(features["proc_ccs"]) or features["proc_ccs"] == "[]": features["proc_ccs"] = [0]
                    else: features["proc_ccs"] = [x for x in processString(line[9],charsToRemove = "[]\' ").split(',')]
                else: features["proc_ccs"] = None
                
                if "small_diag_icd9" in Features:
                    features["small_diag_icd9"] = line[11]
                    if pd.isna(features["small_diag_icd9"]) or features["small_diag_icd9"] == "[]": features["small_diag_icd9"] = [0]
                    else: features["small_diag_icd9"] = [x for x in processString(line[11],charsToRemove = "[]\' ").split(',')]
                else: features["small_diag_icd9"] = None
                
                if "small_proc_icd9" in Features:
                    features["small_proc_icd9"] = line[12]
                    if pd.isna(features["small_proc_icd9"]) or features["small_proc_icd9"] == "[]": features["small_proc_icd9"] = [0]
                    else: features["small_proc_icd9"] = [x for x in processString(line[12],charsToRemove = "[]\' ").split(',')]
                else: features["small_proc_icd9"] = None
                
                if "cui" in Features:
                    features["cui"] = line[13]                    
                    if pd.isna(features["cui"]) or features["cui"] == "[]": features["cui"] = [0]
                    else: features["cui"] = [x for x in processString(line[13],charsToRemove = "[]\' ").split(',')]
                else: features["cui"] = None
                
                
            else:
                features["admittime"] = None
                features["daystonextadmit"] = None
                features["daystoprevadmit"] = None
                features["duration"] = None
                features["diag_ccs"] = None
                features["proc_ccs"] = None
                features["small_diag_icd9"] = None
                features["small_proc_icd9"] = None
                features["cui"] = None

            features["hadm_id"] = line[1]
            features["label"] = str(int(line[14]))
            features["text_a"] = line[15]

            examples.append(
                InputExample(guid=guid, features=features, text_b=None))
        return examples


def processString(string, charsToRemove):
    for char in charsToRemove: string = string.replace(char, "")
    return string


def convertToIds(feature, idsMappingDict):
    # featureIds=[]
    # for entry in feature:
    #     featureIds.append(idsMappingDict[str(entry)])
    #     print(entry)
    # return featureIds
    return [idsMappingDict[str(entry)] for entry in feature]


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, Features, maxLenDict):
    """Loads a data file into a list of `InputBatch`s."""

    if Features is not None:
        """Load the mapping dictionaries for additional features, to be used later to convert to ids"""
        if "small_diag_icd9" in Features or "small_proc_icd9" in Features:
            with open("../data/extended/preprocessing/idxFiles/smallIcd9ToIdx.json","r") as file:
                smallIcd9MappingDict = json.load(file)

        if "diag_ccs" in Features or "proc_ccs" in Features:
            with open("../data/extended/preprocessing/idxFiles/CCSToIdx.json","r") as file:
                ccsMappingDict = json.load(file)

        if "cui" in Features:
            with open("../data/extended/preprocessing/idxFiles/cui_NDCToIdx.json","r") as file:
                cuiMappingDict = json.load(file)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        feature = dict()
        
        if "clinical_text" in Features:
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[0:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            #print (example.label)
            label_id = label_map[example.label]

            if ex_index < 2:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label_id))

            feature["input_ids"]=input_ids
            feature["input_mask"]=input_mask
            feature["segment_ids"]=segment_ids
        
        label_id = label_map[example.label]
        feature["label_id"]=label_id
        feature["hadm_id"]=example.hadm_id

        if "admittime" in Features: feature["admittime"] = example.admittime
        if "daystonextadmit" in Features: feature["daystonextadmit"] = example.daystonextadmit
        if "daystoprevadmit" in Features: feature["daystoprevadmit"] = example.daystoprevadmit
        if "duration"  in Features: feature["duration"] = example.duration
        if "diag_ccs"  in Features:
            feature["diag_ccs"] = convertToIds(example.diag_ccs, ccsMappingDict)
            while len(feature["diag_ccs"]) < maxLenDict["small_icd9_ccs_maxlen"]:
                feature["diag_ccs"].append(0)
        if "proc_ccs"  in Features:
            feature["proc_ccs"] = convertToIds(example.proc_ccs, ccsMappingDict)
            while len(feature["proc_ccs"]) < maxLenDict["small_icd9_ccs_maxlen"]:
                feature["proc_ccs"].append(0)
        if "small_diag_icd9" in Features:
            feature["small_diag_icd9"] = convertToIds(example.small_diag_icd9, smallIcd9MappingDict)
            while len(feature["small_diag_icd9"]) < maxLenDict["small_icd9_ccs_maxlen"]:
                feature["small_diag_icd9"].append(0)
        if "small_proc_icd9" in Features:
            feature["small_proc_icd9"] = convertToIds(example.small_proc_icd9, smallIcd9MappingDict)
            while len(feature["small_proc_icd9"]) < maxLenDict["small_icd9_ccs_maxlen"]:
                feature["small_proc_icd9"].append(0)
        if "cui" in Features:
            feature["cui"] = convertToIds(example.cui, cuiMappingDict)
            while len(feature["cui"]) < maxLenDict["cui_maxlen"]:
                feature["cui"].append(0)

        features.append(InputFeatures(feature))
    return features
