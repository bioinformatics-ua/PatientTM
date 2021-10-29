import json
import numpy
import torch
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertModel


SAPBERT = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'

print("Load Tokenizer")
tokenizer = BertTokenizerFast.from_pretrained(SAPBERT)

print("Load BERT")
bert_model = TFBertModel.from_pretrained(SAPBERT,
                                         output_attentions = False,
                                         output_hidden_states = False,
                                         return_dict=True,
                                         from_pt=True)
@tf.function
def get_cls(**kwargs):
    return bert_model(kwargs)["last_hidden_state"][:,0,:] # NONE, 512, 768

def get_text_embedding(descriptor):
    encoding = tokenizer.encode_plus(descriptor,
                                  max_length = 25,
                                  padding="max_length",
                                  return_token_type_ids = True,
                                  return_attention_mask = True,
                                  return_tensors = "pt")
    return get_cls(**encoding)


def get_embedding_layer(text_map, idx_map):
    embeddings_list=[]
    for code in sorted(idx_map, key=idx_map.get):
        # print(code, idx_map[code])
        embedding = get_text_embedding(text_map[code])
        embeddings_list.append(embedding)

    tensor = tf.concat(embeddings_list,axis=0)
    return tensor.numpy()
         
    
print("Processing CCS mentions")
with open("ICDandCCSmappings/merged_ccs_text.json","r") as file:
    mentions = json.load(file)
    
with open("idxFiles/CCSToIdx.json","r") as file:
    idx = json.load(file)
    
ccs_tensor = get_embedding_layer(mentions, idx)
numpy.save("embeddings/ccs_embeddings.npy", ccs_tensor)


print("Processing small ICD9 mentions")
with open("ICDandCCSmappings/merged_simplified_icd_text.json","r") as file:
    mentions = json.load(file)
    
with open("idxFiles/smallIcd9ToIdx.json","r") as file:
    idx = json.load(file)
    
small_icd9_tensor = get_embedding_layer(mentions, idx)
numpy.save("embeddings/small_icd9_embeddings.npy", small_icd9_tensor)


print("Processing ICD9 mentions")
with open("ICDandCCSmappings/merged_icd_text.json","r") as file:
    mentions = json.load(file)
    
with open("idxFiles/Icd9ToIdx.json","r") as file:
    idx = json.load(file)
    
icd9_tensor = get_embedding_layer(mentions, idx)
numpy.save("embeddings/icd9_embeddings.npy", icd9_tensor)


print("Processing CUI mentions")
with open("NDCmappings/cui_text.json","r") as file:
    mentions = json.load(file)
    
with open("idxFiles/cui_NDCToIdx.json","r") as file:
    idx = json.load(file)
    
cui_tensor = get_embedding_layer(mentions, idx)
numpy.save("embeddings/cui_embeddings.npy", cui_tensor)


# array = numpy.load("embeddings/ccs_embeddings.npy")
# print(array.shape, array.dtype)

# array = numpy.load("embeddings/small_icd9_embeddings.npy")
# print(array.shape, array.dtype)

# array = numpy.load("embeddings/icd9_embeddings.npy")
# print(array.shape, array.dtype)

# array = numpy.load("embeddings/cui_embeddings.npy")
# print(array.shape, array.dtype)