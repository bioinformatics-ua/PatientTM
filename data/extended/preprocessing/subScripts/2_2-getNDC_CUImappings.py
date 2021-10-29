import json

ndcToCuiMap = {}
cuis = set()

with open("ndcsFromUMLS.txt", "r") as file:
    for line in file:
        cui, _, _, ndc  = line.split("|")
        ndc = ndc.strip("\n")
        ndcToCuiMap[ndc] = cui
        cuis.add(cui)
    ndcToCuiMap["0"]="0"
    
with open("NDCmappings/ndc_cui_map.json", "w") as file:
    json.dump(ndcToCuiMap, file)

with open("NDCmappings/ndc_listofcuis.txt", "w") as file:
    for code in list(cuis):
        file.write(code+"\n")

