import json

codeTextMap = {}
codeTextMap[0] = ""
simplifiedCodeTextMap = {}
simplifiedCodeTextMap[0] = ""

with open("icd9FromUMLS.txt", "r") as file:
    for line in file:
        dictionary, textType, code, text = line.split("|")
        text = text.rstrip()
        
        if "-" in code:
            pass
        else:
            codeParcels = code.split(".")
            
            ##Confirm that we are dealing with the real base code and not a child code
            if len(codeParcels[0]) == len(code):                
            #Firstly we create the mapping for a "simplified" ICD hierarchy where only the first level is used
            #Procedure codes go from 00-99, diagnosis codes go from 000-999 and include E000-E999 and V00-V99)
                if codeParcels[0].startswith("E") or codeParcels[0].startswith("V"):
                    smallCode = "D_" + codeParcels[0]
                elif len(codeParcels[0]) == 3:
                    smallCode = "D_" + codeParcels[0]
                elif len(codeParcels[0]) == 2:
                    smallCode = "P_" + codeParcels[0]

                if smallCode not in simplifiedCodeTextMap.keys():
                    simplifiedCodeTextMap[smallCode] = text
                else:
                    prev_text = simplifiedCodeTextMap[smallCode]
                    ## If the previous mention is the abbreviated one
                    if len(prev_text) <= len(text):
                        ## If the abbreviation is part of complete mention, only the complete one is saved
                        if prev_text in text:
                            simplifiedCodeTextMap[smallCode] = text
                        ## If it is not part, both strings are concatenated
                        else:
                            simplifiedCodeTextMap[smallCode] = prev_text + ", " + text
                    ## If the previous mention is the complete one
                    else:
                        ## If the abbreviation is part of complete mention, the complete one is kept
                        if text in prev_text:
                            pass
                        ## If the abbreviation is not part of the complete mention, both strings are concatenated
                        else:
                            simplifiedCodeTextMap[smallCode] = prev_text + ", " + text

            
        #If the first segment has less than 3 digits, it corresponds to a procedure code and not a diagnosis one. We also remove dots to standardize
        #with the icd-ccs mappings     
            if len(codeParcels[0]) < 3:
                code = "P_" + code.replace(".", "")
            else:
                code = "D_" + code.replace(".", "")
                            
            if code not in codeTextMap.keys():
                codeTextMap[code] = text
            else:
                prev_text = codeTextMap[code]
                ## If the previous mention is the abbreviated one
                if len(prev_text) <= len(text):
                    ## If the abbreviation is part of complete mention, only the complete one is saved
                    if prev_text in text:
                        codeTextMap[code] = text
                    ## If it is not part, both strings are concatenated
                    else:
                        codeTextMap[code] = prev_text + ", " + text
                ## If the previous mention is the complete one
                else:
                    ## If the abbreviation is part of complete mention, the complete one is kept
                    if text in prev_text:
                        pass
                    ## If the abbreviation is not part of the complete mention, both strings are concatenated
                    else:
                        codeTextMap[code] = prev_text + ", " + text
                
                # prev_text = codeTextMap.pop(code, None)
                # codeTextMap[code] = prev_text + ", " + text

with open("ICDandCCSmappings/merged_icd_text.json", "w") as file:
    json.dump(codeTextMap, file)
    
with open("ICDandCCSmappings/merged_simplified_icd_text.json", "w") as file:
    json.dump(simplifiedCodeTextMap, file)
