## Create config.json (for fusion) from npy_summary.csv
import numpy as np 
import json

data = np.loadtxt("npy_summary.csv", dtype=str, delimiter="\t")

header = data[0]
index_w = np.where(header=="w")[0]
index_timestamp = np.where(header=="timestamp")[0]
index_dataset = np.where(header=="dataset")[0]
index_split = np.where(header=="split")[0]
index_modality = np.where(header=="modality")[0]


data = np.array([d for d in data if d[index_w] == 'True'])
data_dict = {}
data_dict["no_retrain"] = np.array([d for d in data if d[index_timestamp][0].startswith("1593")])
data_dict["retrain"] = np.array([d for d in data if d[index_timestamp][0].startswith("1594")])

config_dict = {}

for k in data_dict:
    aux_dict = {}
    for d in data_dict[k]:
        split = d[index_split][0]
        dataset = d[index_dataset][0]
        modality = d[index_modality][0]
        timestamp = d[index_timestamp][0]

        key = "{}_s{}".format(dataset, split)
        if key not in aux_dict:
            aux_dict.update({key:{}})
        
        aux_dict[key].update({modality:timestamp})

    config_dict[k] = aux_dict

with open("config.json", 'w') as json_file:
    json.dump(config_dict, json_file, sort_keys=True, indent=4, separators=(',', ': '))






