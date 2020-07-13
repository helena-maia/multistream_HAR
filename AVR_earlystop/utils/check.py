# check if the test script was executed before the final model
# the script should be placed in the scripts/NPY directory

import os
import glob
import json

lista = glob.glob("./*")

for d in lista:
    if os.path.isdir(d):
        args_path = os.path.join(d, "args.json")
        args_status = os.stat(args_path)
        a_last_modif = args_status[-2]

        args_file = open(args_path)
        args_dict = json.load(args_file)

        model_path = args_dict["model_path"]
        model_path = os.path.join("..", model_path)
        model_status = os.stat(model_path)
        m_last_modif = model_status[-2]

        if a_last_modif <= m_last_modif:
            print(d, "Error")
        else:
            print(d, "Ok")


