import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('arg1_path')
parser.add_argument('arg2_path')
args = parser.parse_args()

arg_file = open(args.arg1_path)
args1 = json.load(arg_file)

arg_file = open(args.arg2_path)
args2 = json.load(arg_file)

diff_items = [ (k,args1[k],args2[k]) for k in args1 if k in args2 and args1[k] != args2[k]]
if(diff_items): 
    print("Different items: ", diff_items)

missing_items1 = [ k for k in args1 if k not in args2 ]
missing_items2 = [ k for k in args2 if k not in args1 ]
if(missing_items1 or missing_items2): 
    print("Missing items: ", missing_items1, missing_items2)
