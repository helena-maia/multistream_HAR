from fusion import fusion, methods
import argparse
import itertools

checkpoint_dict = {}


# checkpoint_dict['ucf101_s1'] = {'rhythm': '1593559102.0596259',
#                             'rgb': '1593621990.2010984',
#                             'flow': '1593661622.8446941'}

# checkpoint_dict['hmdb51_s1'] = {'rhythm': '1593559473.2116444',
#                              'rgb': '1593562222.21443',
#                              'flow': '1593710997.4206576'}

# checkpoint_dict['ucf101_s2'] = {'rhythm': '1593940571.1122901',
#                                 'rgb': '1593992611.6218164',
#                                 'flow': '1593993145.2747655'}

# checkpoint_dict['hmdb51_s2'] = {'rhythm': '1593940916.9126217',
#                                 'rgb': '1593992753.7957015',
#                                 'flow': '1593993271.2409515'}

# checkpoint_dict['ucf101_s3'] = {'rhythm': '1593944372.6974711',
#                                 'rgb': '1593992837.1588902',
#                                 'flow': '1593993396.2428837'}

# checkpoint_dict['hmdb51_s3'] = {'rhythm': '1593944268.1883364',
#                                 'rgb': '1593992961.2348812',
#                                 'flow': '1593993491.3737884'}

dataset = ['ucf101','hmdb51']
npy_path_fmt = "../scripts/NPYS/{}/{}_{}_inception_v3_s{}.npy"
splits = [1, 2, 3]
settings = "../datasets/settings_earlystop"
#modalities = ["rhythm", "rgb", "flow"]
modalities = ["rgb", "flow"]
#methods.remove("SVM")
#methods.remove("sugeno_fuzzy")
methods = ["SVM"]


n_modalities = len(modalities)
comb = None
for r in range(2, n_modalities+1):
    c = itertools.combinations(modalities, r=r)
    comb = itertools.chain(comb, c) if comb else c

it = itertools.product(dataset, splits, methods, comb)

with open("output2.csv", "w") as f:
    f.write("dataset \t split \t method \t combination \t prec\n")

for d, s, m, c in it:
    key = "{}_s{}".format(d,s)
    
    npy_paths = []
    for mod in c:
        npy_path = npy_path_fmt.format(checkpoint_dict[key][mod],
                                       d, mod, s)
        npy_paths.append(npy_path)

    args = argparse.Namespace(d=d, m=m, npy_paths=npy_paths, 
                              s=s, settings=settings)

    print(args)
    prec = fusion(args)
    
    with open("output2.csv", "a") as f:
        f.write("{}\t{}\t{}\t{}\t{:04f}\n".format(d, s, m, c, prec))
