import numpy as np

fusion = np.loadtxt("fusion.csv", dtype=str, delimiter="\t")
method = np.unique(fusion[:,0])
datasets = np.unique(fusion[:,1])
comb = np.loadtxt("comb_ordem.txt", dtype=str, delimiter="\n")
print(comb, fusion[0][3])

matrix = np.zeros((2,57,4))


for i in range(2):
    for j in range(57):
        for k in range(4):
             d = fusion[:,1]
             f1 = fusion[d==datasets[i]]
             m = f1[:,0]
             f2 = f1[m == method[k]]
             c = f2[:,3]
             f3 = f2[c == comb[j]]

             if len(f3) == 1: 
                 matrix[i][j][k] = f3[0][-1]

saida1 = [[""]+list(method)]
saida1+= [ [comb[i]]+list(row) for i,row in enumerate(matrix[0])]
saida1 = np.array(saida1)

saida2 = [[""]+list(method)]
saida2+= [ [comb[i]]+list(row) for i,row in enumerate(matrix[1])]
saida2 = np.array(saida2)

print(saida2.shape)

np.savetxt("fusion_hmdb.csv", saida1, fmt="%s", delimiter="\t")
np.savetxt("fusion_ucf.csv", saida2, fmt="%s", delimiter="\t")
