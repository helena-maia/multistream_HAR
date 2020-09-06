import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("output.csv", dtype=str, delimiter="\t")
combination = (data[:,0])[15:]
ucf_rates = data[:,1].astype(float)[15:]
hmdb_rates = data[:,2].astype(float)[15:]
print(ucf_rates.shape, hmdb_rates.shape, combination.shape)

colors = plt.get_cmap('tab10').colors
plt.ylim(bottom=0.15, top=0.45)
plt.xlim(left=15, right=58)
plt.hlines(np.arange(0.2,0.5,0.05), xmin=15, xmax=58, linestyles='dashed', color='lightgray')
plt.yticks(np.arange(0.15,0.5,0.05), fontsize=12)
plt.vlines(np.array([15,35,50,56]), ymin=0.15, ymax=0.5, linestyles='solid', color='lightgray')
plt.xticks(np.arange(16,58,2), fontsize=12)
#plt.xticks(np.array([15,35,50,56]), fontsize=12)
#plt.xticks(np.arange(1,58), combination, rotation='vertical')
plt.ylabel('Interrater agreement', fontsize=12)
plt.xlabel('Combination', fontsize=12)

p = [plt.plot(np.arange(16,58), ucf_rates, color=colors[0], marker='.')[0]]
p.append(plt.plot(np.arange(16,58), hmdb_rates, color=colors[1], marker='.')[0])


legend = plt.legend(p, ["UCF101","HMDB51"], loc="upper right", ncol=1, fontsize=12)
fig = plt.gcf()
fig.set_size_inches(10,5)
fig.savefig("fig.eps", dpi=100, bbox_inches='tight')
