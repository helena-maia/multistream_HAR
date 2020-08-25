import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("fusion_hmdb.csv", dtype=str, delimiter="\t")

labels = data[0,1:]
labels = np.delete(labels, 2, axis=0)
labels[labels==""]

fusion = data[1:, 1:].astype(float)
fusion = np.delete(fusion, 2, axis=1).transpose()*100.

colors = plt.get_cmap('tab10').colors
plt.ylim(bottom=45, top=100)
plt.xlim(left=0, right=58)
plt.hlines(np.arange(50,100,5), xmin=0, xmax=58, linestyles='dashed', color='lightgray')
plt.vlines(np.array([15,35,50,56]), ymin=45, ymax=100, linestyles='solid', color='lightgray')
plt.xticks(np.array([15,35,50,56]), fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.xlabel('Combination', fontsize=12)

p = []
for i,f in enumerate(fusion):
	p.append(plt.plot(np.arange(1,58), f, color=colors[i])[0])


legend = plt.legend(p, labels, loc="upper center", bbox_to_anchor=(0.5, 1.2), title="Method", ncol=3, fontsize=12)
legend.get_title().set_fontsize('12')
fig = plt.gcf()
fig.set_size_inches(10, 5)
fig.savefig("fig.eps", dpi=100, bbox_inches='tight')
