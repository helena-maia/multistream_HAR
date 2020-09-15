import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("fusion_ucf.csv", dtype=str, delimiter="\t")

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
#plt.xticks(np.array([15,35,50,56]), fontsize=12)
plt.xticks(np.arange(1,58,3), fontsize=12)
plt.yticks(np.arange(50,105,10), fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Combination', fontsize=12)

labels2 = ["WA", "FC", "SVM"]
fusion2 = np.array([fusion[2], fusion[0], fusion[1]])

p = []
for i,f in enumerate(fusion2):
	p.append(plt.plot(np.arange(1,58), f, color=colors[i], marker='.')[0])


legend = plt.legend(p, labels2, loc="lower right", title="Method", ncol=1, fontsize=12)
legend.get_title().set_fontsize('12')
fig = plt.gcf()
fig.set_size_inches(10, 5)
fig.savefig("fig.eps", dpi=100, bbox_inches='tight')
