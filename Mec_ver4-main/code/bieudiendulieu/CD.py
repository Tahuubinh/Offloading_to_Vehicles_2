import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import table
import seaborn as sns
import  math
sns.set_style("white")
"""
ax=plt.figure(figsize=(15,10))

ax1=ax.add_subplot(2,2,1)
ax2=ax.add_subplot(2,2,2)
ax3=ax.add_subplot(2,2,3)
ax4=ax.add_subplot(2,2,4)"""
xxx=open("because.csv","w")
xxx.write("thuoctinh,fuzzy and deep q,deep q learning,fuzzy,random\n")
x=[i for i in range(0,161218)]


c=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/result4/fuzzy/chiatask.csv")[0:124924]
d=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/result4/random/chiatask.csv")[0:124924]

a=[]
b=[]
bins=[i*0.1 for i in range(0,21)]
for i in range(1,6):
    # files3=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/result4/Combine fuzzy and deep q "+str(i)+"/chiatask.csv")[0:124924]
    # a=np.concatenate((a,files3["reward"].to_numpy()), axis=0)
    files4=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/result4/deep q learning "+str(i)+"/chiatask.csv")[0:124924]
    b=np.concatenate((b,files4["reward"].to_numpy()), axis=0)
#plt.hist(files3["reward"][0:124924],bins=bins, density=True, histtype='step', cumulative=1,label='FCDQL')
#plt.hist(files4["reward"][0:124924],bins=bins, density=True, histtype='step', cumulative=1,label='DQL')
# print("a")
# sns.kdeplot(a,cumulative=True,label='FCDQL',marker='^', markevery=0.2)
# print("b")
sns.kdeplot(b,cumulative=True,label='EGDQL',marker='o', markevery=0.2)

#sns.kdeplot(c,cumulative=True,label='EGDQL',marker='o', markevery=0.2)
#sns.kdeplot(d,cumulative=True,label='EGDQL',marker='o', markevery=0.2)
#plt.hist(files1["reward"][0:124924],bins=bins,cumulative=1, density=True,histtype='step',label='FC')
#plt.hist(files2["reward"][0:124924],bins=bins,cumulative=1, density=True, histtype='step',label='Random')

plt.ylim(0,1.0)
plt.xlim(0,1.05)
plt.xticks([i*0.1 for i in range(0,11)])
plt.legend()
plt.savefig("distributionct2.pdf")
print("h")
plt.show()
#plt.savefig("distributionct.pdf")
