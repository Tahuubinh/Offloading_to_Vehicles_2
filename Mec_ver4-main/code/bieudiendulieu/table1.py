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
xxx.write("thuoctinh,fuzzy and deep q,deep q learning,mab,fuzzy\n")
x=[i for i in range(0,95162)]


c=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/chiatask_mab.csv")[0:95162]
d=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/chiatask_fuzzy.csv")[0:95162]
"""
a=[]
b=[]
bins=[i*0.1 for i in range(0,21)]
#for i in range(1,6):"""
a=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/chiatask_fdqo.csv")[0:95162]
b=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/chiatask_dql.csv")[0:95162]

#a.append(files3["reward"][0:95162])
#print(files3["reward"].to_numpy())
#print(files4["reward"][0:95162].to_numpy())
#b.append(files4["reward"][0:95162])
#plt.hist(files3["reward"][0:95162],bins=bins, density=True, histtype='step', cumulative=1,label='FCDQL')
#plt.hist(files4["reward"][0:95162],bins=bins, density=True, histtype='step', cumulative=1,label='DQL')
#sns.kdeplot(np.average(a,axis=0),cumulative=True,label='FCDQL',marker='^', markevery=0.2)
#sns.kdeplot(np.average(b,axis=0),cumulative=True,label='EGDQL',marker='o', markevery=0.2)
#plt.hist(files1["reward"][0:95162],bins=bins,cumulative=1, density=True,histtype='step',label='FC')
#plt.hist(files2["reward"][0:95162],bins=bins,cumulative=1, density=True, histtype='step',label='Random')

#plt.ylim(0,1.0)
#plt.xlim(0,1.05)
#plt.xticks([i*0.1 for i in range(0,11)])
#plt.legend()
#plt.show()
#plt.savefig("distributionct.pdf")
xxx.write("may0,\n")
files1=a[a["somay"]==0]
files2=b[b["somay"]==0]
files3=c[c["somay"]==0]
files4=d[d["somay"]==0]
xxx.write("average_time,"+str(np.around(np.average(files1["may0"]),decimals=2))+","+str(np.around(np.average(files2["may0"]),decimals=2))+","\
    +str(np.around(np.average(files3["may0"]),decimals=2))+","+str(np.around(np.average(files4["may0"]),decimals=2))+"\n")
xxx.write("max_time,"+str(np.around(np.max(files1["may0"]),decimals=2))+","+str(np.around(np.max(files2["may0"]),decimals=2))+","\
    +str(np.around(np.max(files3["may0"]),decimals=2))+","+str(np.around(np.max(files4["may0"]),decimals=2))+"\n")
xxx.write("min_time,"+str(np.around(np.min(files1["may0"]),decimals=2))+","+str(np.around(np.min(files2["may0"]),decimals=2))+","\
    +str(np.around(np.min(files3["may0"]),decimals=2))+","+str(np.around(np.min(files4["may0"]),decimals=2))+"\n")
xxx.write("median_time,"+str(np.around(np.median(files1["may0"]),decimals=2))+","+str(np.around(np.median(files2["may0"]),decimals=2))+","\
    +str(np.around(np.median(files3["may0"]),decimals=2))+","+str(np.around(np.median(files4["may0"]),decimals=2))+"\n")
xxx.write("soluong,"+str(np.around((len(files1["may0"])/(95162*1)),decimals=2))+","+str(np.around((len(files2["may0"])/(95162*1)),decimals=2))+","\
    +str(np.around((len(files3["may0"])/(95162)),decimals=2))+","+str(np.around((len(files4["may0"])/(95162)),decimals=2))+"\n")
xxx.write("average_quality,"+str(np.around(np.average(files1["reward"]),decimals=2))+","+str(np.around(np.average(files2["reward"]),decimals=2))+","\
    +str(np.around(np.average(files3["reward"]),decimals=2))+","+str(np.around(np.average(files4["reward"]),decimals=2))+"\n")
xxx.write("var_time,"+str(np.around(np.var(files1["may0"]),decimals=2))+","+str(np.around(np.var(files2["may0"]),decimals=2))+","\
    +str(np.around(np.var(files3["may0"]),decimals=2))+","+str(np.around(np.var(files4["may0"]),decimals=2))+"\n")
xxx.write("may1,\n")
files1=a[a["somay"]==1]
files2=b[b["somay"]==1]
files3=c[c["somay"]==1]
files4=d[d["somay"]==1]
xxx.write("average_time,"+str(np.around(np.average(files1["may1"]),decimals=2))+","+str(np.around(np.average(files2["may1"]),decimals=2))+","\
    +str(np.around(np.average(files3["may1"]),decimals=2))+","+str(np.around(np.average(files4["may1"]),decimals=2))+"\n")
xxx.write("max_time,"+str(np.around(np.max(files1["may1"]),decimals=2))+","+str(np.around(np.max(files2["may1"]),decimals=2))+","\
    +str(np.around(np.max(files3["may1"]),decimals=2))+","+str(np.around(np.max(files4["may1"]),decimals=2))+"\n")
xxx.write("min_time,"+str(np.around(np.min(files1["may1"]),decimals=2))+","+str(np.around(np.min(files2["may1"]),decimals=2))+","\
    +str(np.around(np.min(files3["may1"]),decimals=2))+","+str(np.around(np.min(files4["may1"]),decimals=2))+"\n")
xxx.write("median_time,"+str(np.around(np.median(files1["may1"]),decimals=2))+","+str(np.around(np.median(files2["may1"]),decimals=2))+","\
    +str(np.around(np.median(files3["may1"]),decimals=2))+","+str(np.around(np.median(files4["may1"]),decimals=2))+"\n")
xxx.write("soluong,"+str(np.around((len(files1["may1"])/(95162*1)),decimals=2))+","+str(np.around((len(files2["may1"])/(95162*1)),decimals=2))+","\
    +str(np.around((len(files3["may1"])/(95162)),decimals=2))+","+str(np.around((len(files4["may1"])/(95162)),decimals=2))+"\n")
xxx.write("average_quality,"+str(np.around(np.average(files1["reward"]),decimals=2))+","+str(np.around(np.average(files2["reward"]),decimals=2))+","\
    +str(np.around(np.average(files3["reward"]),decimals=2))+","+str(np.around(np.average(files4["reward"]),decimals=2))+"\n")
xxx.write("var_time,"+str(np.around(np.var(files1["may1"]),decimals=2))+","+str(np.around(np.var(files2["may1"]),decimals=2))+","\
    +str(np.around(np.var(files3["may1"]),decimals=2))+","+str(np.around(np.var(files4["may1"]),decimals=2))+"\n")
xxx.write("may2,\n")
files1=a[a["somay"]==2]
files2=b[b["somay"]==2]
files3=c[c["somay"]==2]
files4=d[d["somay"]==2]
xxx.write("average_time,"+str(np.around(np.average(files1["may2"]),decimals=2))+","+str(np.around(np.average(files2["may2"]),decimals=2))+","\
    +str(np.around(np.average(files3["may2"]),decimals=2))+","+str(np.around(np.average(files4["may2"]),decimals=2))+"\n")
xxx.write("max_time,"+str(np.around(np.max(files1["may2"]),decimals=2))+","+str(np.around(np.max(files2["may2"]),decimals=2))+","\
    +str(np.around(np.max(files3["may2"]),decimals=2))+","+str(np.around(np.max(files4["may2"]),decimals=2))+"\n")
xxx.write("min_time,"+str(np.around(np.min(files1["may2"]),decimals=2))+","+str(np.around(np.min(files2["may2"]),decimals=2))+","\
    +str(np.around(np.min(files3["may2"]),decimals=2))+","+str(np.around(np.min(files4["may2"]),decimals=2))+"\n")
xxx.write("median_time,"+str(np.around(np.median(files1["may2"]),decimals=2))+","+str(np.around(np.median(files2["may2"]),decimals=2))+","\
    +str(np.around(np.median(files3["may2"]),decimals=2))+","+str(np.around(np.median(files4["may2"]),decimals=2))+"\n")
xxx.write("soluong,"+str(np.around((len(files1["may2"])/(95162*1)),decimals=2))+","+str(np.around((len(files2["may2"])/(95162*1)),decimals=2))+","\
    +str(np.around((len(files3["may2"])/(95162)),decimals=2))+","+str(np.around((len(files4["may2"])/(95162)),decimals=2))+"\n")
xxx.write("average_quality,"+str(np.around(np.average(files1["reward"]),decimals=2))+","+str(np.around(np.average(files2["reward"]),decimals=2))+","\
    +str(np.around(np.average(files3["reward"]),decimals=2))+","+str(np.around(np.average(files4["reward"]),decimals=2))+"\n")
xxx.write("var_time,"+str(np.around(np.var(files1["may2"]),decimals=2))+","+str(np.around(np.var(files2["may2"]),decimals=2))+","\
    +str(np.around(np.var(files3["may2"]),decimals=2))+","+str(np.around(np.var(files4["may2"]),decimals=2))+"\n")
xxx.write("may3,\n")
files1=a[a["somay"]==3]
files2=b[b["somay"]==3]
files3=c[c["somay"]==3]
files4=d[d["somay"]==3]
xxx.write("average_time,"+str(np.around(np.average(files1["may3"]),decimals=2))+","+str(np.around(np.average(files2["may3"]),decimals=2))+","\
    +str(np.around(np.average(files3["may3"]),decimals=2))+","+str(np.around(np.average(files4["may3"]),decimals=2))+"\n")
xxx.write("max_time,"+str(np.around(np.max(files1["may3"]),decimals=1))+","+str(np.around(np.max(files2["may3"]),decimals=1))+","\
    +str(np.around(np.max(files3["may3"]),decimals=2))+","+str(np.around(np.max(files4["may3"]),decimals=2))+"\n")
xxx.write("min_time,"+str(np.around(np.min(files1["may2"]),decimals=2))+","+str(np.around(np.min(files2["may3"]),decimals=2))+","\
    +str(np.around(np.min(files3["may3"]),decimals=2))+","+str(np.around(np.min(files4["may3"]),decimals=2))+"\n")
xxx.write("median_time,"+str(np.around(np.median(files1["may3"]),decimals=2))+","+str(np.around(np.median(files2["may3"]),decimals=2))+","\
    +str(np.around(np.median(files3["may3"]),decimals=2))+","+str(np.around(np.median(files4["may3"]),decimals=2))+"\n")
xxx.write("soluong,"+str(np.around((len(files1["may3"])/(95162*1)),decimals=2))+","+str(np.around((len(files2["may3"])/(95162*1)),decimals=2))+","\
    +str(np.around((len(files3["may3"])/(95162)),decimals=2))+","+str(np.around((len(files4["may3"])/(95162)),decimals=2))+"\n")
xxx.write("average_quality,"+str(np.around(np.average(files1["reward"]),decimals=2))+","+str(np.around(np.average(files2["reward"]),decimals=2))+","\
    +str(np.around(np.average(files3["reward"]),decimals=2))+","+str(np.around(np.average(files4["reward"]),decimals=2))+"\n")
xxx.write("var_time,"+str(np.around(np.var(files1["may3"]),decimals=2))+","+str(np.around(np.var(files2["may3"]),decimals=2))+","\
    +str(np.around(np.var(files3["may3"]),decimals=2))+","+str(np.around(np.var(files4["may3"]),decimals=2))+"\n")
