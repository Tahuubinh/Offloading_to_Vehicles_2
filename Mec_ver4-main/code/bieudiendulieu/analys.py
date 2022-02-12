import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import table
import seaborn as sns
import  math
sns.set_style("white")

xxx=open("max-min.csv","w")
xxx.write("thuoctinh,fuzzy and deep q,deep q learning,MAB,fuzzy\n")
x= 104838


c=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/MAB_5phut.csv")[0:100]
d=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/Fuzzy_5phut.csv")[0:100]

a=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/FDQO_5phut.csv")[0:100]

b=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/DQL_5phut.csv")[0:100]
c1=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/n_quality_tasks_mab.csv")[0:100]
d1=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/n_quality_tasks_fuzzy.csv")[0:100]

a1=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/n_quality_tasks_fdqo.csv")[0:100]

b1=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/n_quality_tasks_dql.csv")[0:100]
xxx.write("max,"+str(np.max(a["mean_reward"]))+","+str(np.max(b["mean_reward"]))+","+str(np.max(c["mean_reward"]))+","+str(np.max(d["mean_reward"]))+"\n")
xxx.write("min,"+str(np.min(a["mean_reward"]))+","+str(np.min(b["mean_reward"]))+","+str(np.min(c["mean_reward"]))+","+str(np.min(d["mean_reward"]))+"\n")
xxx.write("average,"+str(np.average(a["mean_reward"]))+","+str(np.average(b["mean_reward"]))+","+str(np.average(c["mean_reward"]))+","+str(np.average(d["mean_reward"]))+"\n")
xxx.write("good,"+str(np.sum(a1["good"])/x)+","+str(np.sum(b1["good"])/x)+","+str(np.sum(c1["good"])/x)+","+str(np.sum(d1["good"])/x)+"\n")
xxx.write("bad,"+str(np.sum(a1["bad"])/x)+","+str(np.sum(b1["bad"])/x)+","+str(np.sum(c1["bad"])/x)+","+str(np.sum(d1["bad"])/x)+"\n")
xxx.close()