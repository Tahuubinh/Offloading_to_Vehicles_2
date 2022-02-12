import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def column_chart1(strs):
    x=[i for i in range(1,251)]
    labels=["Combine fuzzy and deep q 1","deep q learning 1","fuzzy","random"]
    fig, ax = plt.subplots()
    files=pd.read_csv("../n_quality_tasks_dql_1.csv")[0:250]
    m=files["good"]+files["medium"]+files["bad"]
    print(m)
    print([np.mean(files["good"]/m)]*len(x))
    ax.plot(x,[np.mean(files["good"]/m)]*len(x), label='Mean good',color="violet", linestyle='--')
    ax.plot(x,[np.mean(files["medium"]/m)]*len(x), label='Mean medium',color="y", linestyle='--')
    ax.plot(x,[np.mean(files["bad"]/m)]*len(x), label='Mean bad',color="gray", linestyle='--')
    ax.plot(x,files["good"]/m,label="Good",color="violet",marker='*',markevery=5)
    ax.plot(x,files["medium"]/m,marker='^',label="Medium",color="y",markevery=5)
    #ax1.plot(x,files["bus2"],":",label="vehicular fog 2")
    ax.plot(x,files["bad"]/m,label="Bad",color="gray",markevery=5)
    ax.set_xlabel("Time slots",fontsize=15)
    ax.set_ylabel("Ratio",fontsize=15)
    ax.set_ylim(0,1)
    #plt.show()
    ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=6)
    #plt.show()
    plt.grid(alpha=0.5)
    plt.savefig("fdqoqualitywithtimeslots.eps")
def pie_chart1(strs):
    x=[i for i in range(1,101)]
    labels=["Combine fuzzy and deep q 1","deep q learning 1","fuzzy","random"]
    fig, ax = plt.subplots()
    files=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/result4/deep q learning 1/chatluong.csv")[0:100]
    for i in range(2,6):
        files1=pd.read_csv("D:/Binh/Projects/GitDownload/Mec_ver4-main/Mec_ver4-main/result4/deep q learning "+str(i)+"/chatluong.csv")[0:100]
        files["good"]=files1["good"]+files["good"]
        files["medium"]=files1["medium"]+files["medium"]
        files["bad"]=files1["bad"]+files["bad"]
    good=np.sum(files["good"])
    medium=np.sum(files["medium"])
    bad=np.sum(files["bad"])
    alls=[good,medium,bad]
    labels=["good","medium","bad"]
    colors=[(1,0,0),(0,1,0),(0,0,1)]
    fig=ax.pie(alls,labels=labels,autopct='%1.1f%%',wedgeprops={'linewidth':1,'edgecolor':"black"},colors=colors, textprops=dict(color="white",size=15,weight='bold'))
    #plt.legend()
    #ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=3)
    #plt.show()
    plt.savefig("DQLpiechart.pdf")
column_chart1("m")