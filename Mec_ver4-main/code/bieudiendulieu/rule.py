import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def threshold_time(strs):
    labels=["combine fuzzy and deep q","deep q learning","fuzzy","random"]
    ax=plt.subplot()
    x=[0,1.2,1.4,2,2.5,3]
    x2=[0,0.5,1.2,2,2.5,5]
    x3=[0,0.6,1,2.5,4,5]
    y1=[1,1,0,0,0,0]
    y2=[0,0,1,1,0,0]
    y3=[0,0,0,0,1,1]
    ax.plot(x,y1,label="T-P",linewidth=3)
    ax.plot(x,y2,label="N-P",linewidth=3)
    ax.plot(x,y3,label="L-P",linewidth=3)
    ax.set_xlabel("Seconds",fontsize=15,)
    ax.set_ylim(0,1.1)
    plt.grid()
    ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .3),fontsize=15, ncol=3)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("threshold_time.eps")
    #plt.show()
def bus_time(strs):
    labels=["combine fuzzy and deep q","deep q learning","fuzzy","random"]
    ax=plt.subplot()
    x=[0,1.2,1.4,2,2.5,3]
    x2=[0,0.5,1.2,2,2.5,5]
    x3=[0,0.6,1,2.5,4,5]
    y1=[1,1,0,0,0,0]
    y2=[0,0,1,1,0,0]
    y3=[0,0,0,0,1,1]
    ax.plot(x2,y1,label="low",linewidth=3)    
    ax.plot(x2,y2,label="medium",linewidth=3)
    ax.plot(x2,y3,label="high",linewidth=3)
    ax.set_xlabel("Seconds",fontsize=15)
    ax.set_ylim(0,1.1)
    ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .3),fontsize=15, ncol=3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid()
    plt.savefig("bus_time.eps")
def server_time(strs):
    labels=["combine fuzzy and deep q","deep q learning","fuzzy","random"]
    ax=plt.subplot()
    x=[0,1.2,1.4,2,2.5,3]
    x2=[0,0.5,1.2,2,2.5,5]
    x3=[0,0.6,1,2.5,4,5]
    y1=[1,1,0,0,0,0]
    y2=[0,0,1,1,0,0]
    y3=[0,0,0,0,1,1]
    ax.plot(x3,y1,label="low",linewidth=3)    
    ax.plot(x3,y2,label="medium",linewidth=3)
    ax.plot(x3,y3,label="high",linewidth=3)
    ax.set_xlabel("Seconds",fontsize=15)
    ax.set_ylim(0,1.1)
    ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .3),fontsize=15, ncol=3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.savefig("server_time.eps")
    #plt.show()
threshold_time("an")
#bus_time("an")
#server_time("an")
