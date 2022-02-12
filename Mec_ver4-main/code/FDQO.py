import copy
import json
import timeit
import warnings
from tempfile import mkdtemp

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from rl.agents.ddpg import DDPGAgent
#from rl.agents.dqn import DQNAgent
from rl.agents.sarsa import SARSAAgent
from rl.callbacks import Callback, FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.backend import cast
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Dropout,
                                     Flatten, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from FDQO_method import DQNAgent
from policy import *
from fuzzy_controller import *

class BusEnv(gym.Env):


    def __init__(self):
        self.guess_count = 0
        self.number=1
        self.a1=[0,0,0,0]
        self.action_space = spaces.Discrete(5)
        self.observation_space=spaces.Box(0,100,[16])
        #(xe1_x,xe1_y,xe1_time,xe1_cpu,......,server_cpu,server_time,task_cpu,task_memory)
        self.i=0
        data900=pd.read_excel("/home/vutrian/Desktop/MEC_network_ver1-master/data/data9000.xlsx", index_col=0).to_numpy()
        data900=data900[:,13:15]
        data901=pd.read_excel("/home/vutrian/Desktop/MEC_network_ver1-master/data/data9001.xlsx", index_col=0).to_numpy()
        data901=data901[:,13:15]
        data902=pd.read_excel("/home/vutrian/Desktop/MEC_network_ver1-master/data/data9002.xlsx", index_col=0).to_numpy()
        data902=data902[:,13:15]
        self.data_bus={"900":data900,"901":data901,"902":data902}
        self.data=pd.read_csv("/home/vutrian/Desktop/MEC_network_ver1-master/data_task/datatask"+str(self.i)+".csv").to_numpy()
        self.file_luu=open("thongso.csv","w")
        self.file_luu.write("server,bus1,bus2,bus3\n")
        self.file_luu1=open("chatluong.csv","w")
        self.file_luu1.write("good,medium,bad\n")
        self.file_luu2=open("chiatask.csv","w")
        self.file_luu2.write("somay,distance,may0,may1,may2,may3,reward\n")
        self.data=np.sort(self.data, axis = 0)
        self.data[:,2]=self.data[:,2]/1000
        self.data[:,1]=self.data[:,1]/1024

        self.chatluong=[0,0,0]
        self.hangdoi=copy.deepcopy(self.data[self.data[:,0]==self.data[0][0]])
        self.data=self.data[self.data[:,0]!=self.data[0][0]]
        self.ketqua=[]
        self.Pr = 46
        self.Pr2=24
        self.Wm=10
        self.a=2
        self.b=0
        self.chinh=0
        self.o2=100
        self.time_last=self.data[-1][0]
        self.time=self.hangdoi[0][0]
        self.z=0
        self.fuzzy_logic=Fuzzy_Controller()
        self.observation=np.array([self.readexcel(900,self.hangdoi[0][0]),0.0,1\
            ,self.readexcel(901,self.hangdoi[0][0]),0,1.2\
            ,self.readexcel(902,self.hangdoi[0][0]),0,1,\
            0,3,\
            self.hangdoi[0][1],self.hangdoi[0][2],self.hangdoi[0][4]])
        self.seed()
    def readexcel(self,number_bus,time):
        data=self.data_bus[str(number_bus)]
        #print(data)
        m=data[data[:,1]>=time]
        m2=data[data[:,1]<=time]
        if len(m)==0:
            return 1.8
        las=m[0]
        first=m2[-1]
        if las[1]!=first[1]:
            mmm= (las[0]*(las[1]-time)+first[0]*(-first[1]+time))/(las[1]-first[1])
        else:
            mmm= las[0]
        #print(mmm)
        return mmm
    def step(self,action):
        #self.number=self.number+1
        if action>0 and action<4:
            Rmi=(10*np.log2(1+46/(np.power(self.observation[(action-1)*3],4)*100)))/8
            self.observation[1+(action-1)*3]=self.observation[11]/(self.observation[2+(action-1)*3])+max(self.observation[12]/Rmi,self.observation[1+(action-1)*3])
            #print(self.observation[1+(action-1)*3])
            di1 = self.readexcel(900+action-1,self.observation[1+(action-1)*3]+self.time)
            Rmi1 = 1*(10*np.log2(1+46/(np.power(self.observation[(action-1)*3],4)*100)))/8

            #print(str(Rmi)+","+str(Rmi1)+"\n")
            time_cal=(self.observation[1+(action-1)*3]+self.hangdoi[0][3]/(Rmi1*1000))
            self.file_luu2.write(str(action)+","+str(self.observation[(action-1)*3])+","+str(self.observation[9])+","+str(self.observation[1])+","+str(self.observation[4])+","+str(self.observation[7]))
        if action==0:
            self.observation[9]+=self.observation[11]/(self.observation[10])
            time_cal=self.observation[9]
            self.file_luu2.write(str(action)+","+str(0)+","+str(self.observation[9])+","+str(self.observation[1])+","+str(self.observation[4])+","+str(self.observation[7]))
        self.a1[action]=self.a1[action]+1

        #print(self.observation)
        reward=max(0,min((2*self.observation[13]-time_cal)/self.observation[13],1))
        self.file_luu2.write(","+str(reward)+"\n")
        if reward==1:
            self.chatluong[0]+=1
        elif reward==0:
            self.chatluong[2]+=1
        else:
            self.chatluong[1]+=1
        if len(self.hangdoi)!=0:
            self.hangdoi=np.delete(self.hangdoi,(0),axis=0)
        if len(self.hangdoi)==0 and len(self.data)!=0:
            self.hangdoi=copy.deepcopy(self.data[self.data[:,0]==self.data[0][0]])
            
            for a in range(3):
                self.observation[0+a*3]=self.readexcel(900+a,self.data[0][0])
            time=self.data[0][0]-self.time
            self.observation[1]=max(0,self.observation[1]-time)
            self.observation[4]=max(0,self.observation[4]-time)
            self.observation[7]=max(0,self.observation[7]-time)
            self.observation[9]=max(0,self.observation[9]-time)
            self.time=self.data[0][0]
            self.data=self.data[self.data[:,0]!=self.data[0,0]]
        if len(self.hangdoi)!=0:
            self.observation[11]=self.hangdoi[0][1]
            self.observation[12]=self.hangdoi[0][2]
            self.observation[13]=self.hangdoi[0][4]

        self.z+=1
        done= len(self.hangdoi)==0 and len(self.data)==0
        if done:
            print(self.a1)
            self.file_luu.write(str(self.a1[0])+","+str(self.a1[1])+","+str(self.a1[2])+","+str(self.a1[3])+","+"\n")
            self.file_luu1.write(str(self.chatluong[0])+","+str(self.chatluong[1])+","+str(self.chatluong[2])+"\n")

            if self.i==100:
                self.file_luu1.close()
                self.file_luu.close()
                self.file_luu2.close()
        return self.observation, reward, done,{"number": self.number, "guesses": self.guess_count}
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        self.ketqua=[]
        self.number=0
        self.z=0
        self.guess_count = 0
        #self.i+=1
        #self.b=0
        self.chinh=0
        self.chatluong=[0,0,0]
        self.a1=[0,0,0,0]
        self.i=self.i+1
        #self.i=np.random.randint(0,120)
        self.data=pd.read_csv("/home/vutrian/Desktop/MEC_network_ver1-master/data_task/datatask"+str(self.i)+".csv").to_numpy()
        self.data=np.sort(self.data, axis = 0)
        self.data[:,2]=self.data[:,2]/1000
        self.data[:,1]=self.data[:,1]/1024

        self.hangdoi=copy.deepcopy(self.data[self.data[:,0]==self.data[0][0]])
        self.data=self.data[self.data[:,0]!=self.data[0][0]]
        
        self.hangdoi=self.hangdoi
        self.time=self.hangdoi[0][0]
        
        self.observation=np.array([self.readexcel(900,self.hangdoi[0][0]),max(0,self.observation[1]-(self.time-self.time_last)),1\
            ,self.readexcel(901,self.hangdoi[0][0]),max(0,self.observation[4]-(self.time-self.time_last)),1.2\
            ,self.readexcel(902,self.hangdoi[0][0]),max(0,self.observation[7]-(self.time-self.time_last)),1,\
            max(0,self.observation[9]-(self.time-self.time_last)),3,\
            self.hangdoi[0][1],self.hangdoi[0][2],self.hangdoi[0][4]])
        #print(self.observation[4])
        self.time_last=self.data[-1][0]
        #print(str(self.time)+","+str(self.time_last))
        return self.observation
    def render(self,mode='human'):
        pass
env=BusEnv()
np.random.seed(123)
env.seed(123)
def build_model(state_size, num_actions):
    input = Input(shape=(1,state_size))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)

    x = Dense(32, activation='relu')(x)

    x = Dense(32, activation='relu')(x)
  
    x = Dense(8, activation='relu')(x)

    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    return model
class CustomerTrainEpisodeLogger(Callback):
    def __init__(self,filename):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0
        self.files=open(filename,"w")
        self.files.write("total_reward,mean_reward\n")

    def on_train_begin(self, logs):
        """ Print training values at beginning of training """
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_train_end(self, logs):
        """ Print training time at end of training """
        duration = timeit.default_timer() - self.train_start
        self.files.close()

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        """ Compute and print training statistics of the episode when done """
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = np.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]
        metrics_text = metrics_template.format(*metrics_variables)

        nb_step_digits = str(
            int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        template = '{step: ' + nb_step_digits + \
            'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}, mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}], mean observation: {obs_mean:.3f} [{obs_min:.3f}, {obs_max:.3f}], {metrics}'
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'obs_mean': np.mean(self.observations[episode]),
            'obs_min': np.min(self.observations[episode]),
            'obs_max': np.max(self.observations[episode]),
            'metrics': metrics_text,
        }

        #print(template.format(**variables))
        self.files.write(str(variables["episode_reward"])+","+str(variables["reward_mean"])+"\n")
        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1
class TestLogger11(Callback):
    """ Logger Class for Test """
    def __init__(self,path):
        self.files=path
    def on_train_begin(self, logs):
        """ Print logs at beginning of training"""

        #print('Testing for {} episodes ...'.format(self.params['nb_episodes']))

    def on_episode_end(self, episode, logs):
        
        """ Print logs at end of each episode """
        template = 'Episode {0}: reward: {1:.3f}, steps: {2}'
        variables = [
            episode + 1,
            logs['episode_reward'],
            logs['nb_steps'],
        ]
        self.files.write(str(variables[1])+"\n")

callbacks=CustomerTrainEpisodeLogger("ketqua_oneday111.csv")
callback2=ModelIntervalCheckpoint("weight_res111.h5f",interval=50000)

files=open("test_res_tang_test1s11.csv","w")
files.write("kq\n")
callback3=TestLogger11(files)
num_actions=4
model=build_model(14,4)
memory = SequentialMemory(limit=5000, window_length=1)

policy =EpsGreedyQPolicy(0.0)
#policy=ProposePolicy()
dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy,gamma=0.9,memory_interval=2)
#dqn=SARSAAgent(model=model,nb_actions=5,policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=124924, visualize=False, verbose=2,callbacks=[callbacks,callback2])#
#dqn.load_weights('C:/Users/vutri/OneDrive/Desktop/15092020/code/result/deep_q_learning/weight_res1s.h5f')
#dqn.test(env, nb_episodes=1000, visualize=False,callbacks=[callback3])
files.close()
"""
###code greedy

fuzzy_logic=fuzzy_logic()
files=open("fuzzy_150.csv","w")
files.write("kq,sl\n")
start = timeit.default_timer()
env.reset()
for i in range(129):
  tong=0
  h=0
  soluong=0

  a=env.observation
  c=False
  while c==False:
    Rmi=(10*np.log2(1+46/(np.power(a[(1-1)*3],4)*100)))/8
    m1=a[11]/a[2+(1-1)*3]+max(a[12]/(Rmi),a[1+(1-1)*3])
    Rmi=(10*np.log2(1+46/(np.power(a[(2-1)*3],4)*100)))/8
    m2=a[11]/a[2+(2-1)*3]+max(a[12]/(Rmi),a[1+(2-1)*3])
    Rmi=(10*np.log2(1+46/(np.power(a[(3-1)*3],4)*100)))/8
    m3=a[11]/a[2+(3-1)*3]+max(a[12]/(Rmi),a[1+(3-1)*3])
    m0=a[9]+a[11]/a[10]
    #action=np.argmin([m0,m1,m2,m3])
    
    action=np.random.choice([0,0,0,1,2,3])
    #action=0
    #action=fuzzy_logic.fuzzy_choose_action(a)
    a,b,c,d=env.step(action)
    tong+=b
    soluong+=1
    if c==True :
        if i!=128:
            env.reset()
        files.write(str(tong)+","+str(soluong)+"\n")
        print(tong)
stop = timeit.default_timer()
print('Time: ', stop - start)  
files.close()

###only server


files=open("onlyserver_950.csv","w")
files.write("kq\n")
for i in range(15):
  tong=0
  h=0
  a=[]
  c=False
  while c==False:
    if h==0:
        action=0
    else:
        action=np.argmin(np.array([a[9],a[1],a[4],a[7]]))
    h+=1
    a,b,c,d=env.step(0)
    tong+=b
    if c==True:
    
      if (i!=14):
        env.reset()
      files.write(str(tong)+"\n")
      print(tong)

#random
"""