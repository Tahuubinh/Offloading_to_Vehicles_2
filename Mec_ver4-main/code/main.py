from tensorflow.keras.optimizers import Adam
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
#from rl.policy import EpsGreedyQPolicy
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.backend import cast
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Dropout,
                                     Flatten, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from keras.models import load_model

import sys

from fuzzy_controller import *
from enviroment import *
from model import *
from policy import *
from callback import *
from fuzzy_controller import *
import os
from config import *
from MyGlobal import MyGlobals

from dqnMEC import DQNAgent, BDQNAgent
from SarsaMEC import SARSAAgent

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
def Run_Random():
    files=open("random2.csv","w")
    files.write("kq\n")
    for i in range(15):
        tong=0
        h=0
        a=[]
        c=False
        while c==False:
            a,b,c,d=env.step(np.random.choice([0,0,0,0,0,1,2,3]))
            tong+=b
            if c==True :
                if i!=14:
                    env.reset()
                files.write(str(tong)+"\n")
                print(tong)

def Run_Fuzzy():
    sumreward = 0
    nreward = 0
    fuzzy_logic = Fuzzy_Controller()
    files = open("Fuzzy_5phut.csv","w")
    files1 = open("testFuzzy.csv","w")

    files.write("kq,sl,mean_reward\n")
    env = BusEnv("Fuzzy")
    #env.seed(123)
    start = timeit.default_timer()
    #env.reset()
    for i in range(100):
        tong=0
        h=0
        soluong=0

        a=env.observation
        c=False
        while c==False:
            #Rmi=(10*np.log2(1+46/(np.power(a[(1-1)*3],4)*100)))/8
            #m1=a[11]/a[2+(1-1)*3]+max(a[12]/(Rmi),a[1+(1-1)*3])
            # Rmi=(10*np.log2(1+46/(np.power(a[(2-1)*3],4)*100)))/8
            # m2=a[11]/a[2+(2-1)*3]+max(a[12]/(Rmi),a[1+(2-1)*3])
            # Rmi=(10*np.log2(1+46/(np.power(a[(3-1)*3],4)*100)))/8
            # m3=a[11]/a[2+(3-1)*3]+max(a[12]/(Rmi),a[1+(3-1)*3])
            # m0=a[9]+a[11]/a[10]
            #action=np.argmin([m0,m1,m2,m3])
            
            action=np.random.choice([0,0,0,1,2,3])
            action=0
            action=fuzzy_logic.choose_action(a)
            a,b,c,d=env.step(action)
            tong+=b
            sumreward = sumreward +b
            nreward = nreward + 1
            soluong+=1
            files1.write(str(sumreward / nreward)+"\n")
            if c==True :
                if i!=99:
                    env.reset()
                files.write(str(tong)+","+str(soluong)+","+str(tong/soluong)+"\n")
                print(tong)
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    files.close()
    
def Run_RGreedy(i, file):
    sumreward = 0
    nreward = 0
    files = open("RGreedy_5phut.csv","w")
    files1 = open("testRGreedy.csv","w")

    files.write("kq,sl,mean_reward\n")
    env = BusEnv("RGreedy")
    env.modifyEnv(i, file)
    for i in range(201):
        tong=0
        h=0
        soluong=0

        a=env.observation
        c=False
        while c==False:
            #Rmi=(10*np.log2(1+46/(np.power(a[(1-1)*3],4)*100)))/8
            #m1=a[11]/a[2+(1-1)*3]+max(a[12]/(Rmi),a[1+(1-1)*3])
            # Rmi=(10*np.log2(1+46/(np.power(a[(2-1)*3],4)*100)))/8
            # m2=a[11]/a[2+(2-1)*3]+max(a[12]/(Rmi),a[1+(2-1)*3])
            # Rmi=(10*np.log2(1+46/(np.power(a[(3-1)*3],4)*100)))/8
            # m3=a[11]/a[2+(3-1)*3]+max(a[12]/(Rmi),a[1+(3-1)*3])
            # m0=a[9]+a[11]/a[10]
            #action=np.argmin([m0,m1,m2,m3])

            predict_reward = env.predict_reward()
            action = int(max(predict_reward, key=predict_reward.get))
            a,b,c,d=env.step(action)
            tong+=b
            sumreward = sumreward +b
            nreward = nreward + 1
            soluong+=1
            files1.write(str(sumreward / nreward)+"\n")
            if c==True :
                if i!=201:
                    env.reset()
                files.write(str(tong)+","+str(soluong)+","+str(tong/soluong)+"\n")
    stop = timeit.default_timer()
    #print('Time: ', stop - start)  
    files.close()

#using for DQL
def build_model(state_size, num_actions):
    input = Input(shape=(1,state_size))
    x = Flatten()(input)
    #x = Dense(16, activation='relu')(x)

    # x = Dense(32, activation='relu')(x)

    # x = Dense(32, activation='relu')(x)
  
    # x = Dense(16, activation='relu')(x)
    for i in range(Config.length_hidden_layer):
        x = Dense(Config.n_unit_in_layer[i], activation='relu')(x)

    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    return model

def get_model():
    try:
        model = load_model('my_model.h5')
    except Exception as e: 
        print(e)
    return model

def initRun(folder_name):
    MyGlobals.folder_name = folder_name + '/'
    try:
        os.makedirs(RESULT_DIR + MyGlobals.folder_name)
    except OSError as e:
        print(e)
    folder = RESULT_DIR + MyGlobals.folder_name
    #create memory
    memory = SequentialMemory(limit=25000, window_length=1)
    #open files
    # files = open("testFDQO.csv","w")
    # files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger(folder +"/cb_5phut.csv")
    callback2 = ModelIntervalCheckpoint(folder +"/weight_cb.h5f",interval=50000)
    # callback3 = TestLogger11(files)
    return folder, memory, callbacks, callback2

def Run_DQL(folder_name):
    model=build_model(14,4)
    folder, memory, callbacks, callback2 = initRun(folder_name)
    policy = EpsGreedyQPolicy(0.1)
    env = BusEnv("DQL")
    env.seed(123)
    memory = SequentialMemory(limit=25000, window_length=1)
    try:
        dqn = DQNAgent(model=model, nb_actions=NUM_ACTIONS, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy, gamma=0.9, memory_interval=1, 
              file = folder)
    except Exception as e:
        print(e)
    # files = open("testDQL.csv","w")
    # files.write("kq\n")
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= 200000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
    except Exception as e:
        print(e)
    
def Run_BDQL(folder_name):
    #model=build_model(14,4)
    model = get_model()
    folder, memory, callbacks, callback2 = initRun(folder_name)
    # using static by setting policy2
    # for dynamic, epsilon = min(epsilon, epsilon - k(average_reward - baseline))
    # epsilon = max(epsilon, 0.01)
    baseline = 0.55   
    k = 0.6
    epsilon = 0.12
    policy = EpsGreedyQPolicy(epsilon)
    policy2 = None #EpsGreedyQPolicy(0.05)      # None if not used, mean: using dynamic insted
    reward_capacity = 10000      # Queue that save the last "reward_capacity" rewards
    env = BusEnv("BDQL")
    env.seed(123)
    memory = SequentialMemory(limit=25000, window_length=1)
    
    dqn = BDQNAgent(model=model, nb_actions=NUM_ACTIONS, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy, policy2 = policy2, gamma=0.9,
              memory_interval=1, file = folder, reward_capacity = reward_capacity,
              k = k, epsilon = epsilon)
        
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps= 200000, visualize=False, verbose=2,callbacks=[callbacks,callback2],
            baseline = baseline)
    
def Run_Static_BDQL(folder_name):
    model=build_model(14,4)
    folder, memory, callbacks, callback2 = initRun(folder_name)
    # using static by setting policy2
    # for dynamic, epsilon = min(epsilon, epsilon - k(average_reward - baseline))
    # epsilon = max(epsilon, 0.01)
    baseline = 0.6   
    k = 0.5
    epsilon = 0.1
    policy = EpsGreedyQPolicy(epsilon)
    policy2 = EpsGreedyQPolicy(0.05)
    reward_capacity = 10000      # Queue that save the last "reward_capacity" rewards
    env = BusEnv("BDQL")
    env.seed(123)
    memory = SequentialMemory(limit=25000, window_length=1)
    
    dqn = BDQNAgent(model=model, nb_actions=NUM_ACTIONS, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy, policy2 = policy2, gamma=0.9,
              memory_interval=1, file = folder, reward_capacity = reward_capacity,
              k = k, epsilon = epsilon)
        
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps= 200000, visualize=False, verbose=2,callbacks=[callbacks,callback2],
            baseline = baseline)
    
def Run_DDQL(folder_name):
    model=build_model(14,4)
    folder, memory, callbacks, callback2 = initRun(folder_name)
    policy = EpsGreedyQPolicy(0.1)
    env = BusEnv("DDQL")
    env.seed(123)
    memory = SequentialMemory(limit=25000, window_length=1)
    
    try:
        dqn = DQNAgent(model=model, nb_actions=NUM_ACTIONS, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy, gamma=0.9, memory_interval=1, 
              file = folder)
    except Exception as e:
        print(e)
        
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps= 200000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
    
def Run_Sarsa(i, file):
    model=build_model(14,4)
    num_actions = 4
    # policy has been changed to also return "exploit", be care
    policy = EpsGreedyQPolicy(0.1) 
    env = BusEnv("Sarsa")
    env.modifyEnv(i, file)
    env.seed(123)
    
    try:
        dqn = SARSAAgent(model=model, nb_actions=num_actions, nb_steps_warmup=10,
                     policy=policy,gamma=0.8)
    except Exception as e:
        print(e)
    files = open("testSarsa.csv","w")
    files.write("kq\n")
    #create callback
    callbacks = CustomerTrainEpisodeLogger("./"+ str(file) +"/Sarsa_5phut_"+ str(i) +".csv")
    callback2 = ModelIntervalCheckpoint("./"+ str(file) +"/weight_Sarsa_"+ str(i) +".h5f",interval=50000)
    callback3 = TestLogger11(files)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    try:
        dqn.fit(env, nb_steps= 200000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
    except Exception as e:
        print(e)

def Run_FDQO(folder_name):
    folder, memory, callbacks, callback2 = initRun(folder_name)
    FDQO_method = Model_Deep_Q_Learning(14,4)    #In model  size, action
    baseline = None  # None if using FDQO, >0 and <1 if using baseline
    threshold = 0.9     # if reward received bigger than threshold, using Fuzzy Logic
    k = 0.6     # Same formula as BDQL
    epsilon = 0.1
    model = FDQO_method.build_model(epsilon = epsilon, file = folder,
                                    k = k, threshold = threshold)
    #Create enviroment FDQO
    env = BusEnv("FDQO")
    env.seed(123)
    model.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    model.fit(env, nb_steps= 200000, visualize=False, verbose=2,callbacks=[callbacks,callback2],
              baseline = baseline, eps = 1)
    model.close_files()
    
def Run_BFDQO(folder_name):
    folder, memory, callbacks, callback2 = initRun(folder_name)
    FDQO_method = Model_Deep_Q_Learning(14,4)    #In model  size, action
    baseline = 0.45  # None if using FDQO, >0 and <1 if using baseline
    threshold = 0.85     # if reward received bigger than threshold, using Fuzzy Logic
    k = 0.6     # Same formula as BDQL
    epsilon = 0.12
    model = FDQO_method.build_model(epsilon = epsilon, file = folder,
                                    k = k, threshold = threshold)
    #Create enviroment FDQO
    env = BusEnv("FDQO")
    env.seed(123)
    model.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    model.fit(env, nb_steps= 200000, visualize=False, verbose=2,callbacks=[callbacks,callback2],
              baseline = baseline, eps = 1)
    model.close_files()

if __name__=="__main__":
    file = "csvFilesNorm200steps" # Location to save all the results
    # types = "DQL"
    # if len(sys.argv) > 1:
    #     types = sys.argv[1]
    # if types =="FDQO":
    #     Run_FDQO()
    # elif types == "Random":
    #     Run_Random()
    # elif types == "Fuzzy":
    #     Run_Fuzzy()
    # elif types == "DQL":
    #     Run_DQL()
    # elif types == "DDQL":
    #     Run_DDQL()
    #create model FDQO
    for i in range(1,2): # 6,11
        try:
            #Run_DQL("DQN/" + str(i))
            #Run_BDQL("Db-DQN_b_0.55_k_0.6_e_0.12/" + str(i))
            #Run_BDQL("Temp/" + str(i))
            #Run_Static_BDQL("Sb-DQN/" + str(i))
            #Run_DDQL("DDQN/" + str(i))
            #Run_FDQO("a")
            Run_BFDQO("b-FDQO_b_0.55_k_0.6_e_0.12/" + str(i))
            #Run_RGreedy("M900_1000_200_tslots", file)
            #Run_Sarsa("M900_1000", file)
        except:
            continue
   
    # for i in range(1,6):
    #     try:
    #         Run_DDQL("M900_1000_" + str(i), file)
    #     except:
    #         continue   
    # for i in range(1,6):
    #     try:
    #         Run_FDQO("M900_1000_0.9_baseline"+i, file)
    #     except:
    #         continue  



























