from tensorflow.keras.backend import cast
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Dropout,
                                     Flatten, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from FDQO_method import DQNAgent
from policy import EpsGreedyQPolicy, EpsGreedyFuzzyPolicy
from config import Config
from rl.memory import SequentialMemory

class Model_Deep_Q_Learning:
    def __init__(self,state_size,num_actions):
        self.state_size=state_size
        self.num_actions=num_actions
        self.memory = SequentialMemory(limit=5000, window_length=1)

    def build_model(self, epsilon = .1, name = None, file = None, k = 0, threshold = 0.8):
        input = Input(shape=(1,self.state_size))
        x = Flatten()(input)
        for i in range(Config.length_hidden_layer):
            x = Dense(Config.n_unit_in_layer[i], activation='relu')(x)
        output = Dense(self.num_actions, activation='linear')(x)
        model = Model(inputs=input, outputs=output)
        model.summary()
        policy =EpsGreedyQPolicy(epsilon) #EpsGreedyFuzzyPolicy must have select_action
        #print(file)
        dqn = DQNAgent(model=model, nb_actions=self.num_actions, memory=self.memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy, gamma=0.9, memory_interval=2,
              i = name, file = file, k = k, epsilon = epsilon, threshold = threshold)
        return dqn