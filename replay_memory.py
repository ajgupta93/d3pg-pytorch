import random
import numpy as np

class Replay(object):
    def __init__(self, max_size):
        self.buffer = []
        self.capacity = max_size
        self.position = 0
        self.initialize(init_length=1000)
        
    def add_experience(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (np.asarray(state), action, reward,\
                                      np.asarray(next_state), done)
        self.position = (self.position+1)%self.capacity
    
    def initialize(self, init_length, env=env):
        state = env.reset()
        while True:
            action = np.random.uniform(-1.0, 1.0, size=env.action_space.shape)
            next_state, reward, done, _ = env.step(action)
            self.add_experience(state, action, reward, next_state, done)
            if done:
                state = env.reset()
                if len(self.buffer)>=init_length:
                    break
            else:
                state = next_state
    
    def sample(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminates = []
        samples = random.sample(self.buffer, batch_size)
        for state, action, reward, next_state, done in samples:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminates.append(done)
        
        states = np.array(states, dtype=np.float).reshape(batch_size,-1)
        actions = np.array(actions, dtype=np.float).reshape(batch_size,-1)
        rewards = np.array(rewards, dtype=np.float).reshape(batch_size,-1)
        next_states = np.array(next_states, dtype=np.float).reshape(batch_size,-1)
        terminates = np.array(terminates, dtype=np.float).reshape(batch_size,-1)
        return states, actions, rewards, next_states, terminates
