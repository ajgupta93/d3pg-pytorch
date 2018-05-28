import gym
import argparse
from ddpg import DDPG
from utils import *
from shared_adam import SharedAdam
import numpy as np
from normalize_env import NormalizeAction
import torch.multiprocessing as mp
import pdb

# Parameters
parser = argparse.ArgumentParser(description='async_ddpg')

#parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--n_workers', type=int, default=1, help='how many training processes to use (default: 4)')
parser.add_argument('--rmsize', default=50000, type=int, help='memory size')
#parser.add_argument('--init_w', default=0.003, type=float, help='')
#parser.add_argument('--window_length', default=1, type=int, help='')
parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
parser.add_argument('--discount', default=0.9, type=float, help='')
parser.add_argument('--env', default='Pendulum-v0', type=str, help='Environment to use')
parser.add_argument('--max_steps', default=500, type=int, help='Maximum steps per episode')
parser.add_argument('--n_eps', default=2000, type=int, help='Maximum number of episodes')
parser.add_argument('--debug', default=True, type=bool, help='Print debug statements')
#parser.add_argument('--epsilon', default=10000, type=int, help='linear decay of exploration policy')
parser.add_argument('--warmup', default=10000, type=int, help='time without training but only filling the replay memory')
#parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
#parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
#parser.add_argument('--load_weights', dest="load_weights", action='store_true', help='load weights for actor and critic')
#parser.add_argument('--shared', dest="shared", action='store_true')
#parser.add_argument('--use_more_states', dest="use_more_states", action='store_true')
#parser.add_argument('--num_states', default=4, type=int)


args = parser.parse_args()

env = NormalizeAction(gym.make(args.env))
discrete = isinstance(env.action_space, gym.spaces.Discrete)

# Get observation and action space dimensions
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n if discrete else env.action_space.shape[0]

class Worker(object):
    def __init__(self, name, optimizer_global_actor, optimizer_global_critic):
        self.env = NormalizeAction(gym.make(args.env).env)
        self.env._max_episode_steps = args.max_steps
        self.name = name

        self.model = DDPG(obs_dim=obs_dim, act_dim=act_dim, memory_size=args.rmsize,\
                          batch_size=args.bsize, tau=args.tau)
        self.model.assign_global_optimizer(optimizer_global_actor, optimizer_global_critic)
        print('Intialized worker :',self.name)

    def warmup(self):
        n_steps = 0
        self.model.actor.eval()
        # for i in range(args.n_eps):
        #     state = self.env.reset()
        #     for j in range(args.max_steps):
        #
        state = self.env.reset()
        for n_steps in range(args.warmup):
            state = state.reshape(1,-1)
            action = np.random.uniform(-1.0, 1.0, size=act_dim)
            next_state, reward, done, _ = self.env.step(action)
            self.model.replayBuffer.append(state, action, reward, done)

            if done:
                state = self.env.reset()
            else:
                state = next_state


    def work(self, global_model):
        avg_reward = 0.
        n_steps = 0

        self.warmup()

        for i in range(args.n_eps):
            #pdb.set_trace()
#            self.model.sync_local_global(global_model)
            state = self.env.reset()
            total_reward = 0.
            for j in range(args.max_steps):
                # pdb.set_trace()
                # self.model.sync_local_global(global_model)

                self.model.actor.eval()

                state = state.reshape(1,-1)
                noise = self.model.noise.sample()
                action = to_numpy(self.model.actor(to_tensor(state))).reshape(-1,) + noise
                #print('Action :', action)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                self.model.replayBuffer.append(state, action, reward, done)

                #pdb.set_trace()
                self.model.actor.train()
                self.model.train(global_model)

                n_steps += 1

                if done:
                    break

                state = next_state


            avg_reward = 0.95*avg_reward + 0.05*total_reward
            if i%1==0:
                print('Episode ',i,'\tWorker :',self.name,'\tAvg Reward :',avg_reward,'\tTotal reward :',total_reward,'\tSteps :',n_steps)

if __name__ == '__main__':
    global_model = DDPG(obs_dim=obs_dim, act_dim=act_dim, memory_size=args.rmsize,\
                        batch_size=args.bsize, tau=args.tau)
    optimizer_global_actor = SharedAdam(global_model.actor.parameters(), lr=1e-4)
    optimizer_global_critic = SharedAdam(global_model.critic.parameters(), lr=1e-3)

    optimizer_global_actor.share_memory()
    optimizer_global_critic.share_memory()
    global_model.share_memory()

    worker = Worker('1', optimizer_global_actor, optimizer_global_critic)
    worker.work(global_model)

    #processes = []
    #for i in range(args.n_workers):
    #    worker = Worker(str(i), optimizer_global_actor, optimizer_global_critic)
    #    p = mp.Process(target=worker.work, args=[global_model])
    #    p.start()
    #    processes.append(p)

    #for p in processes:
    #    p.join()