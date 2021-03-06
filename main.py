from __future__ import division
import gym
import argparse
from ddpg import DDPG
from utils import *
from shared_adam import SharedAdam
import numpy as np
from normalize_env import NormalizeAction
import torch.multiprocessing as mp
from pdb import set_trace as bp
import datetime
import time
import pickle

# Parameters
parser = argparse.ArgumentParser(description='async_ddpg')

#parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--n_workers', type=int, default=2, help='how many training processes to use (default: 4)')
parser.add_argument('--rmsize', default=60000, type=int, help='memory size')
parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
parser.add_argument('--gamma', default=0.99, type=float, help='')
parser.add_argument('--env', default='Pendulum-v0', type=str, help='Environment to use')
parser.add_argument('--max_steps', default=500, type=int, help='Maximum steps per episode')
parser.add_argument('--n_eps', default=2000, type=int, help='Maximum number of episodes')
parser.add_argument('--debug', default=True, type=bool, help='Print debug statements')
parser.add_argument('--warmup', default=10000, type=int, help='time without training but only filling the replay memory')
#parser.add_argument('--num_states', default=4, type=int)
parser.add_argument('--multithread', default=0, type=int, help='To activate multithread')
parser.add_argument('--logfile', default='train_logs', type=str, help='File name for the train log data')
parser.add_argument('--n_steps', default=5, type=int, help='number of steps to rollout')

args = parser.parse_args()

env = NormalizeAction(gym.make(args.env))
discrete = isinstance(env.action_space, gym.spaces.Discrete)

# Get observation and action space dimensions
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n if discrete else env.action_space.shape[0]

def configure_env_params():
    if args.env == 'Pendulum-v0':
        args.v_min = -1000.0
        args.v_max = 100
    elif args.env == 'InvertedPendulum-v2':
        args.v_min = -100
        args.v_max = 500
    elif args.env == 'HalfCheetah-v1':
        args.v_min = -1000
        args.v_max = 1000
    else:
        print("Undefined environment. Configure v_max and v_min for environment")


class Worker(object):
    def __init__(self, name, optimizer_global_actor, optimizer_global_critic):
        self.env = NormalizeAction(gym.make(args.env).env)
        self.env._max_episode_steps = args.max_steps
        self.name = name

        self.ddpg = DDPG(obs_dim=obs_dim, act_dim=act_dim, env=self.env, memory_size=args.rmsize,\
                          batch_size=args.bsize, tau=args.tau, gamma = args.gamma, n_steps = args.n_steps)
        self.ddpg.assign_global_optimizer(optimizer_global_actor, optimizer_global_critic)
        print('Intialized worker :',self.name)

    def warmup(self):
        n_steps = 0
        self.ddpg.actor.eval()
        # for i in range(args.n_eps):
        #     state = self.env.reset()
        #     for j in range(args.max_steps):
        #
        state = self.env.reset()
        for n_steps in range(args.warmup):
            action = np.random.uniform(-1.0, 1.0, size=act_dim)
            next_state, reward, done, _ = self.env.step(action)
            self.ddpg.replayBuffer.append(state, action, reward, done)

            if done:
                state = self.env.reset()
            else:
                state = next_state


    def work(self, global_ddpg):
        avg_reward = 0.
        n_steps = 0
        #self.warmup()

        self.ddpg.sync_local_global(global_ddpg)
        self.ddpg.hard_update()

        # Logging variables
        self.start_time = datetime.datetime.utcnow()
        self.train_logs = {}
        self.train_logs['avg_reward'] = []
        self.train_logs['total_reward'] = []
        self.train_logs['time'] = []
        self.train_logs['x_val'] = []
        self.train_logs['info_summary'] = "DDPG"
        self.train_logs['x'] = 'steps'
        step_counter = 0

        for i in range(args.n_eps):
            state = self.env.reset()
            total_reward = 0.

            episode_states = []
            episode_rewards = []
            episode_actions = []

            for j in range(args.max_steps):
                self.ddpg.actor.eval()

                state = state.reshape(1, -1)
                noise = self.ddpg.noise.sample()
                action = np.clip(to_numpy(self.ddpg.actor(to_tensor(state))).reshape(-1, ) + noise, -1.0, 1.0)
                # action = to_numpy(self.ddpg.actor(to_tensor(state))).reshape(-1, ) + noise
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                #### n-steps buffer
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)

                if j >= args.n_steps-1:
                    cum_reward = 0.
                    exp_gamma = 1
                    for k in range(-args.n_steps, 0):
                        cum_reward += exp_gamma * episode_rewards[k]
                        exp_gamma *= args.gamma
                    self.ddpg.replayBuffer.add(episode_states[-args.n_steps].reshape(-1), episode_actions[-args.n_steps], cum_reward, next_state, done)
                    # self.ddpg.replayBuffer.add_experience(state.reshape(-1), action, reward, next_state, done)
                    #self.ddpg.replayBuffer.append(state.reshape(-1), action, reward, done)

                self.ddpg.actor.train()
                self.ddpg.train(global_ddpg)
                step_counter += 1
                n_steps += 1

                if done:
                    break


                state = next_state
                # print("Episode ", i, "\t Step count: ", n_steps)

            self.ddpg.noise.reset()
            avg_reward = 0.95*avg_reward + 0.05*total_reward
            if i%1==0:
                print('Episode ',i,'\tWorker :',self.name,'\tAvg Reward :',avg_reward,'\tTotal reward :',total_reward,'\tSteps :',n_steps)
                self.train_logs['avg_reward'].append(avg_reward)
                self.train_logs['total_reward'].append(total_reward)
                self.train_logs['time'].append((datetime.datetime.utcnow()-self.start_time).total_seconds()/60)
                self.train_logs['x_val'].append(step_counter)
                with open(args.logfile, 'wb') as fHandle:
                    pickle.dump(self.train_logs, fHandle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(args.logfile_latest, 'wb') as fHandle:
                    pickle.dump(self.train_logs, fHandle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    configure_env_params()
    args.logfile_latest = args.logfile + '_' + args.env + '_latest_DDPG' + '.pkl'
    args.logfile = args.logfile + '_' + args.env + '_DDPG_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl'



    global_ddpg = DDPG(obs_dim=obs_dim, act_dim=act_dim, env=env, memory_size=args.rmsize,\
                        batch_size=args.bsize, tau=args.tau)
    optimizer_global_actor = SharedAdam(global_ddpg.actor.parameters(), lr=5e-5)
    optimizer_global_critic = SharedAdam(global_ddpg.critic.parameters(), lr=5e-5)#, weight_decay=1e-02)

    # optimizer_global_actor.share_memory()
    # optimizer_global_critic.share_memory()
    global_ddpg.share_memory()

    if not args.multithread:
        worker = Worker(str(1), optimizer_global_actor, optimizer_global_critic)
        worker.work(global_ddpg)
    else:
        processes = []
        for i in range(args.n_workers):
          worker = Worker(str(i), optimizer_global_actor, optimizer_global_critic)
          p = mp.Process(target=worker.work, args=[global_ddpg])
          p.start()
          processes.append(p)

        for p in processes:
            p.join()