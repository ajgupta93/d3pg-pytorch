from models import actor, critic
import torch
import torch.optim as optim
import torch.nn as nn
from random_process import OrnsteinUhlenbeckProcess
from utils import *
from replay_memory import Replay

class DDPG:
    def __init__(self, obs_dim, act_dim, critic_lr = 1e-3, actor_lr = 1e-4, gamma = 0.99, alpha_decay=0.93, batch_size = 64):
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # actor
        self.actor = actor(input_size = obs_dim, output_size = act_dim).type(FloatTensor)
        self.actor.cuda()
        self.actor_target = actor(input_size = obs_dim, output_size = act_dim).type(FloatTensor)
        self.actor_target.cuda()
        self.actor_target.load_state_dict(self.actor.state_dict())

        # critic
        self.critic = critic(state_size = obs_dim, action_size = act_dim, output_size = 1).type(FloatTensor)
        self.critic.cuda()
        self.critic_target = critic(state_size = obs_dim, action_size = act_dim, output_size = 1).type(FloatTensor)
        self.critic_target.cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr = critic_lr, weight_decay=1e-2)
        
        # learning rate scheduler
        self.scheduler_actor = optim.lr_scheduler.StepLR(self.optimizer_actor, step_size=5000, gamma=alpha_decay)
        self.scheduler_critic = optim.lr_scheduler.StepLR(self.optimizer_critic, step_size=5000, gamma=alpha_decay)
        
        
        # critic loss
        self.critic_loss = nn.MSELoss()
        
        # noise
        self.noise = OrnsteinUhlenbeckProcess(dimension = act_dim, num_steps = NUM_EPISODES)

        # replay buffer 
        self.replayBuffer = Replay(60000)
        
        
    def train(self):
     
        # sample from Replay
        states, actions, rewards, next_states, terminates = self.replayBuffer.sample(self.batch_size)

        # update critic (create target for Q function)
        target_qvalues = self.critic_target(to_tensor(next_states, volatile=True),\
                                           self.actor_target(to_tensor(next_states, volatile=True)))
        y = to_numpy(to_tensor(rewards) +\
            self.gamma*to_tensor(1-terminates)*target_qvalues)

        q_values = self.critic(to_tensor(states),
                               to_tensor(actions))
        qvalue_loss = self.critic_loss(q_values, to_tensor(y, requires_grad=False))
        
               
        # critic optimizer and backprop step (feed in target and predicted values to self.critic_loss)
        self.critic.zero_grad()
        qvalue_loss.backward()
        self.optimizer_critic.step()
        self.scheduler_critic.step()
        

        # update actor (formulate the loss wrt which actor is updated)
        policy_loss = -self.critic(to_tensor(states),\
                                 self.actor(to_tensor(states)))
        policy_loss = policy_loss.mean()
        

        # actor optimizer and backprop step (loss_actor.backward())
        self.actor.zero_grad()
        policy_loss.backward()
        self.optimizer_actor.step()
        self.scheduler_actor.step()
        
        # sychronize target network with fast moving one
        weightSync(self.critic_target, self.critic)
        weightSync(self.actor_target, self.actor)
        
