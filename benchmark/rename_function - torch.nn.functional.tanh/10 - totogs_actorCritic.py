
import numpy as np
import random
import torch
from torch.autograd import Function
from torch.autograd import gradcheck
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.distributions import Categorical

import copy

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class NN(nn.Module):
	def __init__(self, inSize, outSize, layers=[]):
		super(NN, self).__init__()
		self.layers = nn.ModuleList([])
		for x in layers:
			self.layers.append(nn.Linear(inSize, x))
			inSize = x
		self.layers.append(nn.Linear(inSize, outSize))

	def forward(self, x):
		x = self.layers[0](x)
		for i in range(1, len(self.layers)):
			x = torch.nn.functional.tanh(x)
			x = self.layers[i](x)
		return x


class Pi(nn.Module):
    def __init__(self, inSize, outSize):

        super(Pi, self).__init__()
        self.affine = nn.Linear(inSize, 200)
        self.action_head = nn.Linear(200, outSize)
        self.value_head = nn.Linear(200,1)

    def forward(self, x):

        x = f.relu(self.affine(x))

        action_prob = f.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)

        return action_prob, state_value

eps = np.finfo(np.float32).eps.item()

class Actor(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(Actor, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
        	self.layers.append(nn.Linear(inSize, x))
        	inSize = x
        self.layers.append(nn.Linear(inSize, outSize))


    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
        	x = torch.nn.functional.tanh(x)
        	x = self.layers[i](x)
        distribution = Categorical(f.softmax(x))
        return distribution


class Critic(nn.Module):
    def __init__(self, inSize, layers=[]):
        super(Critic, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
        	self.layers.append(nn.Linear(inSize, x))
        	inSize = x
        self.layers.append(nn.Linear(inSize, 1))

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
        	x = torch.nn.functional.tanh(x)
        	x = self.layers[i](x)
        return x


class ActorCritic():

    def __init__(self, env, state_dim, layers=[128], lrv=0.001, lrpi=0.001, gamma=0.99, verbose=False):

        self.trajectory = list()
        self.n_actions = env.action_space.n
        self.gamma = gamma

        self.Pinet = Actor(state_dim, self.n_actions, layers=layers)
        self.Vnet = Critic(state_dim, layers=layers)

        self.Pioptimizer = optim.Adam(self.Pinet.parameters(),lr=lrpi)
        self.Voptimizer = optim.Adam(self.Vnet.parameters(),lr=lrv)
        self.Vcriterion = nn.SmoothL1Loss()

        self.lobs = None
        self.lact = None
        self.ep=0




    def setState(self, observation):

        self.lobs = torch.from_numpy(observation).float()



    def act(self, obs, reward, done):

        self.lobs = torch.from_numpy(obs).float()

        distrib = self.Pinet(self.lobs.unsqueeze(0))
        value = self.Vnet(self.lobs.unsqueeze(0))

        action = distrib.sample()
        self.lact = action

        self.trajectory.append((reward, value, distrib.log_prob(action).unsqueeze(0)))

        if done:
            self.step()
            self.trajectory = list()

        return action.squeeze().numpy()


    def step(self):

        v_target = list()
        policy_losses = list()
        value_losses = list()
        v=0

        for (reward, _, _)  in reversed(self.trajectory):

            v = reward + self.gamma * v
            v_target.append(v)

        v_target.reverse()


        v_target = torch.tensor(v_target, dtype=torch.float).detach()


        values = torch.cat([val for (_,val,_) in self.trajectory])
        logprobs = torch.cat([logprob for (_,_,logprob) in self.trajectory])
        logprobs.requires_grad_()



        avantages = v_target - values


        Ploss = -(logprobs*avantages.detach()).mean()
        Vloss = avantages.pow(2).mean().requires_grad_()

        self.Voptimizer.zero_grad()
        self.Pioptimizer.zero_grad()
        Ploss.backward()
        Vloss.backward()

        self.Voptimizer.step()
        self.Pioptimizer.step()


        writer.add_scalar('Ploss_per_episode',Ploss.item(),self.ep)
        writer.add_scalar('Vloss_per_episode',Vloss.item(),self.ep)
        self.ep+=1
