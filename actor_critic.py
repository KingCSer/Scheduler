import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import torch.optim as optim
from torch.distributions import Categorical
import math
from collections import deque
import random

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward):
        experience = (state, action, reward)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self,state_dim,action_dim):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_dim, 256)
        # actor's layer
        self.action_head1 = nn.Linear(256,196)
        self.action_head = nn.Linear(196,196)
        self.action_medium = nn.Linear(196,128)
        self.action_tail = nn.Linear(128,action_dim)

        # critic's layer
        self.value_head1 = nn.Linear(256,196)
        self.value_head = nn.Linear(196, 196)
        self.value_medium = nn.Linear(196, 128)
        self.value_tail = nn.Linear(128, 1)

        # replay buffer
        self.ReplayBuffer_size = 1000000 # buffer size
        self.batch_size = 256 # batch_size
        self.replay_buffer = ReplayBuffer(self.ReplayBuffer_size)
        self.saved_actions = []
        self.rewards = []
        self.log_prob_actions = []
        self.values = []
        self.states = []
        self.optimizer = optim.Adam(self.parameters(),lr=0.0001)


    def forward(self,x):
        # print("affinel mean: ",self.affine1(x).mean())
        x0 = F.relu(self.affine1(x))
        # actor: choses action to take from state s_t
        # by returning probability of each action
        out1 = F.relu(self.action_head1(x0))
        out2 = F.relu(self.action_head(out1))
        # print("out1 mean: ",out1.mean())
        out3 = F.relu(self.action_medium(out2))
        # out3 += 1e12 #继续训练，防止合法action_prob为0
        # print("out2 mean: ",out2.mean())
        action_prob = F.softmax(self.action_tail(out3),dim=-1)

        # action_prob = F.softmax(self.action_head(x), dim=-1)
        # critic: evaluates being in the state s_t
        result1 = F.relu(self.value_head1(x0))
        result2 = F.relu(self.value_head(result1))
        # print("result1 mean: ",result1.mean())
        result3 = F.relu(self.value_medium(result2))
        # print("result2 mean: ",result2.mean())
        state_values = self.value_tail(result3)
        # state_values = self.value_head(x)
        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


def select_action(model,state,train_flag,free_GPU):
    # state = torch.FloatTensor(state)
    # state = torch.from_numpy(state),float()
    probs, state_value = model(state)

    m = Categorical(probs)
    # and sample an action using the distribution
    action = m.sample()  # 0 or 1
    # print("probs: ",probs)
    # print("choose first...")
    if action not in free_GPU:
        # print("choose again...")
        prob1 = probs.clone().detach()
        gpu_list = list(range(16))
        for gpu_id in gpu_list:
            if gpu_id not in free_GPU:
                prob1[gpu_id] = 0
        if sum(prob1) == 0:
            action = random.sample(free_GPU, 1)  # if valid action's prob==0, select randomly from free GPU
        # prob_sum = sum(prob1)
        # for i in range(len(prob1)):
        #     prob_temp = prob1[i]
        #     prob1[i] = prob_temp / prob_sum
        # print("prob1: ",prob1)
        else:
            n = Categorical(prob1)
            action = n.sample()
    log_prob_action = m.log_prob(action)

    if train_flag:
        model.values.append(state_value)
    free_GPU.remove(action)
    return action.item(), log_prob_action, probs, free_GPU


def choose_action_again(invalid_action,prob_weights,free_gpu_list):
    prob_weights[invalid_action] = 0
    gpu_list = list(range(2))
    for gpu_id in gpu_list:
        if gpu_id not in free_gpu_list:
            prob_weights[gpu_id] = 0
    prob_sum = sum(prob_weights)
    for i in range(len(prob_weights)):
        prob_temp = prob_weights[i]
        prob_weights[i] = prob_temp/prob_sum
    print("prob_weights: ",prob_weights)


    m = Categorical(prob_weights)
    action = m.sample() #sample in 0~50 based prob_weights
    log_prob_action = m.log_prob(action)
    print("log_prob_action(again): ",log_prob_action)
    return action.item(),log_prob_action,prob_weights


def Train(model):
    R = 0
    policy_losses = []
    value_losses = []
    returns = []
    a_batch = []
    r_batch = []
    s_batch = []
    v_batch = []
    R_batch = np.zeros(len(model.rewards))
    td_batch = []
    log_prob_action_batch = []
    gamma = 0.9
    eps = np.finfo(np.float32).eps.item()
    print("actor.rewards.length: ", len(model.rewards))

    minibatch = model.replay_buffer.get_batch(len(model.rewards))
    s_batch = [data[0] for data in minibatch]
    a_batch = torch.tensor([data[1] for data in minibatch])
    r_batch = [data[2] for data in minibatch]
    for i in range(len(model.rewards)):
        action_prob,value = model(s_batch[i])
        m = Categorical(action_prob)
        log_prob_action = m.log_prob((a_batch[i]))
        log_prob_action_batch.append(log_prob_action)
        v_batch.append(value)
    R_batch[-1] = v_batch[-1]

    # print("values: ",actor.values)
    # for r in r_batch[::-1]:
    #     # calculate the discount value
    #     R = r + gamma * R
    #     returns.insert(0, R)
    for t in reversed(range(len(model.rewards) - 1)):
        R_batch[t] = r_batch[t] + gamma * R_batch[t+1]

    R_batch = torch.tensor(R_batch, dtype=torch.float32)
    R_batch = (R_batch - R_batch.mean()) / (R_batch.std() + eps)
    print("R_batch: ", R_batch)
    for i in range(len(R_batch)):
        TD_error = R_batch[i] - v_batch[i]
        policy_losses.append(-log_prob_action_batch[i] * TD_error)
        value_losses.append(F.smooth_l1_loss(v_batch[i], torch.tensor([R_batch[i]])))

    model.optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    print("loss: ", loss)
    for name,param in model.named_parameters():
        if param.grad is None:
            print(name)
    # perform backprop
    # with torch.autograd.set_detect_anomaly(True):
    #     loss.backward()
    # torch.autograd.set_detect_anomaly(True)
    loss.backward()
    model.optimizer.step()

    #reset rewards , action and log_prob_actions buffer
    del model.rewards[:]
    del model.saved_actions[:]
    del model.log_prob_actions[:]
    del model.values[:]
    del model.states[:]
