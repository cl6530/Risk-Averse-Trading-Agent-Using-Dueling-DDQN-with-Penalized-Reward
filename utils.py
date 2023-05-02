import random
import numpy as np
import torch
import torch.nn as nn
import glob
import pandas as pd
from collections import namedtuple, deque


def select_action(state,epsilon,action_space,policy_net,device,dueling = False):
    '''
    Choose action based on epsilon-greedy policy
    Params
    --------------
    1.state: current state
    2.epsilon: epsilon at t
    3.action_space: space to sample from when explore
    4.policy_net: policy network to use when exploit
    5.device: cpu or gpu
    6.dueling: whether or not the network uses a dueling structure, default FALSE

    Returns
    --------------
    an action
    '''

    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():

            pred = policy_net(torch.tensor(state.astype(np.float32),device=device))
            return pred.max(0)[1]
    else:
        return torch.tensor([[action_space.sample()]], device=device, dtype=torch.long)
    

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    '''
    Replay buffer for DDQN
    '''
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def optimize_model(memory,optimizer,policy_net,target_net,device,BATCH_SIZE = 1024, GAMMA = 0.9, dueling = False):
    '''
    Optimize the model for one iteration

    Params
    --------------
    1.memory: ReplayMemory Object
    2.optimizer: optimizer used for models
    3.policy_net
    4.target_net
    5.device: cpu or gpu
    6.BATCH_SIZE: pre-defined batch size, default 1024
    7.GAMMA: pre-defiend discount factor, default 0.9
    8.dueling: whether or not the network uses a dueling structure, default FALSE
    '''

    if len(memory) > BATCH_SIZE:
      transitions = memory.sample(BATCH_SIZE)
    else:
      transitions = memory.sample(len(memory))
    
    batch = Transition(*zip(*transitions))


    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)
    next_state_batch = torch.stack(batch.next_state)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(len(next_state_batch), device=device)
    with torch.no_grad():
        if dueling:
            raise NotImplementedError()
            # FIXME: When using dueling structure, this need to be changed
        else:
            next_state_values = target_net(next_state_batch).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_data(drive_path = False):
    '''
    get all historical returns data

    if drive_path is Fasle, read data from relative path
    else read from specific google drive location
    '''
    if drive_path == False:
        path = r'dataset'
    else:
        path = drive_path
    all_files = glob.glob(path + "/*.csv")
    print(all_files)
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = df['Close'].str.replace(',','').astype(float)
    prices = df.sort_values(by = 'Date')
    returns = prices['Close'].pct_change()[1:]
    return returns


