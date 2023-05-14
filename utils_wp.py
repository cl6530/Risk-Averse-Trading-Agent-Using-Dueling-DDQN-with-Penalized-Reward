import random
import numpy as np
import torch
import torch.nn as nn
import glob
import pandas as pd
from collections import namedtuple, deque

# Define a function for choosing an action
def select_action(state,epsilon,action_space,policy_net,device):
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

    sample = random.random()# Generate a random number
    if sample > epsilon: # If the random number is greater than epsilon 
        with torch.no_grad():# Disables gradient calculation
            pred = policy_net(torch.tensor(state.astype(np.float32),device=device))# Make a prediction using the policy network
            return pred.max(0)[1]# Return the action that has the maximum value
    else: # If the random number is less than or equal to epsilon
        return torch.tensor([[action_space.sample()]], device=device, dtype=torch.long)# Sample a random action from the action space and return it
    
# Define a named tuple called 'Transition'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Define a class for the replay memory
class ReplayMemory(object):
    '''
    Replay buffer for DDQN
    '''
    def __init__(self, capacity):# Initialize the replay memory with a certain capacity
        self.memory = deque([], maxlen=capacity)# Create a deque with a maximum length equal to the given capacity

    def push(self, *args):# Define a method for adding a transition to the memory
        """Save a transition"""
        self.memory.append(Transition(*args)) # Append the given transition to the memory

    def sample(self, batch_size): # Define a method for sampling a batch of transitions from the memory
        return random.sample(self.memory, batch_size) # Return a random sample of transitions

    def __len__(self):# Define a method for getting the length of the memory
        return len(self.memory)

def optimize_model(memory,optimizer,policy_net,target_net,device,criterion,rewardthresh, penfactor, BATCH_SIZE = 1024, GAMMA = 0.9):
    '''
    Optimize the model for one iteration

    Params
    --------------
    1.memory: ReplayMemory Object
    2.optimizer: optimizer used for models
    3.policy_net
    4.target_net
    5.device: cpu or gpu
    6.criterion: loss function to use
    7.BATCH_SIZE: pre-defined batch size, default 1024
    8.GAMMA: pre-defiend discount factor, default 0.9
    9.dueling: whether or not the network uses a dueling structure, default FALSE
    '''

    if len(memory) > BATCH_SIZE:# If the size of the memory is larger than the batch size
      transitions = memory.sample(BATCH_SIZE) # Sample a batch of transitions from the memory
    else:
      transitions = memory.sample(len(memory))# Sample all transitions from the memory
    
    batch = Transition(*zip(*transitions)) # Unpack the transitions and store them in a 'Transition' named tuple

    # Prepare the batches of states, actions, rewards, and next states
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)
    next_state_batch = torch.stack(batch.next_state)
    state_action_values = policy_net(state_batch).gather(1, action_batch)# Calculate the state-action values

    next_state_values = torch.zeros(len(next_state_batch), device=device)
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]# Calculate the maximum Q-value for the next states using the target network
     
     # Penalize negative rewards
    reward_batch[reward_batch < rewardthresh] *= penfactor
    # Compute the expected Q-values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute the loss
    loss = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

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

    df = pd.concat(li, axis=0, ignore_index=True)# Concatenate all DataFrames in the list
    df['Date'] = pd.to_datetime(df['Date'])# Convert the 'Date' column to datetime
    df['Close'] = df['Close'].str.replace(',','').astype(float) # Clean the 'Close' column and convert it to float
    df.set_index('Date',inplace=True)
    prices = df.sort_values(by = 'Date')# Sort the DataFrame by date
    returns = prices['Close'].pct_change()[1:]# Calculate the percentage change in the 'Close' column
    return returns