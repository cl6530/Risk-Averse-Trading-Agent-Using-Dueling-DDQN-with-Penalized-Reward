import gym
import numpy as np
import pywt

class StockTradingEnv(gym.Env):
    '''
    Trading Simulator
    
    '''
    def __init__(self, stock_returns):
        '''
        init methods, takes an array of returns as input
        '''
        super(StockTradingEnv, self).__init__()

        self.returns = stock_returns
        self.current_step = 5
        self.position = 0
        try:
          self.current_state = self._get_next_state()
        except:
          pass

        # Action space: {0: 'short', 1: 'stay', 2: 'long'}
        self.action_space = gym.spaces.Discrete(3)

        # State space: [1-day return, 5-day return]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)


    def _get_next_state(self):
      '''
      using 1-day and 5-day return as the next state
      '''
      one_day_return = self.returns[self.current_step]
      five_day_return = self.returns[self.current_step - 5]
      return np.array([one_day_return, five_day_return])


    def _get_reward(self):
      '''
      calculate reward
      when posiiton is long, use 1 day return as reward
      when position is short, use negative 1 day return as reward
      FIXME: modify this function for experiment
      '''
      one_day_return = self.returns[self.current_step]
      if self.position == 1:
        return one_day_return
      elif self.position == -1:
        return -one_day_return
      else:
        return 0

    def step(self, action):
      ''' 
      step function
      1. change the position based on given action
      2. calculate next state
      3. get reward from state transition (one day return)
      4. return a transition tuple

      
      '''
      assert self.action_space.contains(action), f"{action} is an invalid action"

      # current state
      state = self.current_state

      # Calculate reward based on the chosen action

      # FIXME: Modify this part to add transcation cost
      if action == 0:  # short
          self.position = -1
      elif action == 1:  # stay
          self.position = 0
      elif action == 2:  # long
          self.position = 1

      #next_state
      next_state = self._get_next_state()
      self.current_state = next_state

      #reward
      reward = self._get_reward()

      # Update the current step
      self.current_step += 1

      # Check if the episode is done (reached the end of the stock prices data)
      done = self.current_step >= len(self.returns) - 1
      return state, action,reward,next_state, done

    def reset(self,stock_returns):
      '''
      reset the env with a new series of returns
      '''
      self.returns = stock_returns
      self.current_step = 5
      return self._get_next_state()


    def close(self):
      pass