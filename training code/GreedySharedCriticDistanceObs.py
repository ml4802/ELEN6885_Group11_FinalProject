import random
import gym
import collections
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import time
import math
from ma_gym.wrappers import Monitor
from datetime import datetime

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n): # Grabs n random samples from the replay memory
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)                                            # State list
            a_lst.append(a)                                            # Action list
            r_lst.append(r)                                            # Reward list
            s_prime_lst.append(s_prime)                                # NewState list
            done_mask_lst.append((np.ones(len(done)) - done).tolist()) # Done list
        # The np.array over a_lst is much faster than converting a list of numpy arrays to tensor directly
        return torch.tensor(s_lst, dtype=torch.float), \
               torch.tensor(np.array(a_lst), dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), \
               torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self): # Number of elements in the buffer
        return len(self.buffer)

class MuNet(nn.Module):
    # Actor Network
    def __init__(self, observation_space, action_space, weight_multiplier = 1):
        super(MuNet, self).__init__()
        self.num_agents = len(observation_space)
        self.action_space = action_space
        for agent_i in range(self.num_agents): # For each agent, instantiate a Individual Actor network
            n_obs = observation_space[agent_i].shape[0] + (self.num_agents-1) # To add the distance observation from all the other agents
            print('N_obs in MuNet',n_obs)
            num_action = action_space[agent_i].n
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, int(128*weight_multiplier)),
                                                                    nn.ReLU(),
                                                                    nn.Linear(int(128*weight_multiplier), int(64*weight_multiplier)),
                                                                    nn.ReLU(),
                                                                    nn.Linear(int(64*weight_multiplier), num_action)))

    def forward(self, obs): # Returns the actions of the agents
        action_logits = [torch.empty(1, _.n) for _ in self.action_space]
        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)
            action_logits[agent_i] = x

        return torch.cat(action_logits, dim=1)

class QNet(nn.Module):
    # Critic network
    def __init__(self, observation_space, action_space,weight_multiplier = 1):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        total_action = sum([_.n for _ in action_space])
        total_obs = sum([_.shape[0] for _ in observation_space]) + self.num_agents*(self.num_agents-1)
        print("Total_obs in QNet",total_obs)
        for agent_i in range(self.num_agents):
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(total_obs + total_action, int(128*weight_multiplier)),
                                                                    nn.ReLU(),
                                                                    nn.Linear(int(128*weight_multiplier), int(64*weight_multiplier)),
                                                                    nn.ReLU(),
                                                                    nn.Linear(int(64*weight_multiplier), 1)))

    def forward(self, obs, action): # Returns the q value evaluations for the agent possible actions
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        x = torch.cat((obs.view(obs.shape[0], obs.shape[1] * obs.shape[2]),
                       action.view(action.shape[0], action.shape[1] * action.shape[2])), dim=1)
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(x)

        return torch.cat(q_values, dim=1)

def add_distance_obs(state,n_agents):
    sqrt2 = math.sqrt(2)
    if n_agents == 2:
        dist = math.dist(state[0][1:3],state[1][1:3])/sqrt2 
        state[0].append(dist)
        state[1].append(dist)
        return state
    elif n_agents == 3:
        distsfor0to1 = math.dist(state[0][1:3],state[1][1:3])/sqrt2 
        distsfor0to2 = math.dist(state[0][1:3],state[2][1:3])/sqrt2 
        distsfor1to2 = math.dist(state[1][1:3],state[2][1:3])/sqrt2 
        state[0].append(distsfor0to1) #1
        state[0].append(distsfor0to2) #2
        state[1].append(distsfor0to1) #0
        state[1].append(distsfor1to2) #2
        state[2].append(distsfor0to2) #0
        state[2].append(distsfor1to2) #1
        return state
    elif n_agents == 4:
        distsfor0to1 = math.dist(state[0][1:3],state[1][1:3])/sqrt2 
        distsfor0to2 = math.dist(state[0][1:3],state[2][1:3])/sqrt2 
        distsfor1to2 = math.dist(state[1][1:3],state[2][1:3])/sqrt2 
        distsfor0to3 = math.dist(state[0][1:3],state[3][1:3])/sqrt2 
        distsfor1to3 = math.dist(state[1][1:3],state[3][1:3])/sqrt2 
        distsfor2to3 = math.dist(state[2][1:3],state[3][1:3])/sqrt2 
        state[0].append(distsfor0to1) #1
        state[0].append(distsfor0to2) #2
        state[0].append(distsfor0to3) #3
        state[1].append(distsfor0to1) #0
        state[1].append(distsfor1to2) #2
        state[1].append(distsfor1to3) #3
        state[2].append(distsfor0to2) #0
        state[2].append(distsfor1to2) #1
        state[2].append(distsfor2to3) #3
        state[3].append(distsfor0to3) #0
        state[3].append(distsfor1to3) #1
        state[3].append(distsfor2to3) #2
        return state
    return "Only supports 2,3,4 agents"

def soft_update(net, net_target, tau): # Copying parts of the weights of the current network to the target network
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, gamma, batch_size):
    state, action, reward, next_state, done_mask = memory.sample(batch_size) # Samples batch_size # of runs to train with

    next_state_action_logits = mu_target(next_state)
    _, n_agents, action_size = next_state_action_logits.shape
    next_state_action_logits = next_state_action_logits.view(batch_size * n_agents, action_size)
    next_state_action = F.gumbel_softmax(logits=next_state_action_logits, tau=0.1, hard=True)
    next_state_action = next_state_action.view(batch_size, n_agents, action_size)

    target = reward + gamma * q_target(next_state, next_state_action) * done_mask
    q_loss = F.smooth_l1_loss(q(state, action), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    state_action_logits = mu(state)
    state_action_logits = state_action_logits.view(batch_size * n_agents, action_size)
    state_action = F.gumbel_softmax(logits=state_action_logits, tau=0.1, hard=True)
    state_action = state_action.view(batch_size, n_agents, action_size)

    mu_loss = -q(state, state_action).mean()  # That's all for the policy loss.
    q_optimizer.zero_grad()
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()

def test(env, num_episodes, mu, n_agents,render=False): # Does not use the critic network, just uses the actor networks
    
    score = np.zeros(env.n_agents)
    epsilon_test = 0.01

    with torch.no_grad():
        for episode_i in range(num_episodes):
            state = add_distance_obs(env.reset(),n_agents)
            done = [False for _ in range(env.n_agents)]

            while not all(done):
                if render == True:
                        env.render()
                        time.sleep(0.0005)
                
                if np.random.rand() < epsilon_test:
                    action = env.action_space.sample()
                else:
                    action_logits = mu(torch.Tensor(state).unsqueeze(0))
                    action = action_logits.argmax(dim=2).squeeze(0).data.cpu().numpy().tolist()

                next_state, reward, done, info = env.step(action)
                next_state = add_distance_obs(next_state,n_agents) # Added
                score += np.array(reward)
                state = next_state

    return sum(score / num_episodes)


def main(env_name, lr_mu, lr_q, tau, gamma, batch_size, buffer_limit, max_episodes, log_interval, test_episodes,
         warm_up_steps, update_iter, gumbel_max_temp, gumbel_min_temp, grid_size, n_agents, n_trees, max_steps, agent_view,n_obstacles,step_cost
         ,max_steps_without_reward,tree_strength,weight_multiplier,render_interval,tree_cutdown_reward):

    load_saved_models = False
    
    gym.envs.register(
        id='my_Lumberjacks-v1',
        entry_point='ma_gym.envs.lumberjacks:Lumberjacks', # Points to the lumberjack class object
        kwargs={'tree_cutdown_reward':tree_cutdown_reward,'tree_strength':tree_strength, 'max_steps_without_reward':max_steps_without_reward, 'n_obstacles':n_obstacles,'n_agents': n_agents, 'n_trees':n_trees, 'full_observable': False, 'step_cost': step_cost, 'grid_shape':(grid_size,grid_size),'agent_view':agent_view,'max_steps':max_steps} # Add additional args
    )

    env = gym.make('my_Lumberjacks-v1')
    test_env = gym.make('my_Lumberjacks-v1')
    # Parameyers to adjust about the environment

    memory = ReplayBuffer(buffer_limit)

    q, q_target = QNet(env.observation_space, env.action_space,weight_multiplier), QNet(env.observation_space, env.action_space,weight_multiplier)
    # Add in part here to laod in the q network
    #q_target.load_state_dict(torch.load('/Users/mingliu/Documents/R Learning/Final Project Code/agent3_grid6/q_target_DIST_OBS_agents3_grid6.pt'))
    q_target.load_state_dict(q.state_dict())

    mu, mu_target = MuNet(env.observation_space, env.action_space,weight_multiplier), MuNet(env.observation_space, env.action_space,weight_multiplier)
    # Add in the part here to load in the mu network
    #mu_target.load_state_dict(torch.load('/Users/mingliu/Documents/R Learning/Final Project Code/agent3_grid6/mu_target_DIST_OBS_agents3_grid6.pt'))
    mu_target.load_state_dict(mu.state_dict())

    score = np.zeros(env.n_agents)

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)

#---- greedy
    epsilon = 1
    epsilon_decay = 0.999

    for episode_i in range(max_episodes):
        temperature = max(gumbel_min_temp,
                          gumbel_max_temp - (gumbel_max_temp - gumbel_min_temp) * (episode_i / (0.6 * max_episodes)))
        state = add_distance_obs(env.reset(),n_agents) # Adds the new observation
        done = [False for _ in range(env.n_agents)]
        step_i = 0

        epsilon *= epsilon_decay
        
        while not all(done):
              # Ensure epsilon doesn't go below a minimum value
            if np.random.rand() < epsilon:
                # Exploration: Random action
                action_one_hot = torch.zeros((n_agents,5))
                action = env.action_space.sample()
                for i in range(n_agents):
                    action_one_hot[i][action[i]] = 1
            else:
                action_logits = mu(torch.Tensor(state).unsqueeze(0))
                action_one_hot = F.gumbel_softmax(logits=action_logits.squeeze(0), tau=temperature, hard=True)
                action = torch.argmax(action_one_hot, dim=1).data.numpy()

            next_state, reward, done, info = env.step(action)
            next_state = add_distance_obs(next_state,n_agents) # Adds the new observation
            step_i += 1
            if step_i >= max_steps or (step_i < max_steps and not all(done)): # max steps set by use outside of the environment
                _done = [False for _ in done]
            else:
                _done = done
            memory.put((state, action_one_hot.data.numpy(), (np.array(reward)).tolist(), next_state,
                        np.array(_done, dtype=int).tolist()))
            score += np.array(reward)
            state = next_state

        if memory.size() > warm_up_steps:
            for i in range(update_iter):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, gamma, batch_size)
                soft_update(mu, mu_target, tau)
                soft_update(q, q_target, tau)

        if episode_i % log_interval == 0 and episode_i != 0:
            test_score = test(test_env, test_episodes, mu, n_agents,render=False)
            print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {} epsilon : {}"
                  .format(episode_i, max_episodes, sum(score / log_interval), test_score, memory.size(), epsilon))
            
#-- greedy
            if episode_i % render_interval == 0: # Render the tests
                #print("Showing 10 rendered games")
                #test(test_env, 3, mu, n_agents,render=True)
                print('Saved q and mu targets')
                torch.save(q_target.state_dict(),'G_q_target_DIST_OBS_agents{}_grid{}.pt'.format(n_agents,grid_size))
                torch.save(mu_target.state_dict(),'G_mu_target_DIST_OBS_agents{}_grid{}.pt'.format(n_agents,grid_size))

            if USE_WANDB:
                wandb.log({'episode': episode_i, 'test-score': test_score, 'gumbel_temperature': temperature,
                           'buffer-size': memory.size(), 'train-score': sum(score / log_interval),'epsilon':epsilon})
                
            score = np.zeros(env.n_agents)
    
    # Save the weights here
    # Perhaps save the optimizers as well later


    env.close()
    test_env.close()

if __name__ == '__main__':
    # Only edit these parts
 
    n_agent = 4# 2,3,4 # Change this
    mapsize = 10 # Pair change to 6, Emily change to 10

    # DON"T EDIT BELOW THIS LINE
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    #----------------------------------------------------------------------
    # DON"T EDIT BELOW THIS LINE
    if mapsize == 6: 
        num_trees = 12
        max_steps = 150
        num_obstacles = 5
        max_episodes = 30000
        multiplier = 1.25
    elif mapsize == 10: 
        num_trees = 36
        max_steps = 250
        num_obstacles = 12
        max_episodes = 30000
        multiplier = 1.25
    else:
        print("DO 4 or 6")
    tree_strength = []
    for i in range(1,n_agent+1):
        tree_strength.extend([int(i)]*int(num_trees / n_agent)) 

    print(tree_strength)
    print(multiplier)
    print(max_steps)
    print(max_episodes)
    print(num_trees)
    print(num_obstacles)
    
    kwargs = {'env_name': 'ma_gym:Lumberjacks-v0',
              'lr_mu': 0.0005,                      # Learning rate for Actors
              'lr_q': 0.001,                        # Learning rate for Critic
              'batch_size': 32,
              'tau': 0.005,
              'gamma': 0.99,
              'buffer_limit': 50000,
              'log_interval': 20,
              'render_interval': 500,              # Every this many games, show an example of the game
              'max_episodes': max_episodes,                # 10000 default
              'test_episodes': 5,
              'warm_up_steps': 2000,
              'update_iter': 10,
              'gumbel_max_temp': 10,
              'gumbel_min_temp': 0.1,
              'weight_multiplier':multiplier,
              
              'grid_size' : mapsize,
              'n_agents' : n_agent, #3,4 change this
              'n_trees' : num_trees,
              'tree_strength' : tree_strength        ,  # #Even spread based on the number of agents
              'max_steps' : max_steps,
              'agent_view' : (2,2),
              'n_obstacles' : num_obstacles,
              'step_cost' : -0.1,
              'tree_cutdown_reward': 10,
              'max_steps_without_reward' : 50000 # Don't really use this
    }

    myobj = datetime.now()
    USE_WANDB = True  # if enabled, logs data on wandb server
    if USE_WANDB:
        import wandb
        wandb.init(project='Greedy_Agent{}Size{}'.format(n_agent,mapsize), config={'type':'DISTANCEOBS','algo': 'maddpg', **kwargs, }, monitor_gym=True)

    main(**kwargs)