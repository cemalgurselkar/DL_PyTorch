"""
Deep Q Learning.
"""
import gymnasium as gym
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import matplotlib.pyplot as plt
import matplotlib
from collections import namedtuple, deque
from itertools import count
from IPython import display

env = gym.make("CartPole-v1", render_mode = "human")
device = "cuda"

Transition = namedtuple("Transition", ("state","action","next_state","reward"))

#replay memory
class ReplayMemory(object):

    def __init__(self, capacity): #initializer
        self.memory = deque([], maxlen=capacity)

    def push(self,*args):# save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):# choose a random sample
        return random.sample(self.memory, batch_size)

    def __len__(self):# memory length
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_action):
        super(DQN,self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128, n_action)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

#hyperparams
batch_size = 128
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 1000
tau = 0.005 #update rate of target network
lr = 1e-4

n_actions = env.action_space.n #type:ignore

state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
memory = ReplayMemory(1000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1*steps_done/eps_decay)
    steps_done += 1

    # eger sample eps_thresholddan büyükse ajan neural network ile action seçer, değilse rastgele action seçer
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    
episode_duration = []

def plot_duration(show_result = False):
    plt.figure(1)
    duration_t = torch.tensor(episode_duration, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.plot(duration_t.numpy())

    if len(duration_t) > 100:
        means = duration_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    display.display(plt.gcf())
    display.clear_output(wait=True)

def optimizer_model():

    #hafızada yeterli sayida deneyim var mı yok mu kontrol et, yoksa fomksiyondan çık
    if len(memory) < batch_size:
        return
    
    #hafizadan rastgele bir grup deneyim örneği alır
    transition = memory.sample(batch_size)
    #ayirma işlemi
    batch = Transition(*zip(*transition))
    # sonraki durumları none olmayan bir boolean maskesi oluşturur
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    #terminal olmayan tüm durumları tek bir tensor olarak birleştir
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # grup icindeki state, action ve reward ile birleştir.
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    expected_state_action_values = (next_state_values*gamma) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

num_episode = 200

for i_episode in range(num_episode):

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated #kaydettik veya yandık

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        # stransitonlari memoryde depola
        memory.push(state, action, next_state, reward)
        # state güncelle
        state = next_state

        #training
        optimizer_model()

        #update network parameters
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)
        if done:
            episode_duration.append(t+1)
            plot_duration()
            break
print("Done")
plot_duration(show_result=True)
plt.ioff()
plt.show()