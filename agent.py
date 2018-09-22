from model import QNet, Dueling_DQN
import torch
import torch.nn.functional as F
import torch.optim as optim
import ReplayBuffer as replay
import random
import numpy as np
class Agent():

    def __init__(self, state_size, num_actions, lr=0.01, buffer_size=int(1e5), batch_size=64, seed=999, update_frequency=4, gamma=0.99, tau=1e-3, use_double_q=True, use_dualing_net=True):

        self.state_size = state_size
        self.num_actions = num_actions
        self.batch_size = batch_size

        # Train Params
        self.gamma = gamma
        self.tau = tau

        self.use_double_q=use_double_q
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("THE DEVICE IS")
        print(self.device)

        if(use_dualing_net):

            self.qnet_local = Dueling_DQN(state_size, num_actions).to(self.device)
            self.qnet_target = Dueling_DQN(state_size, num_actions).to(self.device)
        else:
            self.qnet_local = QNet(state_size, num_actions).to(self.device)
            self.qnet_target = QNet(state_size, num_actions).to(self.device)

        print(self.qnet_local)

        self.optimiser = optim.Adam(self.qnet_local.parameters(), lr=lr)


        self.memory = replay.ReplayBuffer(num_actions, buffer_size, batch_size, seed, self.device)

        self.time_step = 0

        self.update_frequency = update_frequency

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.num_actions))

    def step(self, state, action, reward, next_state, done):

        # Save an experience in the memory buffer

        self.memory.add(state, action, reward, next_state, done)

        self.time_step = (self.time_step + 1) % self.update_frequency

        if self.time_step == 0:
            if(len(self.memory) > self.batch_size):
                experience = self.memory.sample()
                self.learn(experience)

    def learn(self, experiences):
        """

        :param experiences:
        :return:
        """

        states, actions, rewards, next_states, dones = experiences

        # Get the maxiumum Q value
        Q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Calculate Q values for local model
        if self.use_double_q:
            _,next_action = self.qnet_local(next_states).max(1, keepdim=True)

            action_value = self.qnet_target(next_states).gather(1, next_action)

            Q_targets = rewards + (self.gamma * action_value * (1 - dones))
        else:
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get the expected Q value
        Q_expected = self.qnet_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.soft_update(self.qnet_local, self.qnet_target)

    def soft_update(self, local_model, target_model):
        """

        :param local_model:
        :param target_model:
        :return:
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau) * target_param.data)





