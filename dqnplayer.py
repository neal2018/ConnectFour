from player import Player
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, ROW, COL):
        super().__init__()
        self.ROW = ROW
        self.COL = COL
        self.kernel_number1 = 4

        self.conv1 = nn.Conv2d(1, 10, self.kernel_number1)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(10 * (ROW-3) * (COL-3), 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, COL)

    def forward(self, x):
        # x: bsz * 1 * ROW * COL
        bsz = x.shape[0]
        x = self.conv1(x)  # bsz * 10 * (ROW-3) * (COL-3)
        x = self.relu(x)
        x = x.view(bsz, -1)  # bsz * (10 * (ROW-3) * (COL-3))
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


class DQNPlayer(Player):
    def __init__(self, model, target_model, row=6, col=7):
        super().__init__(row, col)
        self.model = model
        self.target_model = target_model
        self.alpha = 0.4
        self.gamma = 0.8

        self.last_state = None
        self.last_step = None
        self.trying_rate = 0.5

        self.loss_fn = F.mse_loss
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def nextstep(self, state):
        state = self.encode(state)
        if self.last_step:
            self.updateQ(self.last_state, state, self.last_step, -0.1)

        self.last_state = state
        step = self.get_step(state)
        self.last_step = step
        return step

    def endgame(self, result, state):
        state = self.encode(state)
        if result==True:
            self.updateQ(self.last_state, state,
                         self.last_step, 50, is_end=True)
        elif result=="draw":
            self.updateQ(self.last_state, state,
                         self.last_step, 10, is_end=True)
        else:
            self.updateQ(self.last_state, state,
                         self.last_step, -70, is_end=True)

        self.last_state = None
        self.last_step = None

    def get_step(self, state):
        if random.random() < self.trying_rate:
            return random.randrange(self.col)

        self.model.eval()
        with torch.no_grad():
            Q_values = self.model(state)
        return np.argmax(Q_values)

    def updateQ(self, last_state, cur_state, last_step, reward, is_end=False):
        Q_sa = self.model(last_state).gather(
            1, torch.LongTensor([[last_step]]))

        next_Q_sa = self.target_model(cur_state).max(1)[0].detach()

        except_Q_sa = (1-is_end)*next_Q_sa*self.gamma + \
            torch.LongTensor([reward])*(1-is_end)

        loss = self.loss_fn(Q_sa, except_Q_sa.unsqueeze(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def encode(self, state):
        return torch.Tensor(state).unsqueeze(0).unsqueeze(0)
