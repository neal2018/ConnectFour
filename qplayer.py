from player import Player
import pickle
import random
import numpy as np
import os


class QPlayer(Player):
    def __init__(self, Q, row=6, col=7):
        super().__init__(row, col)
        self.Q = Q
        self.alpha = 0.4
        self.gamma = 0.8

        self.last_state = None
        self.last_step = None
        self.trying_rate = 0.5

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
        if result:
            self.updateQ(self.last_state, state, self.last_step, 50)
        else:
            self.updateQ(self.last_state, state, self.last_step, -70)
        self.last_state = None
        self.last_step = None

    def get_step(self, state):
        if random.random() < self.trying_rate:
            return random.randrange(self.col)

        if state not in self.Q:
            return random.randrange(self.col)
        else:
            return np.argmax(self.Q[state])

    def updateQ(self, last_state, cur_state, last_step, reward):
        if last_state not in self.Q:
            self.Q[last_state] = [0]*self.col
        if cur_state not in self.Q:
            self.Q[cur_state] = [0]*self.col

        self.Q[last_state][last_step] = np.clip(
            (1-self.alpha)*self.Q[last_state][last_step] + self.alpha*(reward + self.gamma*max(self.Q[cur_state])), -100, 100)

    def encode(self, state):
        return ''.join(''.join([str(c) for c in row]) for row in state)
