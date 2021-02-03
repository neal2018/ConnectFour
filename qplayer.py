from player import Player
from typing import Dict
import pickle
import random
import numpy as np
import os
from constants import (DRAW,
                       LOSE,
                       WIN)


class QPlayer(Player):
    def __init__(self, Q: Dict[int, float], row: int = 6, col: int = 7):
        super().__init__(row, col)
        self.Q = Q
        self.alpha = 0.4
        self.gamma = 0.8

        self.last_state = None
        self.last_step = None
        self.trying_rate = 0.5

    def nextstep(self, state: np.ndarray):
        state = self.encode(state)

        if self.last_step:
            self.updateQ(self.last_state, state, self.last_step, -0.1)

        self.last_state = state
        step = self.get_step(state)
        self.last_step = step
        return step

    def endgame(self, result: int, state: np.ndarray):
        state = self.encode(state)
        if result == WIN:
            self.updateQ(self.last_state, state, self.last_step, 50)
        elif result == LOSE:
            self.updateQ(self.last_state, state, self.last_step, -70)
        self.last_state = None
        self.last_step = None

    def get_step(self, state: np.ndarray):
        if random.random() < self.trying_rate:
            return random.randrange(self.col)

        if state not in self.Q:
            return random.randrange(self.col)
        else:
            return np.argmax(self.Q[state])

    def updateQ(self, last_state: np.ndarray, cur_state: np.ndarray, last_step: int, reward: float):
        if last_state not in self.Q:
            self.Q[last_state] = [0]*self.col
        if cur_state not in self.Q:
            self.Q[cur_state] = [0]*self.col

        self.Q[last_state][last_step] = np.clip(
            (1-self.alpha)*self.Q[last_state][last_step] + self.alpha*(reward + self.gamma*max(self.Q[cur_state])), -100, 100)

    def encode(self, state: np.ndarray) -> int:
        result = 0
        for row in state:
            for c in row:
                result = result*3 + int(c)
        return result


if __name__ == "__main__":
    from tqdm import tqdm
    from game import Game

    MODE = 0  # change MODE to 1, to play with AI
    ROW = 5
    COL = 4
    PATH = f"saved_model/{ROW}_{COL}.pkl"

    # loading saved Q-matrix
    if not os.path.exists(PATH):
        Q = {}
        with open(PATH, 'wb') as f:
            pickle.dump(Q, f)
    with open(PATH, 'rb') as f:
        Q = pickle.load(f)

    p1 = QPlayer(Q, ROW, COL)

    if MODE == 0:
        p2 = QPlayer(Q, ROW, COL)
        # start training
        game = Game(p1, p2, ROW, COL)
        for i in tqdm(range(200000)):
            game.gameplay()
        with open(PATH, 'wb') as f:
            pickle.dump(p1.Q, f)
    else:
        p2 = Player(ROW, COL)
        p1.trying_rate = 0
        while True:
            game = Game(p1, p2, ROW, COL)
            game.gameplay()
