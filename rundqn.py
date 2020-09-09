from game import Game
from dqnplayer import DQNPlayer, DQN
from qplayer import QPlayer
from player import Player
import pickle
import os
from tqdm import tqdm
import torch
import torch.nn as nn

if __name__ == "__main__":
    MODE = 0  # change MODE to 1, to play with AI
    ROW = 5
    COL = 4
    PATH = f"saved_model/dqn_{ROW}_{COL}.pth"

    model = DQN(ROW, COL)
    target_model = DQN(ROW, COL)

    if os.path.exists(PATH):
        model.load_state_dict(torch.load(PATH))
        target_model.load_state_dict(torch.load(PATH))

    p1 = DQNPlayer(model, target_model, ROW, COL)

    if MODE == 0:
        # self play
        p2 = DQNPlayer(model, target_model, ROW, COL)
    elif MODE == 1:
        # man play
        p2 = Player(ROW, COL)
        p1.trying_rate = 0
    elif MODE == 2:
        # vs QPlayer
        QPATH = f"saved_model/{ROW}_{COL}.pkl"
        with open(QPATH, 'rb') as f:
            Q = pickle.load(f)
        p2 = QPlayer(Q, ROW, COL)
        p2.trying_rate = 0


    cnt = 0
    win_cnt = 0
    # start training
    game = Game(p2, p1, ROW, COL)
    for i in tqdm(range(100000)):
        res = game.gameplay()

        if i % 20 == 19:
            p1.update_target_model()

    torch.save(model.state_dict(), PATH)
