from game import Game
from qplayer import QPlayer
from player import Player
import pickle
import os
from tqdm import tqdm

if __name__ == "__main__":
    MODE = 0 # change MODE to 1, to play with AI
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
    else:
        p2 = Player(ROW, COL)
        p1.trying_rate = 0
        SAVE_PERIOD = 1

    # start training
    game = Game(p1, p2, ROW, COL)
    for i in tqdm(range(1000000)):
        game.gameplay()
    with open(PATH, 'wb') as f:
        pickle.dump(p1.Q, f)
