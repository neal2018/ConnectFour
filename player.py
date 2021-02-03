import numpy as np
from constants import (DRAW,
                       LOSE,
                       WIN)


class Player:
    def __init__(self, row: int = 6, col: int = 7):
        self.row = row
        self.col = col

    def nextstep(self, state: np.ndarray) -> int:
        for row in state:
            print(row)
        step = int(input("enter next step column\n"))
        while not (0 <= step < self.col):
            step = int(input("invalid move, enter next step column again\n"))
        return step

    def endgame(self, result: int, state: np.ndarray):
        if result == WIN:
            print("you win")
        elif result == LOSE:
            print("you lose")
        elif result == DRAW:
            print("draw")
        else:
            raise ValueError("result unknown")
        for row in state:
            print(row)


if __name__ == "__main__":
    from game import Game
    ROW = 5
    COL = 4
    p1 = Player(ROW, COL)
    p2 = Player(ROW, COL)
    game = Game(p1, p2, ROW, COL)
    game.gameplay()
