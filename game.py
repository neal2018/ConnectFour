from player import Player
import numpy as np
from typing import Tuple
from constants import (DRAW,
                       LOSE,
                       WIN,
                       PLAYER1,
                       PLAYER2,
                       WIN_CNT)


class Game:
    def __init__(self, player1: Player, player2: Player, row: int = 6, col: int = 7):
        self.player1 = player1
        self.player2 = player2
        self.row = row
        self.col = col
        self.initial()

    def initial(self) -> None:
        # state: 0: empty, 1: self, 2: opponent
        self.state1 = np.zeros((self.row, self.col), dtype=int)  # for player 1
        self.state2 = np.zeros((self.row, self.col), dtype=int)  # for player 2
        self.step = 0
        self.state_count = np.zeros(self.col, dtype=int)
        return

    def gameplay(self) -> Tuple[int, int]:
        """
        return a tuple that contains the results of player1 and player2
        the result can be WIN, LOSE, or DRAW
        """
        while True:
            # player1 plays
            column = self.player1.nextstep(self.state1)
            is_end, result = self.check_result(column, PLAYER1)
            if is_end:
                self.endgame(result)
                return result
            self.step += 1
            if self.step >= self.row * self.col:
                self.endgame((DRAW, DRAW))
                return (DRAW, DRAW)
            # player2 plays
            column = self.player2.nextstep(self.state2)
            is_end, result = self.check_result(column, PLAYER2)
            if is_end:
                self.endgame(result)
                return result

            self.step += 1
            if self.step >= self.row * self.col:
                self.endgame((DRAW, DRAW))
                return (DRAW, DRAW)

    def check_result(self, column: int, player_number: int) -> Tuple[bool, Tuple[int, int]]:
        # if out of scope, lose
        if self.state_count[column] >= self.row:
            if player_number == PLAYER1:
                return True, (LOSE, WIN)
            else:
                return True, (WIN, LOSE)
        # update state
        row = self.row - self.state_count[column] - 1
        self.state_count[column] += 1
        if player_number == PLAYER1:
            self.state1[row][column] = PLAYER1
            self.state2[row][column] = PLAYER2
        else:
            self.state1[row][column] = PLAYER2
            self.state2[row][column] = PLAYER1

        return self.haswon(self.state1, row, column)

    def haswon(self, state: np.ndarray, i: int, j: int) -> Tuple[bool, Tuple[int, int]]:
        direction = ((i-WIN_CNT, j, 1, 0), (i, j-WIN_CNT, 0, 1),
                     (i-WIN_CNT, j-WIN_CNT, 1, 1), (i-WIN_CNT, j+WIN_CNT, 1, -1))

        for start_i, start_j, d_i, d_j in direction:
            cnt = 0
            symbol = 0
            for k in range(2*WIN_CNT+1):
                cur_i = start_i + k*d_i
                cur_j = start_j + k*d_j
                if not (0 <= cur_i < self.row and 0 <= cur_j < self.col):
                    continue
                if symbol == 0 or symbol != state[cur_i][cur_j]:
                    symbol = state[cur_i][cur_j]
                    cnt = 1
                else:
                    cnt += 1
                    if cnt >= 4 and symbol != 0:
                        if symbol == 1:
                            return True, (WIN, LOSE)
                        else:
                            return True, (LOSE, WIN)

        return False, (DRAW, DRAW)

    def endgame(self, result: Tuple[int, int]) -> None:
        self.player1.endgame(result[0], self.state1)
        self.player2.endgame(result[1], self.state2)
        self.initial()
