class Game:
    def __init__(self, player1, player2, row=6, col=7):
        self.player1 = player1
        self.player2 = player2
        self.row = row
        self.col = col
        self.state1 = self.initial_state()
        self.state2 = self.initial_state()

    def initial_state(self):
        return [[0]*self.col for _ in range(self.row)]

    def gameplay(self):
        while True:
            position = self.player1.nextstep(self.state1)
            res = self.check_result(position, 1)
            if res:
                self.endgame(res)
                break
            position = self.player2.nextstep(self.state2)
            res = self.check_result(position, 2)
            if res:
                self.endgame(res)
                break

    def check_result(self, position: int, player_number: int):
        """
        player_number: int, 1 or 2
        """
        player_number -= 1
        res = [True, True]
        row_position = None

        for i in reversed(range(self.row)):
            if self.state1[i][position] == 0:
                row_position = i
                break
        if row_position is None:
            res[player_number] = False
            return res

        self.state1[row_position][position] += 1+player_number
        self.state2[row_position][position] += 1 + (not player_number)
        return self.haswon(self.state1, row_position, position)

    def haswon(self, state, i, j):
        direction = ((i-4, j, 1, 0), (i, j-4, 0, 1),
                     (i-4, j-4, 1, 1), (i-4, j+4, 1, -1))

        for start_i, start_j, d_i, d_j in direction:
            cnt = 0
            symbol = None
            for k in range(9):
                cur_i = start_i + k*d_i
                cur_j = start_j + k*d_j
                if not (0 <= cur_i < self.row and 0 <= cur_j < self.col):
                    continue
                if symbol is None or symbol != state[cur_i][cur_j]:
                    symbol = state[cur_i][cur_j]
                    cnt = 1
                else:
                    cnt += 1
                    if cnt == 4 and symbol != 0:
                        return (True, False) if symbol == 1 else (False, True)
        return False

    def endgame(self, res):
        self.player1.endgame(res[0], self.state1)
        self.player2.endgame(res[1], self.state2)
        self.state1 = self.initial_state()
        self.state2 = self.initial_state()

