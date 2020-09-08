class Player:
    def __init__(self, row=6, col=7):
        self.row = row
        self.col = col

    def nextstep(self, state):
        for row in state:
            print(row)
        step = int(input("enter next step column\n"))
        while not (0 <= step < self.col):
            step = int(input("enter next step column"))
        return step

    def endgame(self, result, state):
        if result:
            print("you win")
        else:
            print("you lose")
