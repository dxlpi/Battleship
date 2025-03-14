import random
from copy import deepcopy
from collections import defaultdict
from Models import MCModel

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

class BattleShip:
    def __init__(self, action_cost=-1, hit_empty=-10, hit_ship=10, destroy_ship=30):
        self.destroy_ship = destroy_ship
        self.hit_ship = hit_ship
        self.hit_empty = hit_empty
        self.action_cost = action_cost
        self.board = [[0 for i in range(10)] for j in range(10)]
        self.seen_board = deepcopy(self.board)
        self.ships_belong = {}
        self.ships = defaultdict(list)

    def start(self):
        ship_sizes = [5, 4, 3, 3, 2]

        for belong_index, ship_size in enumerate(ship_sizes):
            self.__place_ship(ship_size, belong_index)

        #print(self.ships)
        #print("Actual Board")
        #self.visualize(self.board)

    def __place_ship(self, ship_size, belong_index):
        placed = False

        while not placed:
            horizontal = random.choice([0, 1])

            if horizontal == 1:
                x = random.randint(0, 9)
                y = random.randint(0, 9 - ship_size)


                if all(self.board[x][y + i] == 0 for i in range(ship_size)):
                    for i in range(ship_size):
                        self.board[x][y + i] = 1
                        self.ships_belong[x * 10 + (y + i)] = belong_index
                        self.ships[belong_index].append(x * 10 + (y + i))
                    placed = True
            else:
                x = random.randint(0, 9 - ship_size)
                y = random.randint(0, 9)

                if all(self.board[x + i][y] == 0 for i in range(ship_size)):
                    for i in range(ship_size):
                        self.board[x + i][y] = 1
                        self.ships_belong[(x + i) * 10 + y] = belong_index
                        self.ships[belong_index].append((x + i) * 10 + y)
                    placed = True

    def step(self, x, y):
        if self.board[x][y] == 1:
            val = x * 10 + y
            belong_index = self.ships_belong[val]
            self.ships[belong_index].remove(val)
            self.seen_board[x][y] = 1

            if self.ships[belong_index]:
                return self.hit_ship + self.action_cost

            return self.hit_ship + self.action_cost + self.destroy_ship
        else:
            self.seen_board[x][y] = -1
            return self.hit_empty + self.action_cost

        #self.visualize(self.seen_board)

    def end(self):
        if self.ships_left() == 0:
            return True

        return False

    def reset(self):
        self.board = [[0 for i in range(10)] for j in range(10)]
        self.seen_board = deepcopy(self.board)
        self.ships = defaultdict(list)
        #self.visualize(self.seen_board)

    def visualize(self, board):
        print('\n'.join([" ".join(map(str, row)) for row in board]))
        print("- - - - - - - - - -")

    def current_state(self):
        return self.seen_board

    def ships_left(self):
        count = 0

        for ship in self.ships.values():
            if ship:
                count += 1

        return count


if __name__ == '__main__':
    random.seed(10)
    
    # Define hyperparameters
    num_episodes = 30000  # Number of episodes for Monte Carlo training
    epsilon = 0.1        # Exploration rate
    gamma = 0.99         # Discount factor

    env = BattleShip(action_cost=-1, hit_empty=-2, hit_ship=3, destroy_ship=10)
    model = MCModel(env, num_episodes=num_episodes, epsilon=epsilon, gamma=gamma)
    q_table, rewards = model.run()

    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(f"Monte Carlo Learning\nEpisodes: {num_episodes}, Epsilon: {epsilon}, Gamma: {gamma}")
    plt.show()