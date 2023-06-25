import numpy as np
from loguru import logger
import random
from environment.gobang.utils import *


class GoBang:

    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size))
        self.info = dict(board_size=self.board_size, current_player=BLACK_PLAYER, winner=EMPTY)
        self.steps = 0

    def reset(self):
        self.__init__(self.board_size)
        return self.board, self.info

    def step(self, action):
        row, col = action // self.board_size, action % self.board_size
        if self.board[row, col] != EMPTY:
            logger.error(f'{row, col} is not empty.')
            exit()
            return
        self.board[row, col] = self.info['current_player']
        direction_num = [self._cal_num(row, col, direction) for direction in DIRECTIONS]
        done, truncated = False, False
        for dir_num in direction_num:
            if dir_num >= 5:
                done = True
        # dir_num_reward = sum(0.25 * num for num in direction_num)
        self.steps += 1
        if done:
            self.info['winner'] = self.info['current_player']
            return self.board, 0, done, truncated, self.info
        elif not np.any(self.board == EMPTY):
            done = True
            return self.board, -0.2, done, truncated, self.info
        else:
            self.info['current_player'] = -self.info['current_player']
            return self.board, -1, done, truncated, self.info

    def get_legal_actions(self):
        legal = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == EMPTY:
                    legal.append(x * self.board_size + y)
        return legal

    def _cal_num(self, row, col, direction):
        num = 1
        cx, cy = direction
        color = self.board[row, col]
        x, y = row + cx, col + cy
        while x >= 0 and x < self.board_size and y >= 0 and y < self.board_size and self.board[x, y] == color:
            num += 1
            x, y = x + cx, y + cy
        x, y = row - cx, col - cy
        while x >= 0 and x < self.board_size and y >= 0 and y < self.board_size and self.board[x, y] == color:
            num += 1
            x, y = x - cx, y - cy
        return num

    def render(self):
        for x in range(self.board_size):
            for y in range(self.board_size):
                print(F'\t{CHESS_DISPLAY[self.board[x, y]]}', end='')
            print()


class WrapperGoBang(GoBang):
    def reset(self):
        self.__init__(self.board_size)
        return normalize_obs(self.board, self.info), self.info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return normalize_obs(obs, info), reward, done, truncated, info


if __name__ == '__main__':
    game = WrapperGoBang()
    obs, info = game.reset()
    while True:
        game.render()
        action = random.choice(game.get_legal_actions())
        obs, reward, done, truncated, info = game.step(*action)
        print(obs.shape)
        print(action, reward)
        if done or truncated:
            break
