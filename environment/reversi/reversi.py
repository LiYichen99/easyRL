import random

import numpy as np
from loguru import logger
from environment.reversi.utils import *


class Reversi:

    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size))
        self.board[board_size // 2 - 1, board_size // 2 - 1] = BLACK_PLAYER
        self.board[board_size // 2, board_size // 2] = BLACK_PLAYER
        self.board[board_size // 2 - 1, board_size // 2] = WHITE_PLAYER
        self.board[board_size // 2, board_size // 2 - 1] = WHITE_PLAYER
        self.info = dict(board_size=board_size, current_player=BLACK_PLAYER, winner=False)

    def reset(self):
        self.__init__(self.board_size)
        return self.board, self.info

    def step(self, action):
        row, col = action // self.board_size, action % self.board_size
        if not self._is_legal_action(row, col, DIRECTIONS):
            logger.error(f'{row, col} is not legal.')
            exit()
            return
        self.board[row, col] = self.info['current_player']
        reverse_list = self._get_reverse_list(row, col, DIRECTIONS)
        reverse_num = len(reverse_list)
        for x, y in reverse_list:
            self.board[x, y] = self.info['current_player']
        done, truncated = False, False
        self.info['current_player'] = -self.info['current_player']
        if self._has_legal_actions():
            return self.board, reverse_num * 0.1, done, truncated, self.info
        self.info['current_player'] = -self.info['current_player']
        if self._has_legal_actions():
            return self.board, reverse_num * 0.1, done, truncated, self.info

        done = True
        count = np.sum(self.board)
        if count == 0:
            return self.board, reverse_num * 0.1, done, truncated, self.info
        elif count > 0:
            winner = BLACK_PLAYER
        else:
            winner = WHITE_PLAYER
        self.info['winner'] = winner

        if winner == self.info['current_player']:
            return self.board, reverse_num * 0.1 + 1, done, truncated, self.info
        else:
            return self.board, reverse_num * 0.1 - 1, done, truncated, self.info

    def get_legal_actions(self):
        legal = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self._is_legal_action(x, y, DIRECTIONS):
                    legal.append(x * self.board_size + y)
        return legal

    def _has_legal_actions(self):
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self._is_legal_action(x, y, DIRECTIONS):
                    return True
        return False

    def _is_legal_action(self, row, col, direction):
        if not (0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row, col] == EMPTY):
            return False
        color = self.info['current_player']
        for dx, dy in direction:
            x = row + dx
            y = col + dy
            if not (0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == -color):
                continue
            x += dx
            y += dy
            while 0 <= x < self.board_size and 0 <= y < self.board_size:
                if self.board[x, y] == EMPTY:
                    break
                elif self.board[x, y] == color:
                    return True
                x += dx
                y += dy
        return False

    def _get_reverse_list(self, row, col, direction):
        reverse_set = set()
        color = self.info['current_player']
        for dx, dy in direction:
            x = row + dx
            y = col + dy
            if not (0 <= x < self.board_size and 0 <= y < self.board_size) or self.board[x, y] != -color:
                continue
            reverse_temp = [(x, y)]
            x += dx
            y += dy
            while 0 <= x < self.board_size and 0 <= y < self.board_size:
                if self.board[x, y] == EMPTY:
                    break
                elif self.board[x, y] == -color:
                    reverse_temp.append((x, y))
                    x += dx
                    y += dy
                else:
                    reverse_set.update(reverse_temp)
                    break
        return list(reverse_set)

    def render(self):
        for x in range(self.board_size):
            for y in range(self.board_size):
                print(F'\t{CHESS_DISPLAY[self.board[x, y]]}', end='')
            print()


class WrapperReversi(Reversi):
    def reset(self):
        self.__init__(self.board_size)
        return normalize_obs(self.board, self.info), self.info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return normalize_obs(obs, info), reward, done, truncated, info


if __name__ == '__main__':
    game = WrapperReversi()
    obs, info = game.reset()
    while True:
        game.render()
        actions = game.get_legal_actions()
        action = random.choice(game.get_legal_actions())
        obs, reward, done, truncated, info = game.step(action)
        print(obs.shape)
        print(action, reward)
        if done or truncated:
            break