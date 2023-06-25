import numpy as np

BLACK_PLAYER = 1
WHITE_PLAYER = -1
EMPTY = 0
DIRECTIONS = [[0, 1], [1, 1], [1, 0], [1, -1]]

WIN_REWARD = 10

CHESS_DISPLAY = {
    BLACK_PLAYER: 'X',
    WHITE_PLAYER: 'O',
    EMPTY: '.'
}


def normalize_obs(board, info):
    normalized_board = np.zeros((2, info['board_size'], info['board_size']))
    black_player_board = board * info['current_player']
    normalized_board[0] = (black_player_board == BLACK_PLAYER).astype(int)
    normalized_board[1] = (black_player_board == WHITE_PLAYER).astype(int)
    return normalized_board
