import numpy as np
from itertools import product

class TicTacToeState():
    """A tictactoe state is history or sequence of ordered pairs that represent places of X's and O's
    X will play the first move and O will play the second move. """
    def __init__(game, state: list[tuple]):
        game.state: np.ndarray = state
        game.player = 1 if np.sum(state) <= 0 else -1
        game.all_legal_actions = [action for action in product(state.shape) if state[action] == 0]
        game._result = game.game_result()
    
    def transition(game, action: tuple):
        new_state = game.state.copy()
        new_state[action] = game.player
        return TicTacToeState(new_state)
    
    def is_terminal(self) -> bool:
        """tic tac toe is terminal """

    
    def game_result(game) -> int | None:
        """ this property should return 
        1 if player #1 wins 
        -1 if player #2 wins 
        0 if there is a draw 
        None if result is unknown
        """
        rowsum = np.sum(game.board, 0)
        colsum = np.sum(game.board, 1)
        diag_sum_tl = game.board.trace()
        diag_sum_tr = game.board[::-1].trace()

        player_one_wins = any(rowsum == game.board_size)
        player_one_wins += any(colsum == game.board_size)
        player_one_wins += (diag_sum_tl == game.board_size)
        player_one_wins += (diag_sum_tr == game.board_size)

        if player_one_wins:
            game._result = None
            return 1

        player_two_wins = any(rowsum == -game.board_size)
        player_two_wins += any(colsum == -game.board_size)
        player_two_wins += (diag_sum_tl == -game.board_size)
        player_two_wins += (diag_sum_tr == -game.board_size)

        if player_two_wins:
            game._result = None
            return -1

        if np.all(game.board != 0):
            game._result = None
            return 0

        return None