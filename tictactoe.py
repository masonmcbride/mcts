import numpy as np
from itertools import product

class TicTacToeState():
    """A tictactoe state is the state of a tic tac toe board. Tictactoe is perfect information."""
    __slots__ = ('state','player','all_legal_actions','result','is_terminal')
    def __init__(game, state: np.ndarray):
        game.state: np.ndarray = state
        game.player = 1 if np.sum(state) <= 0 else -1
        game.all_legal_actions: list[tuple] = [action for 
                                               action in product(*[range(dim) for dim in state.shape])
                                               if state[action] == 0]
        game.result = game.game_result()
        game.is_terminal = True if game.result is not None else False
    
    def transition(game, action: tuple) -> 'TicTacToeState':
        new_state = game.state.copy()
        new_state[action] = game.player
        return TicTacToeState(new_state)
    
    def game_result(game) -> int | None:
        """ this property should return 
        1 if player #1 wins 
        -1 if player #2 wins 
        0 if there is a draw 
        None if result is unknown
        """
        board = game.state
        three_in_a_row = 3
        rowsum = np.sum(board, 0)
        colsum = np.sum(board, 1)
        diag_sum_tl = board.trace()
        diag_sum_tr = board[::-1].trace()

        player_one_wins = any(rowsum == three_in_a_row)
        player_one_wins += any(colsum == three_in_a_row)
        player_one_wins += (diag_sum_tl == three_in_a_row)
        player_one_wins += (diag_sum_tr == three_in_a_row)

        if player_one_wins:
            return 1

        player_two_wins = any(rowsum == -three_in_a_row)
        player_two_wins += any(colsum == -three_in_a_row)
        player_two_wins += (diag_sum_tl == -three_in_a_row)
        player_two_wins += (diag_sum_tr == -three_in_a_row)

        if player_two_wins:
            return -1

        if len(game.all_legal_actions) == 0:
            return -1

        return None
    
    def __str__(game):
        return str(game.state)