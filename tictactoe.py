import numpy as np
from itertools import product

class TicTacToeState:
    """A tictactoe state is the state of a tic tac toe board. Tictactoe is perfect information."""
    __slots__ = ('state','player','all_legal_actions','result','is_terminal')
    
    def __init__(game, state: np.ndarray):
        game.state: np.ndarray = state
        game.player = 1 if np.sum(state) <= 0 else -1
        game.result = game.game_result()
        game.is_terminal = True if game.result is not None else False
        game.all_legal_actions: list[tuple] = [action for 
                                                action in product(*[range(dim) for dim in state.shape])
                                                if state[action] == 0] if not game.is_terminal else []
    
    def transition(game, action: tuple) -> 'TicTacToeState':
        new_state = game.state.copy()
        new_state[action] = game.player
        return TicTacToe.get_state(new_state)
    
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
        player_one_wins |= any(colsum == three_in_a_row)
        player_one_wins |= (diag_sum_tl == three_in_a_row)
        player_one_wins |= (diag_sum_tr == three_in_a_row)

        if player_one_wins:
            return {1:1, -1:-1}

        player_two_wins = any(rowsum == -three_in_a_row)
        player_two_wins |= any(colsum == -three_in_a_row)
        player_two_wins |= (diag_sum_tl == -three_in_a_row)
        player_two_wins |= (diag_sum_tr == -three_in_a_row)

        if player_two_wins:
            return {1:-1, -1:1}

        if np.all(board != 0):
            return {1:0, -1:0}

        return None
    
    def __str__(game):
        return str(game.state)
    
    def __hash__(game):
        return hash(tuple(game.state.flat)) 

    def __eq__(game, other):
        if isinstance(other, TicTacToeState):
            return np.array_equal(game.state, other.state)
        return False

class TicTacToe:
    """Class to manage TicTacToe states and cache."""
    game_states: dict[np.ndarray, TicTacToeState]= {}

    @classmethod
    def get_state(cls, state: np.ndarray) -> TicTacToeState:
        state_tuple = tuple(state.flat)
        if state_tuple not in cls.game_states:
            cls.game_states[state_tuple] = TicTacToeState(state)
        return cls.game_states[state_tuple]

    @classmethod
    def reset_cache(cls):
        cls.game_states = {}

def test_tic_tac_toe():
    def all_states(state: TicTacToeState):
        """Generate all states of tictactoe. The number of valid states is 5478. """
        yield state   
        for action in state.all_legal_actions:
            new_state = state.transition(action)
            yield from all_states(new_state)
    empty_board = np.zeros((3, 3))
    new_game = TicTacToe.get_state(empty_board)
    assert len(set(all_states(new_game))) == 5478

