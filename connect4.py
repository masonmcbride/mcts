import numpy as np
from itertools import product

class Connect4State:
    def __init__(game, state: np.ndarray):
        game.state: np.ndarray = state
        game.player = 1 if np.sum(state) <= 0 else -1
        game.result = game.game_result()
        game.is_terminal = True if game.result is not None else False
        game.all_legal_actions: list[int] = [action for 
                                               action in range(state.shape[1]) 
                                               if np.any(state[:,action]==0)] if not game.is_terminal else []
    
    def transition(game, action: int) -> 'Connect4State':
        new_state = game.state.copy()
        for i,row_entry in enumerate(reversed(new_state[:,action])):
            if row_entry == 0:
                new_state[game.state.shape[0]-i-1,action] = game.player
                break
        return Connect4.get_state(new_state)
    
    def game_result(game) -> int | None:
        def check_4by4(view: np.ndarray):
            horiz_sums: set[int] = set(view.sum(axis=1).tolist())
            vert_sums: set[int]  = set(view.sum(axis=0).tolist())
            diag_sums: set[int]  = {view.trace(), view[::-1].trace()}
            all_sums: set[int] = horiz_sums | vert_sums | diag_sums
            
            if 4 in all_sums:      return 1 # Player 1 wins
            if -4 in all_sums:     return -1 # Player -1 wins
            if np.all(game.state): return 0 # Tie 
            else: return None                   # No result
        
        return next((result for result in map(check_4by4,(game.state[i:i+4,j:j+4] 
            for (i, j) in product(range(game.state.shape[0]-3), range(game.state.shape[1]-3))))
            if result is not None), None)

    def __str__(game):
        return str(game.state)
    
    def __hash__(game):
        return hash(tuple(game.state.flat))

    def __eq__(game, other):
        if isinstance(other, Connect4State):
            return np.array_equal(game.state, other.state)
        return False

class Connect4:
    """Class to manage TicTacToe states and cache."""
    game_states: dict[np.ndarray, Connect4State]= {}

    @classmethod
    def get_state(cls, state: np.ndarray) -> Connect4State:
        state_tuple = tuple(state.flat)
        if state_tuple not in cls.game_states:
            cls.game_states[state_tuple] = Connect4State(state)
        return cls.game_states[state_tuple]

    @classmethod
    def reset_cache(cls):
        cls.game_states = {}