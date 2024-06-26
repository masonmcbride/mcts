import numpy as np
from connect4 import Connect4State
from mcts import MCTS

def test_mcts_chooses_winning_move():
    board = np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., -1, 1., 0., 0., 0.],
        [0., 0., -1, 1, -1, 0., 0.]])
    one_move_to_win = Connect4State(state=board)
    mcts = MCTS(game_state=one_move_to_win)
    mcts.search(10)
    answer = np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., -1, 1., 0., 0., 0.],
        [0., 0., -1, 1, -1, 0., 0.]])
    winning_move = max(mcts.root.children, key=lambda child: child.Q)
    assert np.array_equal(winning_move.game_state.state, answer) 