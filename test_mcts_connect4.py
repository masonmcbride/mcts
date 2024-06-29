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
    mcts.search(50)
    answer = np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., -1, 1., 0., 0., 0.],
        [0., 0., -1, 1, -1, 0., 0.]])
    winning_move = max(mcts.root.children, key=lambda child: child.Q)
    assert np.array_equal(winning_move.game_state.state, answer) 

def test_mcts_results_contain_no_losses():
    cant_lose = np.array([
    [ 0, -1, 0, -1,  1, -1,  0],
    [-1,  1, 0,  1, -1,  1, -1],
    [-1,  1, 1,  1, -1,  -1, 1],
    [ 1, -1,  1, -1,  1, -1,  1],
    [ 1, -1,  1, -1,  1, -1,  1],
    [-1,  1, -1,  1, -1,  1, -1]
    ])
    win_or_draw = Connect4State(state=cant_lose)
    mcts = MCTS(game_state=win_or_draw)
    mcts.search(100)
    assert mcts.root.results[-1] == 0

def test_mcts_blocks_win():
    board = np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., -1., 0., 0., 0.],
        [0., 0., -1., 1., 0., 0., 0.],
        [0., 0., -1, 1., 1., 0., 0.],
        [0., 0., -1, 1, -1, 0., 0.]])
    O_can_win = Connect4State(state=board)
    blocked = np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 1., -1., 0., 0., 0.],
        [0., 0., -1., 1., 0., 0., 0.],
        [0., 0., -1, 1., 1., 0., 0.],
        [0., 0., -1, 1, -1, 0., 0.]])
    mcts = MCTS(game_state=O_can_win)
    mcts.search(50) 
    chosen_move = max(mcts.root.children, key=lambda child: child.Q)
    assert np.array_equal(chosen_move.game_state.state, blocked)

def test_one_run_expands_and_selects_one():
    empty_board = np.zeros((6, 7))
    new_game = Connect4State(state=empty_board)
    mcts = MCTS(game_state=new_game)
    mcts.search(1)
    assert mcts.root.N == 8