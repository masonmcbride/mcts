import numpy as np
from tictactoe import TicTacToeState
from mcts import MCTS

def test_mcts_picks_winning_move_when_almost_won():
    one_move_to_win = np.array([
        [1,-1,0],
        [1,1,-1],
        [-1,0,0]])
    almost_won = TicTacToeState(state=one_move_to_win)
    mcts = MCTS(game_state=almost_won)
    mcts.search(8) # With UCB, it figures it out in one search but with PUCT you need 10
    winning_move = max(mcts.root.children, key=lambda child: child.Q)
    assert winning_move.game_state.state[2,2] == 1

def test_mcts_blocks_win():
    board = np.array([
        [1,-1,-1],
        [0,-1,1],
        [0,0,1]])
    O_can_win = TicTacToeState(state=board)
    blocked = np.array([
        [1,-1,-1],
        [0,-1,1],
        [1,0,1]])
    mcts = MCTS(game_state=O_can_win)
    mcts.search(8) # With UCB, it figures it out in one search but with PUCT you need 10
    chosen_move = max(mcts.root.children, key=lambda child: child.N)
    assert np.array_equal(chosen_move.game_state.state, blocked)

def test_one_run_expands_and_selects_one():
    empty_board = np.zeros((3, 3))
    new_game = TicTacToeState(state=empty_board)
    mcts = MCTS(game_state=new_game)
    mcts.search(1)
    assert mcts.root.N == 10