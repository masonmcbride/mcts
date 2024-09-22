import numpy as np
from tictactoe import TicTacToe
from mcgs import MCGS

def test_mcts_picks_winning_move_when_almost_won():
    one_move_to_win = np.array([
        [1,-1,0],
        [1,1,-1],
        [-1,0,0]])
    almost_won = TicTacToe.get_state(state=one_move_to_win)
    mcts = MCGS(game_state=almost_won)
    mcts.search(50) 
    winning_move = max([child for child 
                        in mcts.root.child_to_edge_visits.keys()],
                        key=lambda child: child.Q)
    assert winning_move.game_state.state[2,2] == 1

def test_mcts_results_contain_no_losses():
    one_move_to_win = np.array([
        [1,-1,0],
        [1,1,-1],
        [-1,0,0]])
    almost_won = TicTacToe.get_state(state=one_move_to_win)
    mcts = MCGS(game_state=almost_won)
    mcts.search(50)
    assert mcts.root.results[-1] == 0

def test_mcts_blocks_win():
    board = np.array([
        [-1,1,0],
        [1,-1,0],
        [0,0,0]])
    O_can_win = TicTacToe.get_state(state=board)
    blocked = np.array([
        [-1,1,0],
        [1,-1,0],
        [0,0,1]])
    mcts = MCGS(game_state=O_can_win)
    mcts.search(50) 
    chosen_move = max([child for child 
                        in mcts.root.child_to_edge_visits.keys()],
                        key=lambda child: child.Q)
    assert np.array_equal(chosen_move.game_state.state, blocked)

def test_one_run_expands_and_selects_one():
    empty_board = np.zeros((3, 3))
    new_game = TicTacToe.get_state(state=empty_board)
    mcts = MCGS(game_state=new_game)
    mcts.search(1)
    assert mcts.root.N == 10