from tictactoe import *
from mcts_node import *
from mcts_search import *


reset_state = np.zeros((3, 3))
state = np.array([
        [1,1,-1],
        [1,-1,-1],
        [0,-1,0]])
initial_board_state = TicTacToeGameState(state=state, next_to_move=1)

root = MonteCarloTreeSearchNode(state=initial_board_state, parent=None)
mcts = MCTS(root)

a = mcts.best_action(10000)

#This should output the board and the probability distrubtion of each of the squares
print(a)
