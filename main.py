from tictactoe import *
from mcts_node import *
from mcts_search import *

empty_board = np.zeros((3, 3))
board_s1 = np.array([
        [1,1,-1],
        [1,-1,-1],
        [0,-1,0]])
initial_board_state = TicTacToeGameState(state=board_s1, next_to_move=1)

#This root is very important. I'm gonna use it to get the whole game tree
root = MonteCarloTreeSearchNode(state=initial_board_state, parent=None)
#NOTE Now that I have the root I can keep track of the tree? 
mcts = MCTS(root)

a = mcts.best_action(10000)

#This should output the board and the probability distrubtion of each of the squares
"""
For every state of the game tree run MCTS 10000s.
This should develop the game tree? I need to be collapsing 

Every winning move is a pure strategy so this will work for those. 
One downside is that if two moves both win at the same time, the algorithm will split it's weight equally
NOTE I think this might be fixed from using full exploitation from like tempurate and PUCT 
Also I think I should just check if I'm in a losing/winning square and just pick it to be the new node
since it really doesn't matter what node if my goal is to win.
"""
print(root.state)
print([c.state for c in root.children])
