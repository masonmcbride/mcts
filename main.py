import numpy as np
from tictactoe import TicTacToeState
from mcts import MCTS

empty_board = np.zeros((3, 3))
new_game = TicTacToeState(state=empty_board)
move1: TicTacToeState = new_game.transition((0,0))
print(move1)
move2: TicTacToeState = move1.transition((1,1))
print(move2)

board_s1 = np.array([
        [1,0,0],
        [0,1,-1],
        [-1,0,1]])
#This root is very important. I'm gonna use it to get the whole game tree
mcts = MCTS(game_state=new_game)
mcts.search(1)
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

"""
If it wins instantly always do it, if it loses instantly never do it.

Since the goal is always to win, there exists some pure strategies in states where you are 
one move away from winning. Once the search reaches a winning node, the search should skip
the ucb exploration/exploitation a play that winning move as a pure strategy. Not only is this
correct on it's own, but it enables the search to prune the pure strategy branch into one node 
so when the losing player searches and has their turn on move on a already lost game, that 
counts as a loss. What just happened is that the search has abstracted that action to counting as 
a direct loss. 

The way I need to accomplish this is I need to make the nodes aware in the search whether they
are in a terminal state and then whether that state is losing or winning. Once I have this awareness,
the search just needs to check the nodes as losing and then if the state is pure or winning for the player,
then it can do the magic.

NODES AWARE

There must exist some method or algorithm that over lim n -> inf iterations 
it can correctly converge to the strategy 
for example, this algorithm must be able to converge to a pure strategy over lim n -> inf iterations
"""
