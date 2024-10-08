import cProfile
import pstats
import io
import numpy as np
from visualize import visualize_tree
from tictactoe import TicTacToe
from connect4 import Connect4
from mcgs import MCGS

profiler = cProfile.Profile()

cant_lose = np.array([
    [ 0, -1, 0, -1,  1, -1,  0],
    [-1,  1, 0,  1, -1,  1, -1],
    [-1,  1, 1,  1, -1,  -1, 1],
    [ 1, -1,  1, -1,  1, -1,  1],
    [ 1, -1,  1, -1,  1, -1,  1],
    [-1,  1, -1,  1, -1,  1, -1]
    ])
board = np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., -1., 0., 0., 0.],
        [0., 0., -1., 1., 0., 0., 0.],
        [0., 0., -1, 1., 1., 0., 0.],
        [0., 0., -1, 1, -1, 0., 0.]])
board = np.array([
        [1,-1,0],
        [1,1,-1],
        [-1,0,0]])
win_or_draw = TicTacToe.get_state(state=board)
mcts = MCGS(game_state=win_or_draw)

profiler.enable()

for _ in range(100):
    mcts.run()

profiler.disable()

# Create a string stream to store the profiling results
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
ps.strip_dirs().sort_stats(sortby).print_stats()
print(s.getvalue())
print(mcts.root.results)
for child in mcts.root.child_to_edge_visits.keys():
#for child in mcts.root.children:
    print(f"move player={child.game_state.player}")
    print(child.game_state)
    print(f"results {child.Q=}")
    print(child.results)

graph = visualize_tree(mcts.root)
graph.render('game_tree', format='png', cleanup=True)
