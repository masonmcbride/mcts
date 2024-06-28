import cProfile
import pstats
import io
import numpy as np
from graphviz import Digraph
from tictactoe import TicTacToe, test_tic_tac_toe
from mcts import MCTS

profiler = cProfile.Profile()
profiler.enable()

board = np.array([
        [1,-1,-1],
        [0,-1,1],
        [0,0,1]])
O_can_win = TicTacToe.get_state(state=board)
mcts = MCTS(game_state=O_can_win)
mcts.search(1000) # With UCB, it figures it out in one search but with PUCT you need 10
for child in mcts.root.children:
    print(child.Q)
chosen_move = max(mcts.root.children, key=lambda child: child.Q)

profiler.disable()

# Create a string stream to store the profiling results
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
ps.strip_dirs().sort_stats(sortby).print_stats()

#print(s.getvalue())
    
def format_board(board: np.ndarray):
    """Convert numpy array to a string format suitable for graph labels."""
    player = 1 if np.sum(board) <= 0 else -1
    return f'player {player}\n' + '\n'.join(' '.join(str(int(cell)) for cell in row) for row in board)

def add_nodes_edges(graph, node, seen):
    node_label = format_board(node.game_state.state)
    if node in seen:
        return
    seen.add(node)
    graph.node(name=str(id(node)), label=node_label, shape='circle')
    for child in node.children:
        graph.edge(str(id(node)), str(id(child)))
        add_nodes_edges(graph, child, seen)

def visualize_tree(root):
    graph = Digraph(comment='game_tree', format='png')
    seen = set()
    add_nodes_edges(graph, root, seen)
    return graph

graph = visualize_tree(mcts.root)
graph.render('game_tree', format='png', cleanup=True)
