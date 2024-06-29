import cProfile
import pstats
import io
import numpy as np
from graphviz import Digraph
from pprint import pprint
from tictactoe import TicTacToe, test_tic_tac_toe
from test_mcts_tictactoe import test_mcts_picks_winning_move_when_almost_won
from mcts import MCTS

profiler = cProfile.Profile()
profiler.enable()
empty_board = np.zeros((3, 3))
new_game = TicTacToe.get_state(state=empty_board)
mcts = MCTS(game_state=new_game)
mcts.search(1_000_000)
profiler.disable()

# Create a string stream to store the profiling results
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
ps.strip_dirs().sort_stats(sortby).print_stats()

print(s.getvalue())
pprint([c.results for c in mcts.root.children])
    
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
