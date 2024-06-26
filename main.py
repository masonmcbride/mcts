import numpy as np
from graphviz import Digraph
from tictactoe import TicTacToeState
from mcts import MCTS, MCTSNode

one_move_to_win = np.array([
        [1,-1,0],
        [1,1,-1],
        [-1,0,0]])
almost_won = TicTacToeState(state=one_move_to_win)
mcts = MCTS(game_state=almost_won)
mcts.search(8) # With UCB, it figures it out in one search but with PUCT you need 10
    
def format_board(board: np.ndarray):
    """Convert numpy array to a string format suitable for graph labels."""
    return '\n'.join(' '.join(str(int(cell)) for cell in row) for row in board)

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