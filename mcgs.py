from typing import Any
import random
import numpy as np

GameState = Any

class MCGSNode:
    def __init__(self, game_state: GameState):
        self.game_state: GameState = game_state  
        self.is_terminal: bool = game_state.is_terminal
        self.is_expanded: bool = False
        self.N: int = 0  # Visit count
        self.Q: float = 0.0  # Regularized value
        self.child_to_edge_visits: dict[MCGSNode, int] = {}  # child node -> edge visits
        self.results = {1: 0, -1: 0, 0: 0}

class MCGS:
    def __init__(tree, game_state: GameState):
        tree.root: MCGSNode = MCGSNode(game_state)
        tree.nodes: dict[Any, MCGSNode] = {game_state: tree.root}

    def get_node(tree, game_state: GameState) -> MCGSNode:
        """Retrieve or create a new MCGSNode for the given game state."""
        if game_state not in tree.nodes:
            tree.nodes[game_state] = MCGSNode(game_state=game_state)
        return tree.nodes[game_state]

    def best_child(tree, node: MCGSNode) -> MCGSNode:
            return max([child for child in node.child_to_edge_visits.keys()], key=lambda x: tree.PUCT(node,x))

    def select(tree) -> list[MCGSNode]:
        """MCGS Selection.
        The tree selects the most promising node until it reaches an unexpanded node"""
        path = [tree.root]
        while path[-1].is_expanded and not path[-1].is_terminal:
            next_node = tree.best_child(path[-1])
            path[-1].child_to_edge_visits[next_node] += 1
            path.append(next_node)
        return path

    def expand(tree, path: list[MCGSNode]) -> list[MCGSNode]:
        """MCGS Expansion.
        The unexpanded node is expanded and the most promising child is returned to be rolled out."""
        expanding_node = path[-1]
        if expanding_node.is_terminal:
            return path
        else:
            game = expanding_node.game_state
            for action in game.all_legal_actions:
                child_node = tree.get_node(game.transition(action))
                expanding_node.child_to_edge_visits[child_node] = 1
                if child_node.N == 0:
                    reward = tree.rollout(child_node)
                    tree.backprop(path + [child_node], reward)
            expanding_node.is_expanded = True
            return path + [tree.best_child(expanding_node)]

    def rollout(tree, node: MCGSNode) -> dict:
        """Estimate the utility of a non-terminal game state.
        Detach game from given MCGSNode, Simluate until end state and return reward"""
        game = node.game_state
        while not game.is_terminal:
            action = random.choice(game.all_legal_actions)
            game = game.transition(action)
        reward = game.result
        return reward

    def backprop(tree, path: list[MCGSNode], reward: int):
        """Update all nodes on the search path with the reward signal recieved from rollout"""
        reward = reward[path[-1].game_state.player]
        for node in reversed(path):
            node.N = 1 + sum(node.child_to_edge_visits.values())
            node.Q = -(1/node.N)*(reward + sum(child.Q * edge_visits for (child, edge_visits) in node.child_to_edge_visits.items()))
            node.results[reward] += 1
            reward = -reward
        
    def run(tree):
        """Perform one playout from the given node."""
        path = tree.select()
        path = tree.expand(path)
        reward = tree.rollout(path[-1])
        tree.backprop(path, reward)
    
    def PUCT(tree, parent, node, c_puct=1.):
        #P_sa = len(parent.child_to_edge_visits)
        N_sa = parent.child_to_edge_visits[node]
        return node.Q + c_puct * 1 * np.sqrt(parent.N) / (1 + N_sa)

    def search(tree, n: int):
        """Perform n playouts from the root node."""
        for _ in range(n):
            tree.run()
