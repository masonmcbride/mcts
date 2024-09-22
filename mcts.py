from typing import Any
import random
import numpy as np

GameState = Any

class MCTSNode:
    def __init__(node, game_state: GameState):
        node.game_state: GameState = game_state # game state that this node is derived from

        node.is_terminal: bool = game_state.is_terminal
        node.is_expanded: bool = False
        node.children: list['MCTSNode'] = []
        node.N: int = 0                  # number of visits
        node.W: int = 0                  # sum of all rewards 
        node.P: float = 1.               # transition probablity of reaching this state
        node.results = {1: 0, -1: 0, 0: 0}
    
    @property
    def Q(self) -> float:
        return self.W / self.N #average value of node

class MCTS:
    def __init__(tree, game_state: GameState):
        tree.root: MCTSNode = MCTSNode(game_state)
        tree.nodes: dict[GameState, MCTSNode] = {game_state: tree.root}
    
    def get_node(tree, game_state: GameState) -> MCTSNode:
        """Retrieve or create a new MCTSNode for the given game state."""
        if game_state not in tree.nodes:
            tree.nodes[game_state] = MCTSNode(game_state=game_state)
        return tree.nodes[game_state]

    def best_child(tree, node: MCTSNode) -> MCTSNode:
        return max([child for child in node.children], key=tree.PUCT)

    def select(tree) -> list[MCTSNode]:
        """MCTS Selection.
        The tree selects the most promising node until it reaches an unexpanded node"""
        node = tree.root 
        path = [node]
        while node.is_expanded and not node.is_terminal:
            node = tree.best_child(node)
            path.append(node)
        return path

    def expand(tree, path: list[MCTSNode]) -> list[MCTSNode]:
        """MCTS Expansion.
        The unexpanded node is expanded and the most promising child is returned to be rolled out."""
        expanding_node = path[-1]
        if expanding_node.is_terminal:
            return path
        else:
            game = expanding_node.game_state
            for action in game.all_legal_actions:
                child_node = tree.get_node(game.transition(action))
                if child_node.N == 0:
                    reward = tree.rollout(child_node)
                    tree.backprop(path + [child_node], reward)
                expanding_node.children.append(child_node)
            expanding_node.is_expanded = True
            return path + [tree.best_child(expanding_node)]

    def rollout(tree, node: MCTSNode) -> int:
        """MCTS Rollout.
        Detach game from given MCTSNode, Simluate until end state and return reward"""
        game = node.game_state
        while not game.is_terminal:
            action = random.choice(game.all_legal_actions)
            game = game.transition(action)
        reward = game.result
        return reward

    def backprop(tree, path: list[MCTSNode], reward: dict) -> None:
        """MCTS Backropagation. 
        The reward from this terminal state must be propagated up stream to update MCTS behavior"""
        value = reward[path[-1].game_state.player]
        for node in reversed(path):
            node.W -= value 
            node.N += 1
            node.results[value] += 1
            value = -value
    
    def UCB(tree, node: MCTSNode, c_param=2) -> float:
        """Returns (U)pper (C)onfidence (B)ound score for MCTSNode"""
        return node.Q + c_param * np.sqrt(np.log(node.parent.N)/node.N)

    def PUCT(tree, node: MCTSNode, c_param=3) -> float:
        """Returns (P)olynomial (U)pper (C)onfidence score for (T)rees for MCTSNode"""
        U = c_param * node.P * np.sqrt(tree.root.N) / (1 + node.N)
        return node.Q + U 

    def run(tree):
        """Perform one (1) entire sequence of MCTS. 
        In my version (for small branching factors), all child nodes are expanded at the same time.
        This is just better (unless you have 1 million actions, but why are you using MCTS), 
        but prove me wrong."""
        path = tree.select() #SELECTION 
        path = tree.expand(path) # EXPANSION
        reward = tree.rollout(path[-1]) # ROLLOUT
        tree.backprop(path, reward) # BACKPROP

    def search(tree, n):
        """Perform n MCTS runs from the root. This is equivalent to MCTS 'thinking' or 'searching'
        the space for a good action."""
        for i in range(n):
            if i % 1000 == 0:
                print(f"run {i}")
            tree.run()