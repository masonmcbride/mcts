from typing import Any
import random
import numpy as np

GameState = Any
class MCTSNode:
    def __init__(node, game_state: GameState, parent: Any | None, prior: float):
        node.game_state: GameState = game_state # game state that this node is derived from
        node.is_terminal: bool = game_state.is_terminal
        node.is_expanded: bool = False
        node.parent: 'MCTSNode' = parent # parent MCTSNode
        node.P: float = prior            # transition probablity of reaching this state
        node.N: int = 0                  # number of visits
        node.W: int = 0                  # sum of all rewards 
        node.children: list['MCTSNode'] = []
        print("node initialized")
    
    @property
    def Q(self) -> float:
        return self.W / self.N #average value of node

class MCTS:
    def __init__(tree, game_state: GameState):
        tree.root: MCTSNode = MCTSNode(game_state,parent=None,prior=1) # parent is none and 100% transition prob

    def best_child(tree, node: MCTSNode) -> MCTSNode:
        return max([child for child in node.children], key=tree.PUCT)

    def select(tree) -> MCTSNode:
        """MCTS Selection.
        The tree selects the most promising node until it reaches an unexpanded node"""
        selected_node = tree.root # Start at the root
        while selected_node.is_expanded and not selected_node.is_terminal:
            selected_node = tree.best_child(selected_node)
        return selected_node

    def expand(tree, node: MCTSNode):
        """MCTS Expansion.
        The unexpanded node is expanded and the most promising child is returned to be rolled out."""
        if node.is_terminal:
            return node
        else:
            game = node.game_state
            node.children = [MCTSNode(game.transition(action),parent=node,prior=1)
                             for action in game.all_legal_actions]
            for child in node.children:
                reward = tree.rollout(child)
                tree.backprop(child, reward)
            
            node.is_expanded = True
            return tree.best_child(node)

    def rollout(tree, node: MCTSNode) -> int:
        """MCTS Rollout.
        Detach game from given MCTSNode, Simluate until end state and return reward"""
        game = node.game_state
        while not game.is_terminal:
            action = random.choice(game.all_legal_actions)
            game = game.transition(action)
        reward = game.result
        return reward

    def backprop(tree, node: MCTSNode, value: int) -> None:
        """MCTS Backropagation. 
        The reward from this terminal state must be propagated up stream to update MCTS behavior"""
        node.W += value
        node.N += 1
        if node.parent:
            tree.backprop(node.parent, -value)
    
    def UCB(tree, node: MCTSNode, c_param=2) -> float:
        """Returns (U)pper (C)onfidence (B)ound score for MCTSNode"""
        return node.Q + c_param * np.sqrt(np.log(node.parent.N)/node.N)

    def PUCT(tree, node: MCTSNode, c_param=1.41) -> float:
        """Returns (P)olynomial (U)pper (C)onfidence score for (T)rees for MCTSNode"""
        U = c_param * node.P * np.sqrt(tree.root.N) / (1 + node.N)
        return node.Q + U 

    def run(tree):
        """Perform one (1) entire sequence of MCTS. 
        In my version (for small branching factors), all child nodes are expanded at the same time.
        This is just better (unless you have 1 million actions, but why are you using MCTS), 
        but prove me wrong."""
        node = tree.select() #SELECTION 
        node = tree.expand(node) # EXPANSION
        reward = tree.rollout(node) # ROLLOUT
        tree.backprop(node, reward) # BACKPROP

    def search(self, n):
        """Perform n MCTS runs from the root. This is equivalent to MCTS 'thinking' or 'searching'
        the space for a good action."""
        for _ in range(n):
            self.run()