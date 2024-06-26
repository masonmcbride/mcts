from typing import Any
import random
import numpy as np

GameState = Any
class MCTSNode:
    def __init__(node, game_state: GameState, parents: list['MCTSNode'], prior: float):
        node.game_state: GameState = game_state # game state that this node is derived from
        node.is_terminal: bool = game_state.is_terminal
        node.is_expanded: bool = False
        node.parents: list['MCTSNode'] = parents # parent MCTSNode
        node.children: list['MCTSNode'] = []
        node.N: int = 0                  # number of visits
        node.W: int = 0                  # sum of all rewards 
        node.P: float = prior            # transition probablity of reaching this state
    
    @property
    def Q(self) -> float:
        return self.W / self.N #average value of node

class MCTS:
    def __init__(tree, game_state: GameState):
        tree.root: MCTSNode = MCTSNode(game_state,parents=[],prior=1) # parent is none and 100% transition prob
        tree.states = {game_state: tree.root}

    def best_child(tree, node: MCTSNode) -> MCTSNode:
        return max([child for child in node.children], key=tree.PUCT)

    def select(tree) -> MCTSNode:
        """MCTS Selection.
        The tree selects the most promising node until it reaches an unexpanded node"""
        selected_node = tree.root # Start at the root
        path = [selected_node]
        while selected_node.is_expanded and not selected_node.is_terminal:
            selected_node = tree.best_child(selected_node)
            path.append(selected_node)
        return selected_node, path

    def expand(tree, node: MCTSNode, path: list[MCTSNode]):
        """MCTS Expansion.
        The unexpanded node is expanded and the most promising child is returned to be rolled out."""
        if node.is_terminal:
            return node
        else:
            game = node.game_state
            for action in game.all_legal_actions:
                if (new_state := game.transition(action)) in tree.states:
                    node.children.append(tree.states[new_state])
                    tree.states[new_state].parents.append(node)
                else:
                    child = MCTSNode(game_state=new_state,parents=[node],prior=1)
                    node.children.append(child)
                    reward = tree.rollout(child)
                    tree.backprop(path + [child], reward)
                    tree.states[new_state] = child
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
        return reward * -game.player

    def backprop(tree, path: list[MCTSNode], value: int) -> None:
        """MCTS Backropagation. 
        The reward from this terminal state must be propagated up stream to update MCTS behavior"""
        for node in reversed(path):
            node.W += value
            node.N += 1
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
        node, path = tree.select() #SELECTION 
        node = tree.expand(node, path) # EXPANSION
        path.append(node)
        reward = tree.rollout(node) # ROLLOUT
        tree.backprop(path, reward) # BACKPROP

    def search(self, n):
        """Perform n MCTS runs from the root. This is equivalent to MCTS 'thinking' or 'searching'
        the space for a good action."""
        for i in range(n):
            if i % 250 == 0:
                print(f"run {i}")
            self.run()