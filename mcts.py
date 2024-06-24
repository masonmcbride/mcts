from typing import Object, Any
import random
import numpy as np

GameState = Object
class MCTSNode:
    def __init__(node, game_state: GameState, parent: Any | None, prior: float):
        node.game_state = game_state # game state that this node is derived from
        node.is_terminal: bool = game_state.is_terminal()
        node.parent = parent # parent MCTSNode
        node.P: float = prior # transition probablity of reaching this state
        node.N: int = 1 # number of visits
        node.W: int = 0 # sum of all rewards 
        node.children: list = []
    
    @property
    def Q(self) -> float:
        return self.W / self.N #average value of node

class MCTS:
    def __init__(tree, game_state: GameState):
        tree.root: MCTSNode = MCTSNode(game_state,parent=None,prior=1) # parent is none and 100% transition prob

    def select(tree):
        """iterative traversal of tree"""
        current_node = tree.root
        while not current_node.is_terminal:
            if current_node.N == 1: # if the node has just been reached
                

            else:
                current_node = tree.best_child(current_node)
        return current_node
    

    def best_child(tree, node: MCTSNode):
        return max([child for child in node.children.values()], key=tree.PUCT)


    def expand(self, node, p):
        action = node.untried_actions.pop()
        next_state = node.state.make_action(action)
        child_node = Node(next_state, parent=node, prior=p[0][action])
        node.children[action] = child_node
        return child_node

    def rollout(tree, node: MCTSNode) -> int:
        """Detach game from MCTS Node, Simluate until end state and return reward"""
        game = node.game_state
        while not (result:=game.is_terminal()):
            action = random.choice(game.all_legal_actions())
            game = game.transition(action)
        reward = result
        return reward

    def backprop(tree, node: MCTSNode, value: int) -> None:
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
        node = tree.select() #SELECTION + EXPANSION
        reward = node.rollout() # ROLLOUT
        tree.backprop(node, reward) # BACKPROP

    def search(self, n):
        for _ in range(n):
            self.run()
