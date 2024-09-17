from typing import Any
import random
import numpy as np

GameState = Any

class MCTSNode:
    def __init__(self, game_state: GameState):
        self.game_state: GameState = game_state  # Game state associated with this node
        self.is_terminal: bool = game_state.is_terminal
        self.N: int = 0  # Visit count
        self.Q: float = 0.0  # Average value
        self.U: float = 0.0  # Utility value (from neural net or rollout)
        self.children_and_edge_visits: dict[Any, tuple['MCTSNode', int]] = {}  # action -> (child node, edge visits)
        self.results = {1: 0, -1: 0, 0: 0}

class MCTS:
    def __init__(tree, game_state: GameState):
        tree.root: MCTSNode = MCTSNode(game_state)
        tree.nodes_by_hash: dict[Any, MCTSNode] = {tree.hash_game_state(game_state): tree.root}

    def hash_game_state(self, game_state: GameState) -> Any:
        """Generate a unique hash for the game state."""
        return hash(game_state)  # Assumes GameState has a 'hash' property

    def get_node(self, game_state: GameState) -> MCTSNode:
        """Retrieve or create a new MCTSNode for the given game state."""
        state_hash = self.hash_game_state(game_state)
        if state_hash not in self.nodes_by_hash:
            self.nodes_by_hash[state_hash] = MCTSNode(game_state=game_state)
        return self.nodes_by_hash[state_hash]

    def rollout(self, node: MCTSNode) -> float:
        """Estimate the utility of a non-terminal game state.
        Detach game from given MCTSNode, Simluate until end state and return reward"""
        game = node.game_state
        while not game.is_terminal:
            action = random.choice(game.all_legal_actions)
            game = game.transition(action)
        reward = game.result
        return reward

    def select_action_according_to_puct(self, node: MCTSNode, c_puct=1.0) -> Any:
        """Select an action based on the PUCT algorithm."""
        total_N = node.N
        actions = node.game_state.all_legal_actions

        best_action = None
        best_puct = -float('inf')

        for action in actions:
            if action in node.children_and_edge_visits:
                child, edge_visits = node.children_and_edge_visits[action]
                Q_sa = child.Q
                N_sa = edge_visits
            else:
                Q_sa = 0.0  # Unvisited child nodes have Q=0
                N_sa = 0

            P_sa = 1 / len(actions)  # Uniform prior probability

            puct = Q_sa + c_puct * P_sa * np.sqrt(total_N) / (1 + N_sa)
            if puct > best_puct:
                best_puct = puct
                best_action = action

        return best_action

    def run(self, node: MCTSNode = None):
        """Perform one playout from the given node."""
        if node is None:
            node = self.root

        if node.is_terminal:
            node.U = node.game_state.result
            node.results[node.U] += 1
        elif node.N == 0:  # New node not yet visited
            node.U = self.get_utility_from_neural_net(node)
            node.results[node.U] += 1
        else:
            action = self.select_action_according_to_puct(node)
            if action not in node.children_and_edge_visits:
                new_game_state = node.game_state.transition(action)
                new_game_state_hash = self.hash_game_state(new_game_state)
                if new_game_state_hash in self.nodes_by_hash:
                    child = self.nodes_by_hash[new_game_state_hash]
                    node.children_and_edge_visits[action] = (child, 0)
                else:
                    new_node = MCTSNode(game_state=new_game_state)
                    node.children_and_edge_visits[action] = (new_node, 0)
                    self.nodes_by_hash[new_game_state_hash] = new_node
                child, edge_visits = node.children_and_edge_visits[action]
            else:
                child, edge_visits = node.children_and_edge_visits[action]

            self.run(child)
            child, edge_visits = node.children_and_edge_visits[action]
            node.children_and_edge_visits[action] = (child, edge_visits + 1)

            # Aggregate results from the child node
            for result_key in node.results:
                node.results[result_key] += child.results.get(result_key, 0)

        # Update the node's statistics
        children_and_edge_visits = node.children_and_edge_visits.values()
        total_edge_visits = sum(edge_visits for (_, edge_visits) in children_and_edge_visits)
        node.N = 1 + total_edge_visits
        node.Q = (1 / node.N) * (
            node.U +
            sum(child.Q * edge_visits for (child, edge_visits) in children_and_edge_visits)
        )

    def search(self, n: int):
        """Perform n playouts from the root node."""
        print("doing a search in updated alg")
        for i in range(n):
            self.run()
    

GameState = Any
class MCTSNode2:
    def __init__(node, game_state: GameState):
        node.game_state: GameState = game_state # game state that this node is derived from

        node.is_terminal: bool = game_state.is_terminal
        node.is_expanded: bool = False
        node.children: list['MCTSNode2'] = []
        node.N: int = 0                  # number of visits
        node.W: int = 0                  # sum of all rewards 
        node.P: float = 1.               # transition probablity of reaching this state
        node.results = {1: 0, -1: 0, 0:0}
    
    @property
    def Q(self) -> float:
        return self.W / self.N #average value of node

class MCTS2:
    def __init__(tree, game_state: GameState):
        tree.root: MCTSNode2 = MCTSNode2(game_state)
        tree.nodes: dict[GameState, MCTSNode] = {game_state: tree.root}
    
    def get_node(tree, game_state: GameState) -> MCTSNode2:
        """Retrieve or create a new MCTSNode for the given game state."""
        if game_state not in tree.nodes:
            tree.nodes[game_state] = MCTSNode2(game_state=game_state)
        return tree.nodes[game_state]

    def best_child(tree, node: MCTSNode2) -> MCTSNode2:
        return max([child for child in node.children], key=tree.PUCT)

    def select(tree) -> list[MCTSNode2]:
        """MCTS Selection.
        The tree selects the most promising node until it reaches an unexpanded node"""
        node = tree.root 
        path = [node]
        while node.is_expanded and not node.is_terminal:
            node = tree.best_child(node)
            path.append(node)
        return path

    def expand(tree, path: list[MCTSNode2]) -> list[MCTSNode2]:
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

    def rollout(tree, node: MCTSNode2) -> int:
        """MCTS Rollout.
        Detach game from given MCTSNode, Simluate until end state and return reward"""
        game = node.game_state
        while not game.is_terminal:
            action = random.choice(game.all_legal_actions)
            game = game.transition(action)
        reward = game.result
        return reward

    def backprop(tree, path: list[MCTSNode2], value: int) -> None:
        """MCTS Backropagation. 
        The reward from this terminal state must be propagated up stream to update MCTS behavior"""
        value *= -path[-1].game_state.player
        for node in reversed(path):
            node.W += value 
            node.N += 1
            value = -value
            node.results[value] += 1
    
    def UCB(tree, node: MCTSNode2, c_param=2) -> float:
        """Returns (U)pper (C)onfidence (B)ound score for MCTSNode"""
        return node.Q + c_param * np.sqrt(np.log(node.parent.N)/node.N)

    def PUCT(tree, node: MCTSNode2, c_param=3) -> float:
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