class MCTS:
    def __init__(self, node):
        #I'm just pointing to a node now
        self.node_pointer: MonteCarloTreeSearchNode = node

    #returns MonteCarloTreeSearchNode with highest Q value. 
    def best_action(self, num_simulations):
        self.search(num_simulations)
        best: MonteCarloTreeSearchNode = self.node_pointer.best_child(c_param=0.)
        return best

    #select and expand of mcts
    def _tree_policy(self):
        current_node = self.node_pointer
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()

        return current_node

    #rollout and backprop of mcts. This is one iteration of the algorithm
    def run(self):
        v = self._tree_policy()
        reward = v.rollout()
        v.backpropagate(reward)

    #searches the node for num_simulations
    def search(self, num_simulations):
        for _ in range(num_simulations):
            self.run()

    #this function should process each node to prune and abstract if possible
    def think(self):
        pass


""""
if best.n / simulations_number >= .99:
    best.parent = self.node_pointer.parent if self.node_pointer.parent else None
    self.node_pointer = best
    print(f"pure strategy with {best.n / simulations_number}")

new addition: this is a crude pure strategy detector
if it is a pure strategy, prune tree by skipping node
TODO Is it possible for the tree to be fine with multiple players being on the same depth n of the node?
"""
