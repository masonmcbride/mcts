class MCTS:
    def __init__(self, node):
        self.node_pointer = node

    def best_action(self, simulations_number):
        for _ in range(0, simulations_number):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        best = self.node_pointer.best_child(c_param=0.)
        if best.n / simulations_number >= .99:
            print(f"pure strategy with {best.n / simulations_number}")
        return best

    def _tree_policy(self):
        current_node = self.node_pointer
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()

        return current_node
