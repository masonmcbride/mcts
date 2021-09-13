import numpy as np
from collections import defaultdict

class MonteCarloTreeSearchNode():

    def __init__(self, state, parent=None, action_taken=None):
        """
        state : 3x3 np array that holds board state
        parent : parent node 
        action_taken : stores the Action object taken to get to that node
        children : list of children, initially empty
        
        q = returns value of node
        n = _number_of_visits : number of visits to node
        _results : holds a dictionary of all results 
        _untried_actions : holds a list of all untried actions 
        _weights : holds the weights of each action
        """
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._untried_actions = None
        self._weights = None

    def __str__(self):
        return f"MCTS Search Node\nn: {self.n}\npi: \n{self.dist}\nstate:\n{self.state}"

    @property
    def untried_actions(self):
        """
        returns list of untried actions
        this updates every single time this property is called. 
        O(n) I think :/
        """
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_moves()
        return self._untried_actions

    @property
    def q(self):
        wins = self._results[self.parent.state.next_to_move]
        loses = self._results[-1 * self.parent.state.next_to_move]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    @property
    def dist(self):
        if self.parent is not None and self.parent._weights is not None:
            out = np.zeros((3,3))
            pi = MonteCarloTreeSearchNode.normalize([c.n for c in self.parent.children])
            for prob, child in zip(pi, self.parent.children):
                a = child.action_taken
                out[a.row][a.col] = prob
            return np.round(out, 3)
        else:
            return None

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(next_state, parent = self, action_taken = action)
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        return self.untried_actions == 0

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_moves()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param = 1.4):
        self._weights = [
                (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n)) 
                for c in self.children]
        return self.children[np.argmax(self._weights)]

    def rollout_policy(self, possible_moves):        
        #roll out policy is random moves
        return possible_moves[np.random.randint(len(possible_moves))]

    @staticmethod
    def normalize(array):
        norm = sum(array)
        normalized = np.array(array)/norm
        return normalized
