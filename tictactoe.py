import numpy as np

class TicTacToeMove:
    
    def __init__(self, row, col, value):
        self.row = row 
        self.col = col
        self.value = value

    def __repr__(self):
        return f"row:{self.row} col:{self.col} v:{self.value}"

class TicTacToeGameState():
    
    X = 1
    O = -1

    def __init__(self, state, next_to_move=1):
        self.board = state
        self.board_size = state.shape[0]
        self.next_to_move = next_to_move

    def __repr__(self):
        return f"{self.board}"

    def game_result(self):
        """
        this property should return
         1 if player #1 wins
        -1 if player #2 wins
         0 if there is a draw
         None if result is unknown
        """
        #check if game is over
        rowsum = np.sum(self.board, 0)
        colsum = np.sum(self.board, 1)
        diag_sum_tl = self.board.trace()
        diag_sum_tr = self.board[::-1].trace()

        player_one_wins = any(rowsum == self.board_size)
        player_one_wins += any(colsum == self.board_size)
        player_one_wins += (diag_sum_tl == self.board_size)
        player_one_wins += (diag_sum_tr == self.board_size)

        if player_one_wins:
            return self.X

        player_two_wins = any(rowsum == -self.board_size)
        player_two_wins += any(colsum == -self.board_size)
        player_two_wins += (diag_sum_tl == -self.board_size)
        player_two_wins += (diag_sum_tr == -self.board_size)

        if player_two_wins:
            return self.O

        if np.all(self.board != 0):
            return 0.

        return None

    def is_game_over(self):
        """
        returns boolean indicating if the game is over,
        simplest implementation may just be
        `return self.game_result() is not None`
        """
        return self.game_result() is not None

    def is_move_legal(self, move):
        if move.value != self.next_to_move:
            return False

        row_in_range = (0 <= move.row < self.board_size)
        if not row_in_range:
            return False

        col_in_range = (0 <= move.col < self.board_size)
        if not col_in_range:
            return False

        return self.board[move.row, move.col] == 0


    def move(self, move):
        if not self.is_move_legal(move):
            raise ValueError(f"move {move} on board {self.board} is not legal")

        new_board = np.copy(self.board)
        new_board[move.row, move.col] = move.value

        if self.next_to_move == TicTacToeGameState.X:
            new_next_to_move = TicTacToeGameState.O
        else:
            new_next_to_move = TicTacToeGameState.X

        return TicTacToeGameState(new_board, new_next_to_move)
        

    def get_legal_moves(self):
        """
        returns list of legal action at current game state
        """
        indices = np.where(self.board==0)
        return [TicTacToeMove(coords[0], coords[1], self.next_to_move) 
                for coords in list(zip(indices[0],indices[1]))]

