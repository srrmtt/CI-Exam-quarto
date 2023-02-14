from extended_quarto import ExtendQuarto

from functools import cache
from math import inf
from copy import deepcopy
from quarto import Player, Quarto

import random
class MinMaxAgent(Player):
    def __init__(self, quarto: Quarto, max_depth:int=3, scheduler:bool=False):
        """
        Instanciate the minmax cache, the quarto board, the maximum depth and the 
        scheduler boolean.
        """
        super().__init__(ExtendQuarto(quarto))
        # game instance
        self.extend_quarto = super().get_game()
        # random choice of the first piece to pass (don't care which is)
        self.selected_piece = random.choice(range(16))
        # boolean true if we want to use the depth scheduler
        self.use_scheduler = scheduler
        # maximum search depth
        self.max_depth = max_depth
        # transposition table
        self._cache = dict()
    
    def __repr__(self):
        return format(f"board: {self.extend_quarto.quarto.get_board_status()}")
    
    def minmax(self, state: ExtendQuarto, max_player=True, alpha=(-1, -7), beta=(1,7), max_depth=2, curr_depth=0, piece_to_place = None):
        """
        A cached recursive method that simulate the minmax algorithm for the quarto game.
        @params
            - state: a nim game stage
            - max_player: bool, maximizing fase if true
            - alpha: current alpha for the node parent
            - beta: current beta for the node parent
            - max_depth: maximum depth of the search
            - piece_to_place: piece from the opponent to place on the board
        @return
            - a tuple with the best move to play according to the algorithm (position, piece to pass)
        """
        def get_winner(state: ExtendQuarto):
            """
            Map the game evaluation in a more coherent way.
            """
            winner = state.quarto.check_winner()
            if  winner == -1:
                return 0
            if winner == 0:
                return 1
            return -1


        # contains all the moves to reach the final state
        current_depth = 0
        
        # terminal leaf
        if state.quarto.check_finished() or curr_depth == max_depth:
            state.winner = get_winner(state)
            # eval the heuristics
            heuristic_score = MinMaxAgent.heuristic(state.quarto.get_board_status().ravel())
            heuristic_score = -heuristic_score if not max_player else heuristic_score
            return (state.winner, heuristic_score), None, alpha, beta

        best_move = None
        state_tuple = tuple(state.quarto.get_board_status().ravel())
        # if the state has already been visited return the cached value      
        if max_player:
            # maximizing 
            best = (-inf, -7)
            # iterate over possible actions
            explored_states = set()
            for child, _move in state.compute_children(piece_to_place):
                # state key
                child_tuple = child.quarto.get_board_status().ravel()
                pos, piece_to_pass = _move
                # check if the state is present in the cache
                _cached = self.check_cache(child_tuple, max_player, curr_depth, beta)
                if _cached:
                    value, move, _ ,_ =  _cached
                else:
                    # recurr if there isn't a hit in cache
                    value, _, _, _ = self.minmax(
                        child, False, alpha, beta, max_depth, piece_to_place=piece_to_pass, curr_depth=curr_depth+1)
                    move = tuple([pos, piece_to_pass])

                if value > best:
                    best_move = move
                # update best and alpha
                best = max(best, value)
                alpha = max(alpha, best)
                # pruning
                if beta <= alpha:
                    break
            # update cache
            state_tuple = tuple(state.quarto.get_board_status().ravel())
            self._cache[(state_tuple, curr_depth, max_player)] = value, best_move, alpha, beta 
            curr_depth -= 1
            return best, best_move, alpha, beta
        else:
            # minimizing
            best = (+inf, 7)
            #iterate over possible states from current state
            for child, _move in state.compute_children(piece_to_place):
                child_tuple = child.quarto.get_board_status().ravel()
                pos, piece_to_pass = _move
                # check cache 
                _cached = self.check_cache(child_tuple, max_player, curr_depth, beta)
                if _cached:
                    value, move, _ ,_ =  _cached
                else:
                    # if no hit, recurr
                    value, _, _, _ = self.minmax(
                        child, True, alpha, beta, max_depth, piece_to_place=piece_to_pass, curr_depth=curr_depth+1)
                    move = tuple([pos, piece_to_pass])
                if value < best:
                    best_move = move
                # update best and beta
                best = min(best, value)
                beta = min(beta, best)

                if beta <= alpha:
                    break
            # update cache
            state_tuple = tuple(state.quarto.get_board_status().ravel())
            self._cache[(state_tuple, curr_depth, max_player)] = value, best_move, alpha, beta
            # rollback
            curr_depth -= 1
            return best, best_move, alpha, beta

    def choose_piece(self) -> int:
        """
        return the piece to pass to the opponent.
        """
        return self.selected_piece
    
    
        
        

    def place_piece(self) -> tuple[int, int]:
        """
        If the agent use a scheduler compute the maximum depth
        and then call minmax, otherwise use a fix max depth. Initialize
        the piece to pass to the opponent in self.selected_piece.
        @return
         - the position on which place the passed piece from the opponent.
        """
        piece = self.extend_quarto.quarto.get_selected_piece()
        if self.use_scheduler:
            self.max_depth = self.deep_scheduler()
        curr_depth = self.extend_quarto.count_possible_pieces()
        position, piece = self.minmax(self.extend_quarto, piece_to_place=piece, curr_depth=curr_depth, max_depth=curr_depth+self.max_depth)[1]
        self.selected_piece = piece
        return position

    def deep_scheduler(self) -> int:
        """
        Schedule the depth of the exploration considering the left pieces to
        place.
        @return
            - max_depth: int, an integer that specifies the maximum depth 
            of the alpha beta search.
        """
        pieces_to_place = len(
            list(
                filter(lambda piece: piece==-1, self.extend_quarto.quarto.get_board_status().ravel())))
        if pieces_to_place >= 10:
            return 3
        if pieces_to_place >= 6:
            return 4
        if pieces_to_place >= 3:
            return 5
        return 6
    
    @staticmethod
    def heuristic(state: tuple) -> int:
        """
        Compute the heuristic function on the given state. In this case the heuristic is the sum 
        of the number of rows, columns and diagonal with 3 pieces that share at least a feature
        and a blank space in the 4th.
        """
        return  ExtendQuarto.count_of_consecutives_3(state) + \
                ExtendQuarto.count_of_consecutives_3(state, row=False) + \
                ExtendQuarto.count_of_consecutives_3_diag(state)

    def check_cache(self, state: tuple, max_player: bool, depth: int, beta: tuple) -> tuple:
        """
        Given a state and the player check if an equivalent state is in the cache and return it.
        """
        for field_symm, field_transformation in ExtendQuarto.get_board_field_symmetries(state):
            for piece_symm, piece_transformation in ExtendQuarto.get_board_piece_symmetries(field_symm):
                # use equivalent board as key
                _key = tuple([tuple(piece_symm), depth, max_player])
                if _key in self._cache:
                    value, best_move, alpha_old, beta_old = self._cache[_key]
                    _best_pos, _best_piece = best_move 
                    # come back to the original move
                    _best_piece = ExtendQuarto.invert_feature(_best_piece, piece_transformation)
                    _best_pos = ExtendQuarto.invert_move_symmetry(_best_pos, field_transformation)
                    return value, (_best_pos, _best_piece), alpha, beta
                    
                    if beta < beta_old:
                        beta = value
        return None
