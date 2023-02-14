import copy
from quarto_new import Quarto
import collections
import random
import numpy as np
from itertools import combinations, chain
from copy import deepcopy

WIN_REWARD = 100
DRAW_REWARD = -50
MAPPING = (0,1)

class ExtendQuarto():
    def __init__(self, quarto: Quarto) -> None:
        self.quarto = quarto
        self.available_pieces = list(range(16))
        self.available_positions = list(range(16))

    def __hash__(self):
        return (tuple([tuple(row) for row in self.quarto.get_board_status()]), self.quarto.get_player()).__hash__()
    
    def __repr__(self):
        return format(f"selected piece: {self.quarto.get_selected_piece()}, board: {self.quarto.get_board_status().ravel()}")


    ##################################################################
    # Methods used by the different players in player.py
    ##################################################################

    def get_pieces(self):
        return [self.quarto.get_piece_charachteristics(i) for i in range(16)]
        #return copy.deepcopy(self._Quarto__pieces)

    #Return the pieces that have not yet been placed
    def possible_pieces(self):
        ret = list(set([e for e in range(16)]) - set([e for e in self.quarto.get_board_status().ravel() if e != -1] + [self.quarto.get_selected_piece()]))
        
        #shuffle the result so the pieces will be considered in a random order
        random.shuffle(ret)
        return ret

    #Return the positions in the board which have not yet been occupied
    #the position is reppresented using a tuple of x and y -> (x,y)
    def possible_positions_matrix(self) -> tuple[int,int]:
        ret = []
        board = self.quarto.get_board_status()
        #print(board)
        for x in range(4):
            for y in range(4):
                if board[y,x] == -1:
                    ret.append((x,y))
        #shuffle the result so the positions will be considered in a random order
        random.shuffle(ret)
        return ret

    #Return the positions in the board which have not yet been occupied
    #The position in reppresented using an index from 0 to 15
    def possible_positions(self):
        ret = []
        board = self.quarto.get_board_status().ravel()
        #print(board)
        for pos in range(len(board)):
            if board[pos] == -1:
                ret.append(pos)

        #shuffle the result so the positions will be considered in a random order
        random.shuffle(ret)
        return ret

    #Starting from an ExtendQuarto find all the safe pieces (pieces with which the opponent can't win) inside the pieces list passed as parameter
    def safe_pieces(self,pieces):
        safe_pieces = []
        for piece in pieces:
            safe = True

            #try all the possible position with one piece and see if that position leads the opponent to win
            for (x,y) in self.possible_positions_matrix():
                q_test = copy.deepcopy(self.quarto)
                if q_test.select(piece):
                    if q_test.place(x,y):

                        #if the opponent with that piece can win the piece is not considered
                        if q_test.check_winner() != -1 :
                            safe = False
            if safe:
                safe_pieces.append(piece)
        return safe_pieces



    #Starting from an ExtendQuarto find all the lose state pieces (pieces that if placed in a certain position guarantees victory) inside the pieces list passed as parameter
    def lose_state_pieces(self,pieces):
        lose_state_pieces = []
        for piece in pieces:
            safe = False
            for (x,y) in self.possible_positions_matrix():
                q_test = copy.deepcopy(self)
                if q_test.quarto.place(x,y):

                    #verify if the piece in a certain position brings a lose state
                    if q_test.check_horizontal_lose_state(y,3) or q_test.check_vertical_lose_state(x,3) or q_test.check_diagonal_lose_state(x,y,3):
                        safe = True

            #only lose state pieces are considered
            if safe:
                lose_state_pieces.append(piece)
        return lose_state_pieces



    #Return two lists with the value of the diagonals
    #diag1 consider -> (0,0) (1,1) (2,2) (3,3)
    #diag2 consider -> (0,3) (1,2) (2,1) (3,0)  
    def take_diagonals(self):
        board = self.quarto.get_board_status()
        diag1 = []
        diag2 = []
        for i in range(self.quarto.BOARD_SIDE):
            diag1.append(board[i,i])
            diag2.append(board[i,self.quarto.BOARD_SIDE -1 -i])
        return diag1,diag2



    #Verify if the placed piece in (x,y) blocked a row
    def check_block(self,x,y):
        board = self.quarto.get_board_status()
        horizontal_line = collections.Counter(board[y,:])[-1] == 0
        vertical_line = collections.Counter(board[:,x])[-1] == 0

        #consider only the diagonal if the piece was placed there
        if x == y:
            diagonal_line = collections.Counter(self.take_diagonals()[0])[-1] ==0
        elif y == self.quarto.BOARD_SIDE -1 -x :
            diagonal_line = collections.Counter(self.take_diagonals()[1])[-1] ==0 
        else:
            diagonal_line = False
        if horizontal_line or vertical_line or diagonal_line:
            return True
        return False



    #Verify if the placed piece generates a lose state in the horizontal lines
    def check_horizontal_lose_state(self,placed_piece,n_pieces):
        board = self.quarto.get_board_status()
        pieces = self.get_pieces()
        rows = []

        #check that the row where the piece was placed contains 3 pieces, if not a lose state cannot occur
        if collections.Counter(board[placed_piece,:])[-1] == 1:
            for y in range(self.quarto.BOARD_SIDE):
                high_values = len([elem for elem in board[y,:] if elem >= 0 and pieces[elem].HIGH])
                coloured_values = len([elem for elem in board[y,:] if elem >= 0 and pieces[elem].COLOURED])
                solid_values = len([elem for elem in board[y,:] if elem >= 0 and pieces[elem].SOLID])
                square_values = len([elem for elem in board[y,:] if elem >= 0 and pieces[elem].SQUARE])
                low_values = len([elem for elem in board[y,:] if elem >= 0 and not pieces[elem].HIGH])
                noncolor_values = len([elem for elem in board[y,:] if elem >= 0 and not pieces[elem].COLOURED])
                hollow_values = len([elem for elem in board[y,:] if elem >= 0 and not pieces[elem].SOLID])
                circle_values = len([elem for elem in board[y,:] if elem >= 0 and not pieces[elem].SQUARE])
                rows.append((high_values,low_values,coloured_values,noncolor_values,solid_values,hollow_values,square_values,circle_values))

            #check if in a row there are 3 pieces with one characteristic and in another one there are 3 pieces with the opposit characteristic  
            for i in range(0,8,2):
                for k in range(self.quarto.BOARD_SIDE):
                    for j in range(k+1,self.quarto.BOARD_SIDE):
                        if (rows[k][i] == n_pieces and rows[k][i] == rows[j][i+1]) or (rows[k][i+1] == n_pieces and rows[k][i+1] == rows[j][i]):
                            return True
        return False



    #Verify if the placed piece generates a lose state in the vertical lines
    def check_vertical_lose_state(self,placed_piece,n_pieces):
        board = self.quarto.get_board_status()
        pieces = self.get_pieces()
        rows = []

        #check that the row where the piece was placed contains 3 pieces, if not a lose state cannot occur
        if collections.Counter(board[placed_piece,:])[-1] == 1:
            for x in range(self.quarto.BOARD_SIDE):
                high_values = len([elem for elem in board[:,x] if elem >= 0 and pieces[elem].HIGH])
                coloured_values = len([elem for elem in board[:,x] if elem >= 0 and pieces[elem].COLOURED])
                solid_values = len([elem for elem in board[:,x] if elem >= 0 and pieces[elem].SOLID])
                square_values = len([elem for elem in board[:,x] if elem >= 0 and pieces[elem].SQUARE])
                low_values = len([elem for elem in board[:,x] if elem >= 0 and not pieces[elem].HIGH])
                noncolor_values = len([elem for elem in board[:,x] if elem >= 0 and not pieces[elem].COLOURED])
                hollow_values = len([elem for elem in board[:,x] if elem >= 0 and not pieces[elem].SOLID])
                circle_values = len([elem for elem in board[:,x] if elem >= 0 and not pieces[elem].SQUARE])
                rows.append((high_values,low_values,coloured_values,noncolor_values,solid_values,hollow_values,square_values,circle_values))

            #check if in a row there are 3 pieces with one characteristic and in another one there are 3 pieces with the opposit characteristic  
            for i in range(0,8,2):
                for k in range(self.quarto.BOARD_SIDE):
                    for j in range(k+1,self.quarto.BOARD_SIDE):
                        if (rows[k][i] == n_pieces and rows[k][i] == rows[j][i+1]) or (rows[k][i+1] == n_pieces and rows[k][i+1] == rows[j][i]):
                            return True
        return False



    #Verify if the placed piece generates a lose state in the diagonal lines
    def check_diagonal_lose_state(self,x,y,n_pieces):
        board = self.quarto.get_board_status()
        pieces = self.get_pieces()
        rows = []
        high_values = []
        coloured_values = []
        solid_values = []
        square_values = []
        low_values = []
        noncolor_values = []
        hollow_values = []
        circle_values = []

        #check that the piece has been placed in a diagonal and if the diagonal contains 3 pieces, if not a lose state cannot occur
        if (y == x or y == self.quarto.BOARD_SIDE -1 -x) and collections.Counter(self.take_diagonals()[0])[-1] ==1 and collections.Counter(self.take_diagonals()[0])[-1] ==1:
            for i in range(4):
                if board[i, i] < 0:
                    continue
                if pieces[board[i, i]].HIGH:
                    high_values.append(board[i, i])
                else:
                    low_values.append(board[i, i])
                if pieces[board[i, i]].COLOURED:
                    coloured_values.append(board[i, i])
                else:
                    noncolor_values.append(board[i, i])
                if pieces[board[i, i]].SOLID:
                    solid_values.append(board[i, i])
                else:
                    hollow_values.append(board[i, i])
                if pieces[board[i, i]].SQUARE:
                    square_values.append(board[i, i])
                else:
                    circle_values.append(board[i, i])

            rows.append((len(high_values),len(low_values),len(coloured_values),len(noncolor_values),len(solid_values),len(hollow_values),len(square_values),len(circle_values)))
            
            high_values = []
            coloured_values = []
            solid_values = []
            square_values = []
            low_values = []
            noncolor_values = []
            hollow_values = []
            circle_values = []
            for i in range(4):
                if board[i, self.quarto.BOARD_SIDE - 1 - i] < 0:
                    continue
                if pieces[board[i, self.quarto.BOARD_SIDE - 1 - i]].HIGH:
                    high_values.append(board[i, self.quarto.BOARD_SIDE - 1 - i])
                else:
                    low_values.append(board[i, self.quarto.BOARD_SIDE - 1 - i])
                if pieces[board[i, self.quarto.BOARD_SIDE - 1 - i]].COLOURED:
                    coloured_values.append(
                        board[i, self.quarto.BOARD_SIDE - 1 - i])
                else:
                    noncolor_values.append(
                        board[i, self.quarto.BOARD_SIDE - 1 - i])
                if pieces[board[i, self.quarto.BOARD_SIDE - 1 - i]].SOLID:
                    solid_values.append(board[i, self.quarto.BOARD_SIDE - 1 - i])
                else:
                    hollow_values.append(board[i, self.quarto.BOARD_SIDE - 1 - i])
                if pieces[board[i, self.quarto.BOARD_SIDE - 1 - i]].SQUARE:
                    square_values.append(board[i, self.quarto.BOARD_SIDE - 1 - i])
                else:
                    circle_values.append(board[i, self.quarto.BOARD_SIDE - 1 - i])

            rows.append((len(high_values),len(low_values),len(coloured_values),len(noncolor_values),len(solid_values),len(hollow_values),len(square_values),len(circle_values)))
            
            #check if in one of the two diagonal there are 3 pieces with one characteristic and in the other one there are 3 pieces with the opposit characteristic  
            for i in range(0,8,2):
                if (rows[0][i] == n_pieces and rows[0][i] == rows[1][i+1]) or (rows[0][i+1] == n_pieces and rows[0][i+1] == rows[1][i]):
                    return True   
        return False



    #Verify if the placed piece generates a row of like pieces in the horizontal lines
    def check_horizontal_like_pieces(self,y,n_like_pieces):
        placed_piece = self.quarto.get_selected_piece()
        board = self.quarto.get_board_status()
        pieces = self.get_pieces()

        #check that the row where the piece was placed contains 3 pieces, if not a row of like pieces cannot occur
        if collections.Counter(board[y,:])[-1] == 1:
            high_values = [elem for elem in board[y,:] if elem >= 0 and pieces[elem].HIGH]
            coloured_values = [elem for elem in board[y,:] if elem >= 0 and pieces[elem].COLOURED]
            solid_values = [elem for elem in board[y,:] if elem >= 0 and pieces[elem].SOLID]
            square_values = [elem for elem in board[y,:] if elem >= 0 and pieces[elem].SQUARE]
            low_values = [elem for elem in board[y,:] if elem >= 0 and not pieces[elem].HIGH]
            noncolor_values = [elem for elem in board[y,:] if elem >= 0 and not pieces[elem].COLOURED]
            hollow_values = [elem for elem in board[y,:] if elem >= 0 and not pieces[elem].SOLID]
            circle_values = [elem for elem in board[y,:] if elem >= 0 and not pieces[elem].SQUARE]
            
            #verify if the placed piece has a certain characteristic and if the number of pieces with that characteristic is equal to n_like_pieces
            if (pieces[placed_piece].HIGH and len(high_values) == n_like_pieces) or (pieces[placed_piece].COLOURED and len(coloured_values) == n_like_pieces) or (pieces[placed_piece].SOLID and len(solid_values) == n_like_pieces) or (pieces[placed_piece].SQUARE and len(square_values) == n_like_pieces) or (not pieces[placed_piece].HIGH and len(low_values) == n_like_pieces) or (not pieces[placed_piece].COLOURED and len(noncolor_values) == n_like_pieces) or (not pieces[placed_piece].SOLID and len(hollow_values) == n_like_pieces) or (not pieces[placed_piece].SQUARE and len(circle_values) == n_like_pieces):
                return True
        return False


    
    #Verify if the placed piece generates a row of like pieces in the vertical lines
    def check_vertical_like_pieces(self,x,n_like_pieces):
        placed_piece = self.quarto.get_selected_piece()
        board = self.quarto.get_board_status()
        pieces = self.get_pieces()

        #check that the row where the piece was placed contains 3 pieces, if not a row of like pieces cannot occur
        if collections.Counter(board[:,x])[-1] == 1:
            high_values = [elem for elem in board[:,x] if elem >= 0 and pieces[elem].HIGH]
            coloured_values = [elem for elem in board[:,x] if elem >= 0 and pieces[elem].COLOURED]
            solid_values = [elem for elem in board[:,x] if elem >= 0 and pieces[elem].SOLID]
            square_values = [elem for elem in board[:,x] if elem >= 0 and pieces[elem].SQUARE]
            low_values = [elem for elem in board[:,x] if elem >= 0 and not pieces[elem].HIGH]
            noncolor_values = [elem for elem in board[:,x] if elem >=0 and not pieces[elem].COLOURED]
            hollow_values = [elem for elem in board[:,x] if elem >= 0 and not pieces[elem].SOLID]
            circle_values = [elem for elem in board[:,x] if elem >= 0 and not pieces[elem].SQUARE]

            #verify if the placed piece has a certain characteristic and if the number of pieces with that characteristic is equal to n_like_pieces
            if (pieces[placed_piece].HIGH and len(high_values) == n_like_pieces) or (pieces[placed_piece].COLOURED and len(coloured_values) == n_like_pieces) or (pieces[placed_piece].SOLID and len(solid_values) == n_like_pieces) or (pieces[placed_piece].SQUARE and len(square_values) == n_like_pieces) or (not pieces[placed_piece].HIGH and len(low_values) == n_like_pieces) or (not pieces[placed_piece].COLOURED and len(noncolor_values) == n_like_pieces) or (not pieces[placed_piece].SOLID and len(hollow_values) == n_like_pieces) or (not pieces[placed_piece].SQUARE and len(circle_values) == n_like_pieces):
                return True
        return False


    
    #Verify if the placed piece generates a row of like pieces in the diagonal lines
    def check_diagonal_like_pieces(self,x,y,n_like_pieces):
        placed_piece = self.quarto.get_selected_piece()
        board = self.quarto.get_board_status()
        pieces = self.get_pieces()
        high_values = []
        coloured_values = []
        solid_values = []
        square_values = []
        low_values = []
        noncolor_values = []
        hollow_values = []
        circle_values = []

        #check that the piece has been placed in a diagonal and if the diagonal contains 3 pieces, if not a like pieces row cannot occur
        if (y == x or y == self.quarto.BOARD_SIDE -1 -x) and collections.Counter(self.take_diagonals()[0])[-1] ==1 and collections.Counter(self.take_diagonals()[0])[-1] ==1:
            for i in range(4):
                if board[i, i] < 0:
                    continue
                if pieces[board[i, i]].HIGH:
                    high_values.append(board[i, i])
                else:
                    low_values.append(board[i, i])
                if pieces[board[i, i]].COLOURED:
                    coloured_values.append(board[i, i])
                else:
                    noncolor_values.append(board[i, i])
                if pieces[board[i, i]].SOLID:
                    solid_values.append(board[i, i])
                else:
                    hollow_values.append(board[i, i])
                if pieces[board[i, i]].SQUARE:
                    square_values.append(board[i, i])
                else:
                    circle_values.append(board[i, i])

            #verify if the placed piece has a certain characteristic and if the number of pieces with that characteristic is equal to n_like_pieces, in the first diagonal
            if (pieces[placed_piece].HIGH and len(high_values) == n_like_pieces) or (pieces[placed_piece].COLOURED and len(coloured_values) == n_like_pieces) or (pieces[placed_piece].SOLID and len(solid_values) == n_like_pieces) or (pieces[placed_piece].SQUARE and len(square_values) == n_like_pieces) or (not pieces[placed_piece].HIGH and len(low_values) == n_like_pieces) or (not pieces[placed_piece].COLOURED and len(noncolor_values) == n_like_pieces) or (not pieces[placed_piece].SOLID and len(hollow_values) == n_like_pieces) or (not pieces[placed_piece].SQUARE and len(circle_values) == n_like_pieces):
                return True
            high_values = []
            coloured_values = []
            solid_values = []
            square_values = []
            low_values = []
            noncolor_values = []
            hollow_values = []
            circle_values = []
            for i in range(4):
                if board[i, self.quarto.BOARD_SIDE - 1 - i] < 0:
                    continue
                if pieces[board[i, self.quarto.BOARD_SIDE - 1 - i]].HIGH:
                    high_values.append(board[i, self.quarto.BOARD_SIDE - 1 - i])
                else:
                    low_values.append(board[i, self.quarto.BOARD_SIDE - 1 - i])
                if pieces[board[i, self.quarto.BOARD_SIDE - 1 - i]].COLOURED:
                    coloured_values.append(
                        board[i, self.quarto.BOARD_SIDE - 1 - i])
                else:
                    noncolor_values.append(
                        board[i, self.quarto.BOARD_SIDE - 1 - i])
                if pieces[board[i, self.quarto.BOARD_SIDE - 1 - i]].SOLID:
                    solid_values.append(board[i, self.quarto.BOARD_SIDE - 1 - i])
                else:
                    hollow_values.append(board[i, self.quarto.BOARD_SIDE - 1 - i])
                if pieces[board[i, self.quarto.BOARD_SIDE - 1 - i]].SQUARE:
                    square_values.append(board[i, self.quarto.BOARD_SIDE - 1 - i])
                else:
                    circle_values.append(board[i, self.quarto.BOARD_SIDE - 1 - i])

            #verify if the placed piece has a certain characteristic and if the number of pieces with that characteristic is equal to n_like_pieces, in the second diagonal
            if (pieces[placed_piece].HIGH and len(high_values) == n_like_pieces) or (pieces[placed_piece].COLOURED and len(coloured_values) == n_like_pieces) or (pieces[placed_piece].SOLID and len(solid_values) == n_like_pieces) or (pieces[placed_piece].SQUARE and len(square_values) == n_like_pieces) or (not pieces[placed_piece].HIGH and len(low_values) == n_like_pieces) or (not pieces[placed_piece].COLOURED and len(noncolor_values) == n_like_pieces) or (not pieces[placed_piece].SOLID and len(hollow_values) == n_like_pieces) or (not pieces[placed_piece].SQUARE and len(circle_values) == n_like_pieces):
                return True
        return False


    #############################################################################
    # Methods used by QLearningPlayer for the training
    ##############################################################################

    #Static method that apply the transformation passed as parameeter to the state also passed as parameeter
    #The method return the transformated state
    @staticmethod
    def apply_field_transformation(state: list, transformation: list) -> list:
        _state_copy = deepcopy(state)
        _state = list(state)
        for _to, _from in enumerate(transformation):
            _state[_to] = _state_copy[_from]
        return _state

    #Static method that extract from a file a set of 32 mask, 
    # that reppresent all the possible transformation that can be applied to a single state to obtain an equivalent state,
    # these transformations are applied and together with the transformed state are added to the rest list
    
    @staticmethod
    def get_board_field_symmetries(state: tuple) -> list:
        ret = []
        transformations = []
        transformed_states = []
        with open(".\symmetries\symmetries.txt") as f_symm: 
            transformations = [list(map(lambda e: int(e) , l.strip().split(","))) for l in f_symm]
            
            for transformation in transformations:
                state_copy = list(deepcopy(state))
                transformed_state = ExtendQuarto.apply_field_transformation(state_copy, transformation)
                transformed_states.append(tuple([transformed_state, transformation]))
        _set = set(map(lambda ts: tuple(ts[0]), transformed_states))
        #print(len(_set))
        return transformed_states



    #Static method that apply the mask passed as parameeter to a set of valid actions
    # Map the valid actions of the original state into the valid actions of the transformed state
    @staticmethod
    def apply_rotation_mask_to_valid_actions(valid_actions,mask):
        ret = []

        for index in valid_actions:
            pos = list(mask).index(index//16)
            x = pos % 4
            y = pos // 4
            ret.append(ExtendQuarto.from_action_to_index(((x,y),index%16)))
            #ret.append((pos,index%16))
        return ret

    # @staticmethod
    # def apply_rotation_mask(board,mask):
        
    #     return [board[i] for i in mask]


    #Return all the possible moves based on the actual quarto board
    #The moves are mapped from 0 to 256 
    def possible_moves(self) -> list[int]:
        
        pieces = self.possible_pieces()
        #If pieces is empty select a random piece, but it is not important because the game will end with that move 
        if len(pieces) == 0:
            pieces = [random.randint(0,15)]
        return [16 * pos + piece for pos in self.possible_positions() for piece in pieces]

    def available_moves(self) -> list[int]:
        return [16 * pos + piece for pos in self.available_positions for piece in self.available_pieces]

    
    @staticmethod
    def all_moves() -> tuple[tuple[int, int], int]:
        for x in range(4):
            for y in range(4):
                for piece in range(16):
                    yield ((x,y), piece)  

    #Map an action in the format ((x,y),piece) into a index from 0 to 256                  
    @staticmethod
    def from_action_to_index(action: tuple[tuple, int]) -> int:
        pos, piece = action
        x, y = pos
        return 16 * (x + y * 4) + piece

    #Reset the quarto board and all the parameters used by the environment at the and of one match
    def reset_game(self):
        #self.quarto.reset_game()
        self.quarto.reset()
        self.available_pieces = list(range(16))
        self.available_positions = list(range(16))
        #start_piece = random.choice(self.available_pieces)
        start_piece = 15
        self.quarto.select(start_piece)
        self.available_pieces.remove(start_piece)
        return np.append(self.quarto.get_board_status().ravel(),start_piece), self.available_moves() 
    
    #Map an index from 0 to 256 to the corresponding anction ((x,y),piece)
    @staticmethod
    def from_index_to_action(index: int):
        piece = index % 16
        pos = index // 16

        x = pos % 4
        y = pos // 4

        return ((x,y), piece)

    #Apply the step for the envirorment
    def step(self, action: int):
        pose,piece = ExtendQuarto.from_index_to_action(action)

        position = action // 16
        piece = action % 16
        # put the piece on the board
        self.quarto.place(pose[0],pose[1])
        # print(position)
        #remove the position from the available ones
        self.available_positions.remove(position)

        # select the next piece for the opponent
        self.quarto.select(piece)
        #print("piece to remove",piece)
        #remove the piece from the available ones
        self.available_pieces.remove(piece)

        # check winner: -1 not finished, 0 player 1, 1 player 2
        if self.quarto.check_winner() != -1:
            #assign a reward of 100 for the victory
            reward = 100
            return np.append(self.quarto.get_board_status().ravel(),piece), reward, True, []
        
        if self.quarto.check_finished():
             #assign a reward of -50 for the draw, we want to push the victories states instead of the drawn ones 
            reward = -50
            return np.append(self.quarto.get_board_status().ravel(),piece), reward, True, []

        #If it is the last move, the env automatically play it and assign the reward
        #Reward of -100 for a lost
        if len(self.available_pieces) == 0:
            last_position = self.available_positions[0]
            last_x = last_position % 4
            last_y = last_position // 4
            self.quarto.place(last_x, last_y)
            self.available_positions.remove(last_position)
            status = self.quarto.check_finished()
            assert status 
            reward = -100 if self.quarto.check_winner() != -1 else -50
            return np.append(self.quarto.get_board_status().ravel(),piece), reward, True, []
        

        return np.append(self.quarto.get_board_status().ravel(),piece), 0, False, self.available_moves()



    ####################################################################
    # Methods for the MinMax player
    ####################################################################

    def available_moves_generator(self) -> tuple[tuple[int, int], int]:
        for piece in self.possible_pieces():
            for (x,y) in self.possible_positions_matrix():
                yield ((x,y), piece)    

    def make_move(self, move: tuple[tuple[int, int], int]):
        """
        This function copy the current board and apply the given move made by a tuple ((x, y), piece).
        @return 
            - new state: a copy of the current board after the application of the move
        """
        position = move[0]
        piece = move[1]
        x,y = position
        state_copy = deepcopy(self)
        
        state_copy.quarto.select(piece)
        state_copy.quarto.place(x, y)
        return state_copy 

    @staticmethod
    def get_board_field_symmetries_for_minmax(state: tuple) -> list:
       ret_transformations = []
       transformations = []
       transformed_states = []
       with open("./symmetries/symmetries.txt") as f_symm:
           transformations = [list(map(lambda e: int(e) , l.strip().split(","))) for l in f_symm]
          
           for transformation in transformations:
               state_copy = list(deepcopy(state))
               transformed_state = ExtendQuarto.apply_field_transformation(state_copy, transformation)
               if tuple(transformed_state) not in transformed_states:
                   transformed_states.append(tuple(transformed_state))
                   ret_transformations.append(tuple(transformation))
       return zip(transformed_states, ret_transformations)      

    def compute_children(self, piece_to_place: int) -> list:
        """
        Starting from the state computes the possible moves given a pice to place
        if the new state is not an equivalent state of the previous ones it will 
        be added to the return list otherwise is discarded.
        @param:
            - piece_to_place: the considered piece to placed
        @return:
            - a list of board with the all possible applied moves starting 
            from the initial one.
        """
        children = set()
        _ret = list()
        for pos, piece_to_pass in self.available_moves_generator():
            move_ = tuple([pos, piece_to_place])
            child = self.make_move(move_)
            
            tuple_child = child.quarto.get_board_status().ravel()
            if tuple(tuple_child) in children:
                continue
            else:
                children = children.union([fs[0] for fs in ExtendQuarto.get_board_field_symmetries_for_minmax(tuple_child)])
                _ret.append([child, tuple([pos, piece_to_pass])])
        return _ret

    def count_possible_pieces(self) -> int:
        """
        Return the number of pieces already played.
        """
        return len([piece for piece in self.quarto.get_board_status().ravel() if piece != -1])
    
    @staticmethod
    def invert_feature(piece: int, transformation:tuple) -> int:
        """
        Given a piece apply a mask and invert the features set to
        1 in the mask. If you apply this function twice you will
        get the original piece.
        @param
         - piece: int, the piece to invert features
         - mask: tuple, the mask to apply
        @return
         - a piece with the inverted features according to the passed
         mask
        """
        mask = 0
        for index in transformation:
            mask |= (1 << index)
        return piece ^ mask
    
    @staticmethod
    def apply_piece_transformation(state: tuple, transformation: tuple) -> tuple:
        """
        Given a state and a trasformation (a mask with 1 in position of the 
        feature to invert) return the equivalent transformed state.
        @param
         - state: tuple, a flatten board
         - trasformation: tuple, the mask to apply on each piece of the state
        @return
         - a copy of the state, but with transformed pieces
        """
        tr_state = [] 
        for piece in state:
            # iterate over pieces
            _piece = piece
            if piece != -1:
                # transform pieces
                _piece = ExtendQuarto.invert_feature(piece, transformation)
            tr_state.append(_piece)
        return tr_state
    
    @staticmethod
    def get_board_piece_symmetries(state: tuple) -> list:
        """
        Apply all the equivalent transformations to the board and return a list
        of equivalent transformed states and the applied transformation (a tuple)
        in order to come back to the original pieces.
        @param:
         - state: tuple, a flatten board
        @return
         - a list containing tuple made of the transformed state and the transformation 
        """
        ret = []
        features = list(range(4))

        transformations = [list(combinations(features, n)) for n in range(1,4)]
        transformations = chain(*transformations) 
        for transformation in transformations:
            transformed_state = ExtendQuarto.apply_piece_transformation(state, transformation)
            ret.append(tuple([transformed_state, transformation]))
        return ret
    
    
    @staticmethod
    def invert_move_symmetry(move: tuple, transformation: list):
        """
        Apply a field transformation on the move (x, y).
        @return 
            - a tuple with the transformed move 
        """
        x, y = move
        # compute index in the transformation
        _move = x + y * 4
        # get the new move
        _original_move = transformation.index(_move)
        # transform in tuple
        x = _original_move % 4
        y = _original_move // 4
        return x, y

    @staticmethod
    def share_features(group: np.array) -> bool:
        """
        Given a group (an array of pieces), compute the piece binary representation
        and check if all of them share a common feature.
        @param
         - a numpy array containing a set of pieces
        @return
         - a boolean set to True if pieces share a common feature, False otherwise
        """
        # binary rapresentation of features
        features = ["{0:b}".format(piece).zfill(4) for piece in group]
        for feature in range(4):
            # if all of them share a common feature return True
            if all(f[feature] == '0' for f in features):
                return True
            if all(f[feature] == '1' for f in features):
                return True
        return False
    
    @staticmethod
    def count_of_consecutives_3(state: tuple, row:bool=True) -> int:
        """
        Given a state (a flatten board) tell us how many rows or columns have 
        at least 3 pieces in them with a sharing feature and a void cell.
        @param:
         - state: a tuple containing a flatten board
         - row: a boolean that indicates if the check must be done by row or by column.
        @return
        The number of columns or rows which within 3 pieces share a feature
        """
        k = 3
        board = np.array(state).reshape((4,4))
        if not row:
            # transpose the board to do the column check
            board = board.T
        count = 0
        # get the combination of len 3 of the column/row index
        _combinations = list(combinations(range(4), 3))
        for row in board:
            # iter over the combinations and the index out of the mask
            for out, _combination in zip(range(3,-1,-1), _combinations):
                # apply the mask and get the elements
                tris = row[np.array(_combination)]
                if -1 in tris:
                    # don't care if there are three -1s.
                    continue
                single = row[out]
                if single != -1:
                    # don't care to count full line 
                    continue
                count += int(ExtendQuarto.share_features(tris))
        return count
        
    @staticmethod
    def count_of_consecutives_3_diag(state: tuple) -> int:
        """
        This method counts the number of diagonal and anti diagonal which 3 pieces share at least a feature
        and a void cell.
        @param
         - state: a flatten board
        @return
        An integer between 0 an 2 indicating the number of diagonal with 3 pieces that shares at least a feature. 
        """
        k = 3
        board = np.array(state).reshape((4,4))
        count = 0
        # get the combination of len 3 of the indexes
        _combinations = list(combinations(range(4), 3))
        #get diagonal and anti-diagonal
        diag = board.diagonal(0)
        anti_diag = np.fliplr(board).diagonal(0)
        # same as count_of_consecutives_3
        for d in [diag, anti_diag]:
            for out, _combination in zip(range(3,-1,-1), _combinations):
                tris = d[np.array(_combination)]
                if -1 in tris:
                    continue
                single = d[out]
                if single != -1:
                    continue
                count += int(ExtendQuarto.share_features(tris))
        return count
    
    @staticmethod
    def get_all_transformation(state: tuple):
        """
        Given a state computes the combination of field symmetry and piece
        symmetry on it.
        """
        _ret = set()
        for field_board, _ in ExtendQuarto.get_board_field_symmetries(state):
            for piece_board, _ in ExtendQuarto.get_board_piece_symmetries(field_board):
                _ret.add(tuple(piece_board))
        return _ret
