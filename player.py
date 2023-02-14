import quarto_new as quarto
import random
import copy
from extended_quarto import ExtendQuarto
import numpy as np

#random player, always choose a random place and a random piece
class RandomPlayer(quarto.Player):

    def __init__(self, quarto:quarto.Quarto) -> None:
        super().__init__(ExtendQuarto(quarto))
        self.__extendQuarto = super().get_game()
        # random.seed()
    
    def choose_piece(self) -> int:
        #print(f"choose piece: {self.extendQuarto.possible_pieces()}")
        return random.choice(self.__extendQuarto.possible_pieces())
        

    def place_piece(self) -> tuple[int, int]:
        #return random.randint(0, 3), random.randint(0, 3)
        #print(f"place_piece: {self.extendQuarto.possible_positions_matrix()}")
        return random.choice(self.__extendQuarto.possible_positions_matrix())


#More intelligent then the random plyer
#The choose piece verify if a piece allow the opponent to win and does not select it 
#The place piece verify if there is a position that is a winning position and select it in case
#A random move and a random piece are selected instead  
class NaivePlayer(quarto.Player):
    
    def __init__(self, quarto:quarto.Quarto) -> None:
        super().__init__(ExtendQuarto(quarto))
        #get the ExtendQuarto from the Player class
        self.__extendQuarto = super().get_game()



    def choose_piece(self) -> int:
        pieces = self.__extendQuarto.safe_pieces(self.__extendQuarto.possible_pieces())
        return random.choice(pieces) if pieces else random.choice(self.__extendQuarto.possible_pieces())



    def place_piece(self) -> tuple[int, int]:
        #check all possible position and simulates placing the piece across an auxilary quarto
        for (x,y) in self.__extendQuarto.possible_positions_matrix():
            q_test = copy.deepcopy(self.__extendQuarto.quarto)
            if q_test.place(x,y) :
                #if the position brings into a victory situation, return the position
                if q_test.check_winner() != -1 :
                    return x,y
        return random.choice(self.__extendQuarto.possible_positions_matrix())



#This player always try to build a row of like pieces, that means a row where there are 3 pieces with at least one common characteristics
#The choose pieces first, find all the safe pieces (pieces with which the opponent can't win), then from this pieces try to find a lose state pieces
#that is a piece that if placed in a certain position guarantees victory, because there are two different rows with pieces with opposit caratteristics
#So, try to select one lose state piece, if there are none, a random safe piece is selected
#if again there are none, a random piece is selected
#The place piece first check if there is a winning position like the Naive player then try to build a row of like piece, avoiding though to create a lose state
class RiskyPlayer(quarto.Player):

    def __init__(self, quarto:quarto.Quarto) -> None:
        super().__init__(ExtendQuarto(quarto))
        self.__extendQuarto = super().get_game()



    def choose_piece(self) -> int:
        safe_pieces = self.__extendQuarto.safe_pieces(self.__extendQuarto.possible_pieces())
        pieces = self.__extendQuarto.lose_state_pieces(safe_pieces)
        return random.choice(pieces) if pieces else random.choice(safe_pieces) if safe_pieces else random.choice(self.__extendQuarto.possible_pieces())



    def place_piece(self) -> tuple[int, int]:
        
        for (x,y) in self.__extendQuarto.possible_positions_matrix():
            q_test = copy.deepcopy(self.__extendQuarto.quarto)
            if q_test.place(x,y) :
                if q_test.check_winner() != -1 :
                    return x,y

        #try to build a row of like pieces
        for (x,y) in self.__extendQuarto.possible_positions_matrix():
            q_test = copy.deepcopy(self.__extendQuarto)
            if q_test.quarto.place(x,y):
                
                #check if the current position create a lose state situation, and skip the position in case
                if q_test.check_horizontal_lose_state(y,3) or q_test.check_vertical_lose_state(x,3) or q_test.check_diagonal_lose_state(x,y,3):
                    continue
                #check if the current positon create a row of loke pieces, and return the position in case 
                if q_test.check_horizontal_like_pieces(y,3) or q_test.check_vertical_like_pieces(x,3) or q_test.check_diagonal_like_pieces(x,y,3):
                    if q_test.safe_pieces(q_test.possible_pieces()):
                        return x,y

        #select a random move if all the previous stategy failed
        return random.choice(self.__extendQuarto.possible_positions_matrix())



#This player always try to block a row where there are already three pieces
#The choose piece is the same as the Naive player
#The place piece first check if there is a winning position like the Naive player then 
#try all the possoble positions and select one that block a row
class BlockPlayer(quarto.Player):

    def __init__(self, quarto:quarto.Quarto) -> None:
        super().__init__(ExtendQuarto(quarto))
        self.__extendQuarto = super().get_game()



    def choose_piece(self) -> int:
        pieces = self.__extendQuarto.safe_pieces(self.__extendQuarto.possible_pieces())
        return random.choice(pieces) if pieces else random.choice(self.__extendQuarto.possible_pieces())



    def place_piece(self) -> tuple[int, int]:

        for (x,y) in self.__extendQuarto.possible_positions_matrix():
            q_test = copy.deepcopy(self.__extendQuarto.quarto)
            if q_test.place(x,y) :
                if q_test.check_winner() != -1 :
                    return x,y

        for (x,y) in self.__extendQuarto.possible_positions_matrix():
            q_test = copy.deepcopy(self.__extendQuarto)
            if q_test.quarto.place(x,y):
                #if the placed piece block a row return the position
                if q_test.check_block(x,y):
                    return x,y

        #select a random move if all other posibbilities failed            
        return random.choice(self.__extendQuarto.possible_positions_matrix())



#This player combine the strategy of the risky player and the block player
#The choose piece is the same of the Risky player
#The plece piece first use the Block player strategy and then use the Risky player strategy       
class BlockAndRiskyPlayer(quarto.Player):

    def __init__(self, quarto:quarto.Quarto) -> None:
        super().__init__(ExtendQuarto(quarto))
        self.__extendQuarto = super().get_game()



    def choose_piece(self) -> int:
        safe_pieces = self.__extendQuarto.safe_pieces(self.__extendQuarto.possible_pieces())
        pieces = self.__extendQuarto.lose_state_pieces(safe_pieces)
        return random.choice(pieces) if pieces else random.choice(safe_pieces) if safe_pieces else random.choice(self.__extendQuarto.possible_pieces())



    def place_piece(self) -> tuple[int, int]:
        
        for (x,y) in self.__extendQuarto.possible_positions_matrix():
            q_test = copy.deepcopy(self.__extendQuarto.quarto)
            if q_test.place(x,y) :
                if q_test.check_winner() != -1 :
                    return x,y

        for (x,y) in self.__extendQuarto.possible_positions_matrix():
            q_test = copy.deepcopy(self.__extendQuarto)
            if q_test.quarto.place(x,y):
                #if the placed piece block a row return the position
                if q_test.check_block(x,y):
                    return x,y
        
        #try to build a row of like pieces
        for (x,y) in self.__extendQuarto.possible_positions_matrix():
            q_test = copy.deepcopy(self.__extendQuarto)
            if q_test.quarto.place(x,y):
                
                #check if the current position create a lose state situation, and skip the position in case
                if q_test.check_horizontal_lose_state(y,3) or q_test.check_vertical_lose_state(x,3) or q_test.check_diagonal_lose_state(x,y,3):
                    continue
                #check if the current positon create a row of loke pieces, and return the position in case 
                if q_test.check_horizontal_like_pieces(y,3) or q_test.check_vertical_like_pieces(x,3) or q_test.check_diagonal_like_pieces(x,y,3):
                    if q_test.safe_pieces(q_test.possible_pieces()):
                        return x,y

        #select a random move if all the previous stategy failed
        return random.choice(self.__extendQuarto.possible_positions_matrix())
        

#Wrapper class for the different players used for training the QLearningPlayer
class OpponentWrapper():
    def __init__(self, inner_player, epsilon, freezed_player = True):
        self.inner_player = inner_player
        self.epsilon = epsilon
        #This parameter is True if the opponent in the training is a QLearning player False otherwise 
        self.freezed_player = freezed_player

    def start(self, state, valid_actions):
        if self.freezed_player:
            inner_action = self.inner_player.start(state, valid_actions)
            if np.random.random() <= self.epsilon:
                return np.random.choice(valid_actions)
        else:
            inner_action = self.step(state,valid_actions,0)    
        return inner_action

    def step(self, state, valid_actions, reward):
        if self.freezed_player:
            inner_action = self.inner_player.step(state, valid_actions, reward)
            if np.random.random() <= self.epsilon:
                return np.random.choice(valid_actions)
        else:
            #The opponent select the place and the piece that compose his next move using the methods of the different players
            pos = self.inner_player.place_piece()
            piece = self.inner_player.choose_piece()
            inner_action = ExtendQuarto.from_action_to_index((pos,piece))
        return inner_action

    def end(self, state, reward):
        if self.freezed_player:
            self.inner_player.end(state, reward)

    def get_freezed(self):
        raise NotImplementedError()