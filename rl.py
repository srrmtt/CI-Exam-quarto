from extended_quarto import ExtendQuarto, WIN_REWARD, DRAW_REWARD

from functools import cache
from math import inf
from copy import deepcopy
from quarto_new import Player, Quarto
import pickle 
import numpy as np
import random
import sys
from tqdm import tqdm
from player import *


class QLearningPlayer(Player):
    def __init__(self, quarto, train_mode):

        super().__init__(ExtendQuarto(quarto))
        self.extended_quarto = super().get_game()
        # Whether we are in training or playing mode
        # In training mode, this player will update its Q-table
        # and sometimes take a random action to explore more
        self.train_mode = train_mode

        # This agent's Q-table.
        # It is a map from state to action value pre action:
        # q_table[state][action]: float
        self.q_table = {}
        self.action_space = 256
        
        # Epsilon scheduling
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.99995
        
        # Q-table update hyperparameters
        self.alpha = 0.1
        self.gamma = 1
        
        # Q-table update helper variables
        self.prev_state = None
        self.prev_action = None

        # piece to be passed
        self.piece_to_pass = None

    @staticmethod
    def from_index_to_action(index: int):
        piece = index % 16
        pos = index // 16

        x = pos % 4
        y = pos // 4

        return ((x,y), piece)

    def set_quarto(self,quarto:Quarto):
        self.extended_quarto = ExtendQuarto(quarto)

    #Method invoked by the run method of the quarto
    #Compute witch is the next action to take
    #Return the position where to place the selected piece
    #Assign to self.piece_to_pass the piece to pass to the opponent
    def place_piece(self) -> tuple[int, int]:
        sel_piece = self.extended_quarto.quarto.get_selected_piece()
        state = np.append(self.extended_quarto.quarto.get_board_status().ravel(),sel_piece)
        valid_actions = self.extended_quarto.possible_moves()
        action_idx = self._take_action(state, valid_actions)
        action, piece = ExtendQuarto.from_index_to_action(action_idx)
        self.piece_to_pass = piece
        return action

    #Method invoked by the run method of the quarto
    #Return the piece to pass to the opponent computed in place_piece()
    def choose_piece(self) -> int:
        state = self.extended_quarto.quarto.get_board_status().ravel()
        if state.all(-1):
            piece = random.choice(self.extended_quarto.possible_pieces())
            return piece
        return self.piece_to_pass

    def start(self, state, valid_actions):
        # First move: take the action
        return self._take_action(state, valid_actions)

    def step(self, state, valid_actions, reward):
        # At every other step: update the q-table and take the next action
        # Since the game hasn't finished yet, we can use current knowledge of the q-table
        # to estimate the future reward.
        if self.train_mode:
            action_values = self._get_action_values(state, valid_actions)
            self._update_q_table(reward + self.gamma * np.max(action_values))
        return self._take_action(state, valid_actions)

    def end(self, state, reward):
        # Last step: update the q-table and schedule the next value for epsilon
        # Here, the expected action-value is simply the final reward
        if self.train_mode:
            self._update_q_table(reward)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def _initialize_state(self):
        return np.zeros(self.action_space)
        #return np.array([random.uniform(0,0.1) for _ in range(self.action_space)])

    def _update_q_table(self, new_value):
        # Based on the reward of the previous action take at the previous step,
        # update the q-table to be closer to the desired value.
        # Note that, if `alpha` is zero, the q-table is left unchanged,
        # if `alpha` is one, the q-table will simply take the `new_value`.
        # With a value in between, one can control the tradeoff between learning too much
        # or too little from a single move
        prev_state = tuple(self.prev_state)
        q_row = self.q_table.setdefault(prev_state, self._initialize_state())
        q_row[self.prev_action] += self.alpha * (new_value - q_row[self.prev_action])

    #Compute all the equivalent state of the one passed as parameeter
    #For every state select transform the valid_actions passed as parameeter and select the valid action with the higher score
    #Select the valid action that is the highest among all states and select the correct valid action for the non transformated state 
    def _get_action_values_considering_symmetries(self,state,valid_actions):
        best_action_values = [] 
        board = state[:-1]
        sel_piece = state[-1]
        rotations_list = ExtendQuarto.get_board_field_symmetries(board)
        for rotate_board,board_mask in rotations_list:
            transf_valid_actions = ExtendQuarto.apply_rotation_mask_to_valid_actions(valid_actions,board_mask)
            action_values = self._get_action_values(rotate_board + [sel_piece], transf_valid_actions)
            best_action_values.append((np.max(action_values),np.argmax(action_values),board_mask))
        best_action = max(best_action_values)
        return valid_actions[best_action[1]]
    
    def _take_action(self, state, valid_actions):
        # Store the current state, copying it, otherwise the environment could mutate it afterwards
        self.prev_state = state.copy()
        
        if self.train_mode and np.random.random() <= self.epsilon:
            # Take a random action
            self.prev_action = np.random.choice(valid_actions)
        else:
            # Take the action that has the highest expected future reward,
            # that is, the highest action value
            if self.train_mode: 
                action_values = self._get_action_values(state, valid_actions)
                self.prev_action = valid_actions[np.argmax(action_values)]
            else:
                #For choosing the action consider also the equivalent states, but not during the training 
                self.prev_action = self._get_action_values_considering_symmetries(state,valid_actions)
        return self.prev_action
    
    def _get_action_values(self, state, valid_actions):
        # Convert from numpy array to tuple
        state = tuple(state)
        if self.train_mode:
            # Return saved action values. If this is the first time this state is visited,
            # set all values to zero
            return self.q_table.setdefault(state, self._initialize_state())[valid_actions]
        # When not in train mode, do not change the Q-table, just return a new default for
        # every new never-visited state
        return self.q_table.get(state, self._initialize_state())[valid_actions]

    def get_freezed(self):
        # Return a copy of the player, but not in train_mode
        # This is used by the training loop, to replace the adversary from time to time
        copy = deepcopy(self)
        copy.train_mode = False
        return copy

    def save(self):
        # Save the q-table on the disk for future use
        with open('./trained_players/player.bin', 'wb') as fp:
            pickle.dump(self, fp, protocol=4)

#Train the player against different type of players   
def train(env, player,train_episodes=10000, eval_episodes=1000, cycles=10, on_cycle_end=None, opponent_epsilon=0.1, eval_player=None):
    """
    Train the given player against it self
    """
    random_player = OpponentWrapper(RandomPlayer(player.extended_quarto.quarto),opponent_epsilon,freezed_player=False)
    best_score = (0.0,0.0)
    for cycle in range(cycles):
        
        adversary = OpponentWrapper(player.get_freezed(), opponent_epsilon,freezed_player=True)

        # Train player against a fixed adversary
        win_rate, draw_rate = run_duel(env, player, adversary, train_episodes)
        
        # Eval the newly trained player against the fixed adversary
        curr_player = OpponentWrapper(player.get_freezed(), opponent_epsilon)
        eval_score_h = run_duel(
            env,
            curr_player,
            adversary,
            eval_episodes)
        
        eval_score_r = run_duel(
            env,
            curr_player,
            random_player,
            eval_episodes)
        
        if best_score < (win_rate,draw_rate):
            best_score = (win_rate,draw_rate)
            player.save()
            print(f"New best player saved:{best_score}")

        adversary = curr_player

        print(
            f'Cycle {cycle+1}/{cycles}: win rate = {win_rate}, draw rate = {draw_rate}')
        print(f"score against himself: {eval_score_h}")
        print(f"score against random: {eval_score_r}")
        if on_cycle_end:
            on_cycle_end(cycle)

#Run a episodes number of match between player1 and player2 
def run_duel(env, player1, player2, episodes) -> tuple[float, float]:
    """
    :param env: Environment
    :param player1: BasePlayer
    :param player2: BasePlayer
    :param episodes: int
    :returns: tuple[float, float] - player 1 win rate and draw rate
    """
    win_rate = 0
    draw_rate = 0
    for _ in tqdm(range(episodes)):
        win_score, draw_score = run_match(env, player1, player2)
        win_rate += win_score
        draw_rate += draw_score
    win_rate /= episodes
    draw_rate /= episodes
    return win_rate, draw_rate

#Entry move for one player
def entry_move(player, env, state, valid_actions):
    action = player.start(state, valid_actions)
    return env.step(action)

#Run a single match between player and opponent
def run_match(env, player, opponent):
    """
    :param env: Environment
    :param player: BasePlayer
    :param opponent: BasePlayer
    :returns: float - the score of the player 1
    """
    # 1 player start, 0 viceversa 
    turn = random.choice([0,1])
    
    # Reset
    state, valid_actions = env.reset_game()

    if turn:
        # Player 1 first action
        state, reward_1, done, valid_actions = entry_move(player, env, state, valid_actions)
        # Player 2 first action
        state, reward_2, done, valid_actions = entry_move(opponent, env, state, valid_actions)
    else:
        # Player 2 first action
        state, reward_2, done, valid_actions = entry_move(opponent, env, state, valid_actions)
        # Player 1 first action
        state, reward_1, done, valid_actions = entry_move(player, env, state, valid_actions)
    while True:
        if turn:
            # Player 1 turn
            action = player.step(state, valid_actions, reward_1-reward_2)
            state, reward_1, done, valid_actions = env.step(action)
            if done:
                player.end(state, reward_1)
                win = 1 if reward_1 == WIN_REWARD else 0
                return  win, not win 
        else:
            # Player 2 turn
            action = opponent.step(state, valid_actions, reward_2-reward_1)
            state, reward_2, done, valid_actions = env.step(action)
            if done:
                player.end(state, reward_1-reward_2)
                lose = 1 if reward_2 == -WIN_REWARD else 0
                return  0, not lose 
        # switch turn
        turn = not turn
