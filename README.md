# Final Project: Quarto-game

For the final project, we tried different strategies to produce the best possible player for the quarto game. We implement some players based on different fixed rules, a player based on minmax strategy and finally a RL player. The code was written and adapted for the first version of the quarto game, contained inside the quarto.py, so do not consider the binary board. To not modify the **Quarto class**, we introduce the **ExtendQuarto class**, which inherits from the Quarto class and introduces some important functions for the different players. Using different strategies, we realize different players that implement a **choose_piece** and a **place_piece** method that are used by the play function of the Quarto class. The choose_piece, return the Id of the piece that is passed to the opponent, while the place_piece, return the position on the quarto board, where to place the selected piece.

## **Fixed rule Players**  

For the players using fixed rules, we use a set of test on the board, that help the player for choosing the best place and the best piece to pass to the opponent, in particular we implement:  

-**Naive Player**: this player verifies if there is a winning position, and in case takes it, while if a piece allows the opponent to win, it is not selected. In all other cases, it is equal to a random player.  

-**Risky Player**: this player is an extension of the previous one, so for the cases mentioned before it behaves in the same way, but in addiction, if there is not a winning position, this player always tries to build a row of like pieces, that means a row where there are 3 pieces with at least one common characteristics.  

-**Block Player**: this player is an extension of the Naïve player, so for the cases mentioned before it behaves in the same way, but in addition, if there is not a winning position, this player always tries to block a row where there are already three pieces.  

-**Block and Risky Player**: this player combine the strategies of the Risky and the Block player, so first, check if there is a winning position, if there is not, try to block a row, where there are already three pieces, and if this move is not possible, try to build a row of like pieces.  

All the players that use fixed rules are optimal against a random player, in fact they all achieved more than 95% of win ratio against it. Moreover, the more intelligent player wins more than 50% of matches against the Naive Player, that is the most basic player. Below are the results of some test between different players (win ratio): 

- **Naive player** VS **Random player** : 96%  
- **Risk player** VS **Random player** : 99%  
- **Block player** VS **Random player** : 99%
- **Block and Risk player** VS **Random player** : 98%
- **Risk player** VS **Naive player** : 63%
- **Block player** VS **Naive player** : 59%
- **Block and Risk player** VS **Naive player** : 64%
- **Risk player** VS **Block player** : 51%
- **Block and Risk player** VS **Block player** : 47%
- **Block and Risk player** VS **Risk player** : 47%

## **RL Player**  

For the player that uses RL strategy, we take part of the code from  

https://github.com/school-of-ai-angers/quarto.git  

All the considered states are saved inside a q_table that is used for choosing the best action. The states are represented as a tuple, composed of all the fifteen board positions, with the piece placed on it, or -1 if empty, and the selected piece. The place_piece and the choose_piece method are for playing a match, while the train method is used for training the player against himself and different players. For the training we decide to assign a reward of 100 to a winning state, a -100 reward for losing state, and a reward of -50 for a draw state, this is because we want to push our player to select moves that bring to a win and not to a draw. To represent the actions taken by the player ((x,y),piece) we use a set of indexes that goes from 0 to 256, using a from_action_to_index and the from_index_to_action functions for performing the mapping. After the training we can freeze the player, so the q_table is no longer updated. Inside a match the player, before choosing a move, consider a set of equivalent boards, obtained by applying a set of mapping masks, and verify if a move is present that is better than the moves present in the original board.  

We tried different configurations for the RL player, but the results against a random player are not very good, in fact, this player can only win at least 60% of the matches. Despite the very high number of games played during training and the use of symmetries to derive equivalent boards, the number of possible states is too high, and the player can’t learn quite well. Moreover, we cannot train our player, or test it, against the fixed rule players or the minmax player, due to computational problems. In fact, the training with these players requires too much time, and does not improve the results, while the test is not feasible, because the dimension of the q_table after the training requires too much memory and slows down the other players.
The file that contains the mask used for computing the equivalent boards is taken from:   
https://sourceforge.net/p/quartoagent/code-0/HEAD/tree/

## **MinMax Player**  

**Heuristic Function**  

In our minmax solution we used an heuristic evaluation function to measure the goodness of the state: in particular the idea is that the higher the number of lines that can be completed with a given piece (rows, columns, or diagonals) the higher the chance that one of these lines can be ultimately completed. So, depending on whose turn is to be evaluated, this measure can be negated to serve well as heuristic function. By comparing the heuristic value to the value of a draw (0), the agent may tend to enforce a draw or take a risk in a certain direction. In particular, we use the numbers of lines with 3 pieces with an identical property as heuristic value, thus the range is in 0 - 7.   A game with any lines with 3 same features has an heuristic value of 0.  

**Symmetries**  

In this solution we consider both field symmetries and piece symmetries. The former indicates the mapping of equivalent boards rotating or mirroring the original one, the latter exploits the fact that changing the same features in all the pieces of a board brings us to an equivalent board . The field symmetries have been computed trough constraint programming using Choco, this part has been taken from this paper and all the symmetries are taken from allSimmetries.txt. Symmetries allow us to not recompute the tree from a single if this an equivalent board has already been explored.  

**Transposition table**  

Transposition tables are used to store partial results in order to avoid the exploration of a state that has been already explored. In this table we store the state and the player (min or max) as key and the heuristic evaluation, the best move found and alpha and beta as value. To improve the chance of encountering the state in the table we also use the symmetries equivalence during the lookup.  

**Deep Scheduler**  

Since starting with a high value for the max depth is not feasible we chose to start with a low value (3) and go up to 7 while the available pieces are reduced in number. In this way when there are less states to evaluate and we are in the final stages of the game we can make more accurate predictions.  


-MinMax Agent (depth=2)	75%  

-MinMax Agent (depth=3)	82%  

-MinMax Agent (deep schedule)	84%

## Conclusions
Due to time limitations, we cannot perform a lot of test between the different models, but we can say that best players in term of performances are the **Fixed rule players**, that obtain the best performance against random player, and also against **MinMax player**, while the worst one is the **RL player**, that achieve only 60% win ratio against the random player. These poor performances are mainly due to the computational limits, which didn't allow us to train the RL against stronger players, but only against himself and the random player.




