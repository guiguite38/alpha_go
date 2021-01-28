from Goban import Board
from node import node
from Annexes.ml_go import make_board
import time
import math
import numpy as np

class mcts():
    """
    This class makes a Monte-Carlo Tree Search on a given board, so as to return the best move.
    """
    
    @staticmethod
    def create_first_node(black_stones = [], white_stones = [], gnugo_board = Board()):
        """
        The Node class requires both a legacy board format, used to interact with the models 
        and a gnugo_board to interact with the Gnugo part of the software.
        This function initialises both and returns the root node for the tree search.
        """
        init_node = node(father=None,
                board= make_board(sample={"black_stones" : black_stones, "white_stones":white_stones}),
                gnugo_board=gnugo_board,
                move=None)                
        return init_node
        
    def chose_best_action(self, black_stones = [], white_stones= [], gnugo_board=Board()):
        """
        Uses Monte Carlo Tree Search to chose the best action.
        Nodes are explored according to the UCB algorithm.
        Rollouts are then computed on the chosen node until game ends, then rollbacks.
        Rollouts/Rollbacks continue while time < 10 sec.
        """
        n1 = mcts.create_first_node(black_stones,white_stones,gnugo_board=gnugo_board)
        start = time.time()
        n1.explore_node()

        count=0
        while time.time() - start < 10:
            # UCB_score = A + sqrt(ln(parentVisits)/visits)
            UCB = np.argmax([child.A + math.sqrt(math.log(n1.nb_child_visit)/(child.nb_child_visit+1)) for child in n1.children])
            victory=n1.children[UCB].rollout()
            n1.children[UCB].rollback(victory)
            count+=1
            print(f"[mcts.chose_best_action] computed rollout-rollback n°{count}")
        print(f"[mcts.chose_best_action] Exploration time limit reached.")

        # pick best child => best A
        best_move = n1.children[np.argmax([child.A for child in n1.children])].move
        return best_move