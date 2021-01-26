from Goban import Board
from node import node
from Annexes.ml_go import make_board
import time
import math
import numpy as np

class mcts():
    """
    TODO :
    computer un temps t (ex : 3 min) d'exploration UCB (en utilisant notre cnn victory)
    --> en reprenant au noeud racine à chaque exploration
    faire les rollouts avec notre algo de choix de move
    update les probas de victoire des enfants
    choisir le coup
    """
    
    @staticmethod
    def create_first_node(black_stones = [], white_stones = []):
        init_node = node(father=None,
                board= make_board(sample={"black_stones" : black_stones, "white_stones":white_stones}),
                gnugo_board=Board(),
                move=None)
                
        print(f"[mcts.create_first_node] gnugoboard {Board()}")
        return init_node
    
    # def __init__():
    #     pass
    
    def chose_best_action(self, black_stones = [], white_stones= []):
        '''
        chose node to explore with UCB
        rollout + rollback over this node
        do again while time < 10 sec
        # TODO : vérifier que l'on ne dépasse pas 3 mn (start = time.time() & end = time.time() & durationSec = end-start)
        # TODO : explore all children
        # TODO : utiliser UCB pour choisir où continuer à creuser / faire un rollout
        '''
        n1 = mcts.create_first_node(black_stones,white_stones)
        start = time.time()
        n1.explore_node()

        count=0
        while time.time() - start < 10:
            # UCB_score = A + sqrt(ln(parentVisits)/visits)
            #TODO : Bad things below (nb_child_visit+1)
            UCB = np.argmax([child.A + math.sqrt(math.log(n1.nb_child_visit)/(child.nb_child_visit+1)) for child in n1.children])
            victory=n1.children[UCB].rollout()
            n1.children[UCB].rollback(victory)
            count+=1
            print(f"[mcts.chose_best_action] computed rollout-rollback n°{count}")
        print(f"[mcts.chose_best_action] Exploration time limit reached.")

        # pick best child => best A
        best_move = n1.children[np.argmax([child.A for child in n1.children])].move
        return best_move
        
