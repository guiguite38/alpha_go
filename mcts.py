from Goban import Board
from node import node
from Annexes.ml_go import make_board

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
    def create_first_node():
        init_node = node(father=None, board=make_board(sample={"black_stones" : [], "white_stones":[]}), gnugo_board=Board())
        return init_node

    def chose_best_action(self):
        n1 = mcts.create_first_node()
        # TODO : vérifier que l'on ne dépasse pas 3 mn (start = time.time() & end = time.time() & durationSec = end-start)
        # TODO : explore all children
        # TODO : utiliser UCB pour choisir où continuer à creuser / faire un rollout

        # TODO : Un jour peut être y mettre du RL (death)
        
