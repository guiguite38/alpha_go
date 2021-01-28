# Projet d'Intelligence Artificielle AlphaGO
### Mené à contribution égales par G. Grosse, M. Mahaut, B. Nicol

## PROBLEMATIQUE
Mettre en place un algorithme de type Monte Carlo Tree Search - MCTS - renforcé par du machine learning pour jouer au Go.

## CONTEXTE
Projet : Projet Applicatif AlphaGo
Spécialité de filière : Intelligence Artificielle
Ecoles d'ingénieurs associées : ENSC et ENSEIRB  
Durée : 3 mois

## PREMIERS PAS / EXECUTION
Executer -- localGame.py --

## CONTENU DU PROJET / PLUS-VALUE APPORTEE

mcts.py
-------
Makes a Monte-Carlo Tree Search on a given board, so as to return the best move.

myPlayer.py
-----------
Player relying on mcts predictions.

node.py
-------
Class used for mcts computations : explorations, rollouts, rollbacks.

Annexes.ml_go.py
----------------
Different tools to define, train and interact with models.


## Pistes d'amélioration :
    Le programme peut être rendu plus intelligent en modifiant ces trois éléments : 
        -   Le temps d'exploration de l'arbre : soit en donnant plus de 10s de temps de calcul (peu agréable pour l'utilisateur), soit en accélerant l'exploration de l'arbre. Notamment, nous effectuons à chaque nouveau noeud de l'arbre une copie complète de l'objet gnugoboard, alors que les fonctions push et pull existent pour justement éviter cela.

        -   La qualité du rollout :i.e. sur un même temps de calcul des explorations plus intéressantes. Pour cela, la génération de nouvelles données en self play permettrait de retravailler ce modèle. Une augmentation experte des données, comme l'ajout du nombre de libertés pour chaque pierre peut aussi être envisagée.

        - La qualité du value network : pourrait lui aussi bénéficier de nouvelles données améliorant la qualité du modèle. 

    Dans la continuation de ce projet, une fois ce travail sur l'intelligence terminé, nous aurions pu tenter d'agrandir le plateau à 13*13, voir à un 19*19.


## Fichiers pré-existants

Goban.py
---------

Fichier contenant les règles du jeu de GO avec les fonctions et méthodes pour parcourir (relativement) efficacement
l'arbre de jeu, à l'aide de legal_moves() et push()/pop() comme vu en cours.

Ce fichier sera utilisé comme arbitre dans le tournoi. Vous avez maintenant les fonctions de score implantés dedans.
Sauf problème, ce sera la methode result() qui donnera la vainqueur quand is_game_over() sera Vrai.

Vous avez un décompte plus précis de la victoire dans final_go_score()

Pour vous aider à parcourir le plateau de jeu, si b est un Board(), vous pouvez avoir accès à la couleur de la pierre
posée en (x,y) en utilisant b[Board.flatten((x,y))]


GnuGo.py
--------

Fichier contenant un ensemble de fonctions pour communiquer avec gnugo


starter-go.py
-------------

Exemples de deux développements aléatoires (utilisant legal_moves et push/pop). Le premier utilise legal_moves et le
second weak_legal_moves, qui ne garanti plus que le coup aléatoire soit vraiment légal (à cause des Ko).

La première chose à faire est probablement de 


localGame.py
------------

Permet de lancer un match de myPlayer contre lui même, en vérifiant les coups avec une instanciation de Goban.py comme
arbitre. Vous ne devez pas modifier ce fichier pour qu'il fonctionne, sans quoi je risque d'avoir des problèmes pour
faire entrer votre IA dans le tournoi.


playerInterface.py
------------------

Classe abstraite, décrite dans le sujet, permettant à votre joueur d'implanter correctement les fonctions pour être
utilisé dans localGame et donc, dans le tournoi. Attention, il faut bien faire attention aux coups internes dans Goban
(appelés "flat") et qui sont utilisés dans legal_moves/weak_legal_moves et push/pop des coups externes qui sont
utilisés dans l'interface (les named moves). En interne, un coup est un indice dans un tableau 1 dimension
-1, 0.._BOARDSIZE^2 et en externe (dans cette interface) les coups sont des chaines de caractères dans "A1", ..., "J9",
"PASS". Il ne faut pas se mélanger les pinceaux.


myPlayer.py
-----------

Fichier que vous devrez modifier pour y mettre votre IA pour le tournoi. En l'état actuel, il contient la copie du
joueur randomPlayer.py


randomPlayer.py
---------------

Un joueur aléatoire que vous pourrez conserver tel quel


gnugoPlayer.py
--------------

Un joueur basé sur gnugo. Vous permet de vous mesurer à lui simplement.


namedGame.py
------------

Permet de lancer deux joueurs différents l'un contre l'autre.
Il attent en argument les deux modules des deux joueurs à importer.


EXEMPLES DE LIGNES DE COMMANDES:
================================

python3 localGame.py
--> Va lancer un match myPlayer.py contre myPlayer.py

python3 namedGame.py myPlayer randomPlayer
--> Va lancer un match entre votre joueur (NOIRS) et le randomPlayer
 (BLANC)

 python3 namedGame gnugoPlayer myPlayer
 --> gnugo (level 0) contre votre joueur (très dur à battre)


