# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
from pacman import GameState
import util

from game import Agent

PACMAN = 0

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, state: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** Selon le pseudocode pr??sent sur wikip??dia : https://fr.wikipedia.org/wiki/Algorithme_minimax***"

        return self.maximise(self.depth, state)[1]  # maximise/minimise retourne (valeur, action)

    def maximise(self, i_depth, t_state: GameState):
        """
        Partie maximalisante de l'algorithme de minimax.
        Elle ne concerne que l'agent pacman (dont le but est de ne pas se faire manger par un fantome)
        """

        # condition finale ; si profondeur atteinte ou ??tat gagnant/perdant
        if i_depth == 0 or t_state.isWin() or t_state.isLose():
            return self.evaluationFunction(t_state), ''     # retourne un score et une action vide

        i_max_val, str_best_move = float('-inf'), ''
        # Pacman ?? un indice 0, les autres fantomes ont un indice de 1 ?? 2
        for str_legal_action in t_state.getLegalActions(PACMAN):    # pour chaque action l??gales
            i_temp_val, str_temp_action = self.minimise(i_depth,
                                                        t_state.getNextState(PACMAN, str_legal_action),
                                                        PACMAN+1)
            if i_max_val < i_temp_val:  # si la valeur d??termin??e est meilleure que la valeur maximale ...
                i_max_val, str_best_move = i_temp_val, str_legal_action

        return i_max_val, str_best_move

    def minimise(self, i_depth, t_state: GameState, i_agent_index):
        """
        Partie minimalisante de l'algorithme de minimax.
        Elle ne concerne que les agents fantomes (dont le but est d'atteindre pacman)
        """
        # condition finale ; si profondeur atteinte ou ??tat gagnant/perdant
        if i_depth == 0 or t_state.isWin() or t_state.isLose():
            return self.evaluationFunction(t_state), ''

        i_min_val, str_best_move = float('inf'), ''
        # Pacman ?? un indice 0, les autres fantomes ont un indice de 1 ?? 2
        for str_legal_action in t_state.getLegalActions(i_agent_index):  # pour chaque action l??gales d'un fantome
            # Tant que l'index d'agent est plus petit que 2, c'est que l'on ??value un fantome -> appel ?? minimise
            if i_agent_index < t_state.getNumAgents()-1:
                i_temp_val, str_temp_action = self.minimise(i_depth,
                                                            t_state.getNextState(i_agent_index, str_legal_action),
                                                            i_agent_index+1)
            else:
                i_temp_val, str_temp_action = self.maximise(i_depth-1,
                                                            t_state.getNextState(i_agent_index, str_legal_action))
            if i_min_val > i_temp_val:
                i_min_val, str_best_move = i_temp_val, str_legal_action

        return i_min_val, str_best_move


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, state: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** Encore selon le pseudocode pr??sent sur wikip??dia : https://fr.wikipedia.org/wiki/Algorithme_minimax ***"
        "*** avec l'ajout des variables alpha et beta ***"
        return self.maximise(self.depth, state)[1]  # maximise/minimise retourne (valeur, action)

    def maximise(self, i_depth, t_state: GameState, i_alpha=float('-inf'), i_beta=float('inf')):
        """
        Partie maximalisante de l'algorithme de minimax avec alpha-beta prunning.
        Elle ne concerne que l'agent pacman (dont le but est de ne pas se faire manger par un fantome)
        """

        # condition finale ; si profondeur atteinte ou ??tat gagnant/perdant
        if i_depth == 0 or t_state.isWin() or t_state.isLose():
            return self.evaluationFunction(t_state), ''     # retourne un score et une action vide

        i_max_val, str_best_move = float('-inf'), ''
        # Pacman ?? un indice 0, les autres fantomes ont un indice de 1 ?? 2
        for str_legal_action in t_state.getLegalActions(PACMAN):    # pour chaque action l??gales
            i_temp_val, str_temp_action = self.minimise(i_depth,
                                                        t_state.getNextState(PACMAN, str_legal_action),
                                                        PACMAN+1,
                                                        i_alpha,
                                                        i_beta)
            if i_max_val < i_temp_val:  # si la valeur d??termin??e est meilleure que la valeur maximale ...
                i_max_val, str_best_move = i_temp_val, str_legal_action
                i_alpha = max(i_max_val, i_alpha)

            if i_max_val > i_beta:  # si la valeur d??termin??e est moin bonne qu'une d??termin??e pr??c??demment...
                return i_max_val, str_best_move     # ...pas besoin d'expendre l'arbre

        return i_max_val, str_best_move

    def minimise(self, i_depth, t_state: GameState, i_agent_index, i_alpha, i_beta):
        """
        Partie minimalisante de l'algorithme de minimax avec alpha-beta prunning.
        Elle ne concerne que les agents fantomes (dont le but est d'atteindre pacman)
        """
        # condition finale ; si profondeur atteinte ou ??tat gagnant/perdant
        if i_depth == 0 or t_state.isWin() or t_state.isLose():
            return self.evaluationFunction(t_state), ''

        i_min_val, str_best_move = float('inf'), ''
        # Pacman ?? un indice 0, les autres fantomes ont un indice de 1 ?? 2
        for str_legal_action in t_state.getLegalActions(i_agent_index):  # pour chaque action l??gales d'un fantome
            # Tant que l'index d'agent est plus petit que 2, c'est que l'on ??value un fantome -> appel ?? minimise
            if i_agent_index < t_state.getNumAgents()-1:
                i_temp_val, str_temp_action = self.minimise(i_depth,
                                                            t_state.getNextState(i_agent_index, str_legal_action),
                                                            i_agent_index+1,
                                                            i_alpha,
                                                            i_beta)
            else:
                i_temp_val, str_temp_action = self.maximise(i_depth-1,
                                                            t_state.getNextState(i_agent_index, str_legal_action),
                                                            i_alpha,
                                                            i_beta)
            if i_min_val > i_temp_val:
                i_min_val, str_best_move = i_temp_val, str_legal_action
                i_beta = min(i_beta, i_min_val)

            if i_min_val < i_alpha:  # si la valeur trouv??e est moin bonne que la pr??c??dente...
                return i_min_val, str_best_move     # ...pas besoin d'??tendre l'arbre

        return i_min_val, str_best_move

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, state: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** Encore selon le pseudocode pr??sent sur wikip??dia : https://fr.wikipedia.org/wiki/Algorithme_minimax ***"
        "*** avec l'utilisation des slides S6 page 31***"
        return self.maximise(self.depth, state)[1]

    def maximise(self, i_depth, t_state: GameState):
        """
        Partie maximalisante de l'algorithme de minimax.
        Elle ne concerne que l'agent pacman (dont le but est de ne pas se faire manger par un fantome)
        """

        # condition finale ; si profondeur atteinte ou ??tat gagnant/perdant
        if i_depth == 0 or t_state.isWin() or t_state.isLose():
            return self.evaluationFunction(t_state), ''     # retourne un score et une action vide

        i_max_val, str_best_move = float('-inf'), ''
        # Pacman ?? un indice 0, les autres fantomes ont un indice de 1 ?? 2
        for str_legal_action in t_state.getLegalActions(PACMAN):    # pour chaque action l??gales
            i_temp_val, str_temp_action = self.minimise(i_depth,
                                                        t_state.getNextState(PACMAN, str_legal_action),
                                                        PACMAN+1)
            if i_max_val < i_temp_val:  # si la valeur d??termin??e est meilleure que la valeur maximale ...
                i_max_val, str_best_move = i_temp_val, str_legal_action

        return i_max_val, str_best_move

    def minimise(self, i_depth, t_state: GameState, i_agent_index):
        """
        La m??thode garde le nom de minimise m??me si il est plus convenable de lui attribuer un autre nom
        """
        # condition finale ; si profondeur atteinte ou ??tat gagnant/perdant
        if i_depth == 0 or t_state.isWin() or t_state.isLose():
            return self.evaluationFunction(t_state), ''

        i_sum_val, str_best_move = 0, ''
        # Pacman ?? un indice 0, les autres fantomes ont un indice de 1 ?? 2
        l_next_legals_actions = t_state.getLegalActions(i_agent_index)
        for str_legal_action in l_next_legals_actions:  # pour chaque action l??gales d'un fantome
            i_prob = 1 / len(l_next_legals_actions)     # probabilit?? d'une action
            # Tant que l'index d'agent est plus petit que 2, c'est que l'on ??value un fantome -> appel ?? minimise
            if i_agent_index < t_state.getNumAgents()-1:

                i_temp_val, str_temp_action = self.minimise(i_depth,
                                                            t_state.getNextState(i_agent_index, str_legal_action),
                                                            i_agent_index+1)
                i_sum_val += i_prob * i_temp_val    # selon la formule du slide : sum = P(a) * value (Result(s,a))

            else:
                i_temp_val, str_temp_action = self.maximise(i_depth-1,
                                                            t_state.getNextState(i_agent_index, str_legal_action))
                i_sum_val += i_prob * i_temp_val    # selon la formule du slide : sum = P(a) * value (Result(s,a))

            if i_sum_val > i_temp_val:
                i_min_val, str_best_move = i_temp_val, str_legal_action

        return i_sum_val, str_best_move

def betterEvaluationFunction(state: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    """
    

# Abbreviation
better = betterEvaluationFunction
