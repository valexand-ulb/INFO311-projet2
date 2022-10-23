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
        "*** Selon le pseudocode présent sur wikipédia : https://fr.wikipedia.org/wiki/Algorithme_minimax***"
        self.maximise(self.depth, state)
        return self.move

    def maximise(self, i_depth, t_state: GameState):
        """
        Partie maximalisante de l'algorithme de minimax.
        Elle ne concerne que l'agent pacman (dont le but est de ne pas se faire manger par un fantome)
        """

        # condition finale ; si profondeur atteinte ou état gagnant/perdant
        if i_depth == 0 or t_state.isWin() or t_state.isLose():
            return self.evaluationFunction(t_state)

        d_all_moves = {}
        # Pacman à un indice 0, les autres fantomes ont un indice de 1 à 3
        for str_legal_action in t_state.getLegalActions(PACMAN):    # pour chaque action légales
            d_all_moves[str_legal_action] = self.minimise(i_depth-1,
                                                          t_state.getNextState(PACMAN, str_legal_action),
                                                          PACMAN+1)
        str_max_val_move = max(d_all_moves)
        self.move = str_max_val_move[1]
        return str_max_val_move[0]

    def minimise(self, i_depth, t_state: GameState, i_agent_index):
        """
        Partie minimalisante de l'algorithme de minimax.
        Elle ne concerne que les agents fantomes (dont le but est d'atteindre pacman)
        """
        # condition finale ; si profondeur atteinte ou état gagnant/perdant
        if i_depth == 0 or t_state.isWin() or t_state.isLose():
            return self.evaluationFunction(t_state)

        d_all_moves = {}
        # Pacman à un indice 0, les autres fantomes ont un indice de 1 à 3
        for str_legal_action in t_state.getLegalActions(i_agent_index):  # pour chaque action légales
            # Tant que l'index d'agent est plus petit que 3, c'est que l'on évalue un fantome -> appel à minimise
            if i_agent_index < t_state.getNumAgents():
                d_all_moves[str_legal_action] = self.minimise(i_depth,
                                                          t_state.getNextState(i_agent_index, str_legal_action),
                                                          i_agent_index+1)
            else:
                d_all_moves[str_legal_action] = self.maximise(i_depth,
                                                              t_state.getNextState(PACMAN, str_legal_action))
        str_min_move = min(d_all_moves)
        return str_min_move[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, state: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(state: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()    

# Abbreviation
better = betterEvaluationFunction
