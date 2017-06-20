# multiAgents.py
# Edited by Xiao Tang
# github.com/namidairo777
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
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        #util.raiseNotDefined()
        numGhosts = gameState.getNumAgents() - 1
        return self.max_value(gameState, 1, numGhosts)
        
    def max_value(self, gameState, depth, numGhosts):
        """
            Maxmizing function of Minimax: Pacman Agent
        """
        # Game over
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        max_val = float("-inf")
        # successor is action (TOP, DOWN, LEFT, RIGHT)
        best_action = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            tempVal = self.min_value(successor, depth, 1, numGhosts)
            if tempVal > max_val:
                max_val = tempVal
                best_action = action
        # depth = 1 return action, otherwise return max_val
        if depth == 1:
            return best_action
        else:
            return max_val

    def min_value(self, gameState, depth, agentIndex, numGhosts):
        """
          Minimizing function of minimax: Ghost Agent
        """
        # Game over
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        min_val = float("inf")
        actions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        # last ghost
        if agentIndex == numGhosts:
            #  < maximal depth
            if depth < self.depth:
                for successor in successors:
                    min_val = min(min_val, self.max_value(successor, depth + 1, numGhosts))
            # == maximal depth, return value
            else:
                for successor in successors:
                    min_val = min(min_val, self.evaluationFunction(successor))
        # calculate next ghost  
        else:
            for successor in successors:
                min_val = min(min_val, self.min_value(successor, depth, agentIndex + 1, numGhosts))
        return min_val
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        numGhosts = gameState.getNumAgents() - 1
        return self.max_value(gameState, 1, numGhosts, float("-inf"), float("inf"))
        
    def max_value(self, gameState, depth, numGhosts, alpha, beta):
        """
            Maxmizing function alpha-beta pruning
        """
        # Game over
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        max_val = float("-inf")
        # successor is action (TOP, DOWN, LEFT, RIGHT)
        best_action = Directions.STOP
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            tempVal = self.min_value(successor, depth, 1, numGhosts, alpha, beta)
            if tempVal > max_val:
                max_val = tempVal
                best_action = action    
            # pruning
            if max_val > beta:
                return max_val
            alpha = max(alpha, max_val)
        # depth = 1 return action, otherwise return max_val
        if depth == 1:
            return best_action
        else:
            return max_val

    def min_value(self, gameState, depth, agentIndex, numGhosts, alpha, beta):
        """
            Minimizing function with alpha pruning
        """
        # Game over
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        min_val = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            # last ghost agent
            if agentIndex == numGhosts:
                # depth not over, move to pacman max layer
                if depth < self.depth:
                    tempVal = self.max_value(successor, depth + 1, numGhosts, alpha, beta)
                # depth over, return evaluation value
                else:
                    tempVal = self.evaluationFunction(successor)
            # next ghost min layer
            else:
                tempVal = self.min_value(successor, depth, agentIndex + 1, numGhosts, alpha, beta)
            if tempVal < min_val:
                min_val = tempVal        

            # pruning
            if min_val < alpha:
                return min_val
            beta = min(beta, min_val)
        return min_val

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        numGhosts = gameState.getNumAgents() - 1
        return self.max_value(gameState, 1, numGhosts)

    def max_value(self, gameState, depth, numGhosts):
        """
          Maxmizing function for Minimax algorithm
        """
        # Game over
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        max_val = float("-inf")
        best_action = Directions.STOP
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            temp_val = self.expect_val(successor, depth, 1, numGhosts)
            if temp_val > max_val:
                max_val = temp_val
                best_action = action
        if depth == 1:
            return best_action
        else:
            return max_val

    def expect_val(self, gameState, depth, agentIndex, numGhosts):
        """
          Different from min layer of Minimax, we return average value of expect layer
        """
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # return expectedVal
        actions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        expectedVal = 0
        prob = 1.0 / len(actions)
        # last ghost agent
        if agentIndex == numGhosts:
            # not over, next is max pacman layer
            if depth < self.depth:
                for successor in successors:
                    expectedVal += prob * self.max_value(successor, depth + 1, numGhosts)
            # depth is over, need to stop recursion
            else:
                for successor in successors:
                    expectedVal += prob * self.evaluationFunction(successor)
        # next ghost agent
        else:
            for successor in successors:
                expectedVal += prob * self.expect_val(successor, depth, agentIndex + 1, numGhosts)
        return expectedVal
            
                    

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

