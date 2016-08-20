# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

# class Node:

#     def __init__(self, state, cost, parent):
#         self.state = state
#         self.cost = cost
#         self.parent = parent
    
#     def getState(self):
#         return self.state

#     def getCost(self):
#         return self.cost

#     def getParent(self):
#         return self.parent
#     def 

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """

    path = recDFS(problem, problem.getStartState(), [], [problem.getStartState()])[1]
    return path

def recDFS(problem, state, path, visited_states):
    print "Called recDFS with path ", path
    if problem.isGoalState(state):
        return [True, path]
    
    successors = problem.getSuccessors(state)
    if len(successors) == 0:
        return [False, []]
    else:
        for suc in successors:
            if suc[0] in visited_states: # this will be most efficent if it searches from the end of vis_states
                continue
            path.append(suc[1])
            visited_states.append(suc[0])
            result = recDFS(problem, suc[0], path, visited_states)
            if result[0]:
                return (True, result[1])
            else:  
                path.pop()
                visited_states.pop()
        return [False, []]

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    # # node tuple containing the state and it's parent
    # visited_nodes = [(problem.getStartState(), )]
    # # node queue containing tuple of (state, direction)
    # nodeQueue = util.Queue()
    # nodeQueue.push((problem.getStartState(),[]))
    # (state, cost, (state,cost,(state,cost,parent)))
    # print "Start state: ", problem.getStartState()
    # print "Successors ", problem.getSuccessors(problem.getStartState())
    # while not nodeQueue.isEmpty():
    #     nextNode = nodeQueue.pop()
    #     if problem.isGoalState(nextNode[0]):
    #         return nextNode[1]
    #     else:
    #         successors = problem.getSuccessors(nextNode[0])
    #         for suc in successors:
    #             if suc[0] in visited_states:
    #                 if cost is more
    #                     continue
    #             visited_states.append(suc[0])
    #             newPath = nextNode[1] + [suc[1]]
    #             nodeQueue.push((suc[0], newPath))
    
    # raise Error("Path no found")

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
