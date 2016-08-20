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

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to ch   ange anything in this class, ever.
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

        print state;
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
    return  [s, s, w, s, w, w, s,w]



dict = {}
goalState=0
class state:
    def __init__(self,state,cost,action,parents):
        self.state=state
        self.cost = cost
        self.action = action
        self.parents = parents

    def getCost(self):
        return  self.cost

    def setCost(self,cost):
        self.cost=cost

    def getParents(self):
        return self.parents

    def setParents(self, parents):
        self.parents = parents

    def setAction(self,action):
        self.action=action

    def getAction(self):
        return self.action

    def __str__(self):
        return "S:%s C:%s P:%s" %(self.state,self.cost,self.parents)

def depthFirstSearch(problem):

        print "Start:", problem.getStartState()
        print "Is the start a goal?", problem.isGoalState(problem.getStartState())
        print "Start's successors:", problem.getSuccessors(problem.getStartState())
        start = problem.getStartState()
        dict[start]=state(start,0,0,0)

        dfs_in(problem, start,0)

        print  dict.values()
        print dict["goalstate"]
        for item in dict.values():
            print item
        path = traceBack( dict["goalstate"])
        path.reverse()
        print path
        return path

def traceBack(goal):
    path=[]
    if(dict.has_key(goal)):
        while (dict[goal].getParents()!= 0):
            path.append(getPath(dict[goal].getAction()))
            parents = dict[goal].getParents()
            goal=parents
    return path
from game import Directions

def getPath(path):
        if path=='South' :
            return Directions.SOUTH
        elif path == 'West':
            return Directions.WEST
        elif path == 'North':
            return Directions.NORTH
        elif path == 'East':
            return Directions.EAST

def dfs_in(problem,parents,cost):

        for children in problem.getSuccessors(parents):

            if problem.isGoalState(children[0]) == False:

                if dict.has_key(children[0]):
                    if(dict[children[0]].getCost()>cost+1):
                        dict[children[0]].setCost(cost+1)
                        dict[children[0]].setParents(parents)
                        dict[children[0]].setAction(children[1])
                        if dfs_in(problem, children[0], cost + 1):
                            return True

                else:
                    childrenState=state(children[0],cost+1,children[1],parents)
                    dict[children[0]] = childrenState;
                    if dfs_in(problem,children[0],cost+1):
                        return True
            else:
                    childrenState = state(children[0], cost + 1, children[1], parents)
                    dict[children[0]] = childrenState;
                    dict["goalstate"]= children[0]
                    return True

# def dfs_in(problem,parents):
#
#     cost=dict[parents][0]+1
#     for children in problem.getSuccessors(parents):
#         child=children[0]
#         if (problem.isGoalState(child) == False):
#             if  dict.has_key(child):
#                 if(dict[child][0]>cost):
#                     dict[child][0] = cost
#                     dict[child][1] = parents
#                     dfs_in(problem,child)
#             else:
#                 dict[child][0]=cost
#                 dict[child][1] = parents
#                 dfs_in(problem, child)
#         else:
#             goal=child
#             return True

# def depthFirstSearch(problem):
#     from game import Directions
#
#     print "Start:", problem.getStartState()
#
#     print "Is the start a goal?", problem.isGoalState(problem.getStartState())
#     print "Start's successors:", problem.getSuccessors(problem.getStartState())
#
#     start=problem.getStartState()
#     nodelist.append(start)
#
#     dpf(problem, start)
#
#     list=[];
#     for tup in path:
#         if tup=='South' :
#             list.append(Directions.SOUTH)
#         elif tup == 'West':
#             list.append(Directions.WEST)
#         elif tup == 'North':
#             list.append(Directions.NORTH)
#         elif tup == 'East':
#             list.append(Directions.EAST)
#     print list
#     return list
#
# def dpf(problem,start):
#         for itemChild in problem.getSuccessors(start):
#                 if(itemChild[0] not in nodelist):
#                     nodelist.append(itemChild[0])
#                     path.append(itemChild[1])
#                     if (problem.isGoalState(itemChild[0]) == False):
#                            if dpf(problem,itemChild[0]):
#                                return True
#                            else:
#                                 path.pop()
#                     else:
#                         return True
#         return False

def breadthFirstSearch(problem):
    import util
    queue= util.Queue
    visited = util.Queue
    start = problem.getStartState()
    queue.push(start)

    while (queue.isEmpty()==False):
        children = problem.getSuccessors(start)
        for item in children:
            queue.push((item, start))
            if (problem.isGoalState(item) == False):
                visited.push((item[0],start))
            else:
                print item;
                break;


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
