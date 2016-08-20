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
class node:
    # action=[]
    def __init__(self,state,cost,action,parent,h=0):
        self.state = state
        self.cost = cost
        self.action = action
        self.parent = parent
        self.h=h

    def getState(self):
        return  self.state

    #heuristic
    def getH(self):
        return self.h

    def setH(self,h):
        self.h = h

    def getCost(self):
        return  self.cost

    def setCost(self,cost):
        self.cost=cost

    def getParent(self):
        return self.parent

    def setParent(self, parent):
        self.parent = parent

    def setAction(self,action):
        self.action=action

    def getAction(self):
        return self.action

    def __str__(self):
        return "S:%s C:%s A:%s P:%s" %(self.state,self.cost,self.action,self.parent)

def depthFirstSearch(problem):

        print "Start:", problem.getStartState()
        print "Is the start a goal?", problem.isGoalState(problem.getStartState())
        print "Start's successors:", problem.getSuccessors(problem.getStartState())
        start = problem.getStartState()
        dict[start] = node(start, 0, [], None)

        dfs_in(problem, start,0)

        return  dict["goal"].getAction()


def traceBack(goal):
    path=[]
    if(dict.has_key(goal)):
        while (dict[goal].getParent()!= None):
            path.append(dict[goal].getAction())
            parent = dict[goal].getParent()
            goal=parent
    path.reverse()
    return path

def dfs_in(problem,parent,cost):

        import copy
        for children in problem.getSuccessors(parent):

            parentState = dict[parent]
            actionlist = copy.copy(parentState.getAction())
            actionlist.append(children[1])

            if problem.isGoalState(children[0]) == False:

                if dict.has_key(children[0]):
                    if(dict[children[0]].getCost()>cost+1):

                        dict[children[0]].setCost(cost+1)
                        dict[children[0]].setParent(parent)
                        dict[children[0]].setAction(actionlist)
                        dfs_in(problem, children[0], cost + 1)

                else:
                    childrenState=node(children[0],cost+1,actionlist,parent)
                    dict[children[0]] = childrenState;
                    dfs_in(problem, children[0], cost + 1)

            else:
                    childrenState = node(children[0], cost + 1, actionlist, parent)
                    dict["goal"] = childrenState;
                    return


def breadthFirstSearch(problem):

     # node tuple containing the state and it's parent
    start_state = problem.getStartState()
    dict[start_state] = node(start_state,0,None,None)

    # node queue contains nodes of type state (actually a node), see class on line ~80
    node_queue = util.Queue()
    node_queue.push(dict[start_state])

    while not node_queue.isEmpty():
        # take the oldest node off the queue and check if it's a goal
        next_node = node_queue.pop()
        if problem.isGoalState(next_node.getState()):
            return traceBack(next_node.getState())

        else:
            # Find this nodes successor states and see if they are either 
            # new states or cheaper nodes
            successors = problem.getSuccessors(next_node.getState())
            for suc in successors:
                if dict.has_key(suc[0]):
                    old_node = dict[suc[0]]
                    if next_node.getCost()+1 < old_node.getCost():

                        new_node = node(suc[0], next_node.getCost() + 1, suc[1], next_node.getState())
                        node_queue.push(new_node)
                        dict[suc[0]] =new_node

                    else:
                        continue
                else:
                    new_node = node(suc[0], next_node.getCost() + 1, suc[1], next_node.getState())
                    node_queue.push(new_node)
                    dict[suc[0]]=new_node

    # raise Error("Path no found")

def uniformCostSearch(problem):

    from util import PriorityQueue
    import copy

    queue = PriorityQueue()
    start = problem.getStartState()
    dict[start] = node(start, 0, [], None)

    queue.push(start,0)

    while(not queue.isEmpty()):

        parent=queue.pop()
        parentState=dict[parent]

        for children in problem.getSuccessors(parent):
            print children
            actionlist = copy.copy(parentState.getAction())
            actionlist.append(children[1])
            newcost = problem.getCostOfActions(actionlist)

            if problem.isGoalState(children[0]) == False:

                if dict.has_key(children[0]):
                        if dict[children[0]].getCost()>newcost:

                            dict[children[0]].setCost(newcost)
                            dict[children[0]].setParent(parent)
                            dict[children[0]].setAction(actionlist)
                            queue.update(children[0],newcost)
                else:
                    childrenState = node(children[0], newcost , actionlist,  parent)
                    dict[children[0]]=childrenState
                    queue.push(children[0],newcost)
            else:
                return actionlist


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    import searchAgents
    return searchAgents.manhattanHeuristic(state, problem)


def aStarSearch(problem, heuristic=nullHeuristic):

    from util import PriorityQueue
    import copy

    start = problem.getStartState()
    queue = PriorityQueue()
    dict[start] = node(start, 0, [], None)
    queue.push(start, heuristic(start,problem))

    while (not queue.isEmpty()):

        parent = queue.pop()
        parentState = dict[parent]

        for children in problem.getSuccessors(parent):

            actionlist = copy.copy(parentState.getAction())
            actionlist.append(children[1])

            if problem.isGoalState(children[0]) == False:

                if dict.has_key(children[0]):
                    if dict[children[0]].getCost() > parentState.getCost()+1:
                        dict[children[0]].setCost(parentState.getCost()+1)
                        dict[children[0]].setParent(parent)
                        dict[children[0]].setAction(actionlist)
                        queue.update(children[0], parentState.getCost()+1+dict[children[0]].getH())

                else:
                    h = nullHeuristic(children[0], problem)

                    childrenState = node(children[0],parentState.getCost()+1, actionlist, parent,h)
                    dict[children[0]] = childrenState
                    queue.push(children[0], h+parentState.getCost()+1)
            else:
                return actionlist

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
