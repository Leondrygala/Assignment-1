# EXTENDED BY LEON FRIEDRICH DRYGALA AND JIA WANG
# FOR University of Melbourne COMP90047 Assignment 1
# Further use or even knowledge of this file may be considered 
# academic misconduct by your university. Be very careful.

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
    vstd_nodes = {}
    start_state = problem.getStartState()
    vstd_nodes[start_state] = node(start_state, 0, [], None)
    queue.push(start_state,0)

    while not queue.isEmpty():

        parent=queue.pop()
        parent_node=vstd_nodes[parent]

        for child in problem.getSuccessors(parent):
            actionlist = copy.copy(parent_node.getAction())
            actionlist.append(child[1])
            child_cost = parent_node.getCost() + child[2]

            if problem.isGoalState(child[0]) == False:

                if vstd_nodes.has_key(child[0]):
                    old_node = vstd_nodes[child[0]]
                    if old_node.getCost()>child_cost:
                        old_node.setCost(child_cost)
                        old_node.setParent(parent)
                        old_node.setAction(actionlist)
                        queue.update(child[0],child_cost)
                else:
                    child_node = node(child[0], child_cost , actionlist,  parent)
                    vstd_nodes[child[0]]=child_node
                    queue.push(child[0],child_cost)
            else:
                return actionlist
    print "uniformCostSearch: No path found!"
    return []


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
    vstd_nodes = {}
    vstd_nodes[start] = node(start, 0, [], None)
    queue.push(start, heuristic(start,problem))
 
    while (not queue.isEmpty()):
        
        parent_item = queue.pop2()
        # parent = queue.pop()
        parent_state = parent_item[2]
        parent_f = parent_item[0]

        parent_node = vstd_nodes[parent_state]

        for child in problem.getSuccessors(parent_state):

            actionlist = copy.copy(parent_node.getAction())
            actionlist.append(child[1])

            if problem.isGoalState(child[0]) == False:
                #if this state already has a ndoe associated with it
                if vstd_nodes.has_key(child[0]):
                    old_node = vstd_nodes[child[0]]
                    child_cost = parent_node.getCost()+child[2]
                    if old_node.getCost() > child_cost:
                        old_node.setCost(child_cost)
                        old_node.setParent(parent_state)
                        old_node.setAction(actionlist)
                        queue.update(child[0], child_cost+old_node.getH())
                        # print "aStarSearch: UPDATED", child[0][0]
                        # if not old_node.getH() == heuristic(child[0], problem):
                        #     print "aStarSearch: old and new h: ", old_node.getH(), ", ", heuristic(child[0], problem)

                else:
                    h = heuristic(child[0], problem)
                    childrenState = node(child[0],parent_node.getCost()+child[2], actionlist, parent_state,h)
                    vstd_nodes[child[0]] = childrenState
                    f = h + parent_node.getCost()+child[2]
                    queue.push(child[0], f)

                    if (parent_f - f) > 1:
                    # if child[0][0] == (18,4) and f != 20:
                        print "aStarSearch: parent: ", parent_state[0], " child: ", child[0][0]
                        print "aStarSearch: parent_f: ", parent_f, " child f:", f
                        print "aStarSearch: parent_g: ", parent_node.getCost(), " child step cost:", child[2]
                        print "aStarSearch: parent_h: ", parent_node.getH(), " child_h:", h
                        print "aStarSearch: parent goals:", len(parent_state[1].asList()), "child_goals: ", len(child[0][1].asList())
                        
            else:
                return actionlist
    print "aStarSearch: No path found!"
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
