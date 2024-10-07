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
from typing import List, Any
from collections import deque

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
        raise NotImplementedError()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        raise NotImplementedError()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        raise NotImplementedError()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        raise NotImplementedError()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: Any) -> List[str]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # Get the initial state of the problem
    startState = problem.getStartState()

    # Initialize a stack to store states to explore (fringe)
    fringe = util.Stack()

    # Initialize a set to keep track of visited states
    visited = set()

    # Push the start state and an empty action list to the fringe
    fringe.push((startState, []))

    # Continue searching while there are states in the fringe
    while not fringe.isEmpty():
        # Pop the current state and its corresponding actions from the fringe
        currentState, actions = fringe.pop()

        # Check if the current state has not been visited
        if currentState not in visited:
            # Mark the current state as visited
            visited.add(currentState)

            # Check if the current state is the goal state
            if problem.isGoalState(currentState):
                # If goal is reached, return the sequence of actions to get here
                return actions

            # Explore successors of the current state
            for state, action, cost in problem.getSuccessors(currentState):
                # If the successor state hasn't been visited
                if state not in visited:
                    # Push the successor state and updated action list to the fringe
                    fringe.push((state, actions + [action]))

    # If no solution is found, return an empty list
    return []

def breadthFirstSearch(problem: Any) -> List[str]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # Get the initial state of the problem
    startState = problem.getStartState()

    # Initialize a deque (double-ended queue) to store states to explore (fringe)
    # Each element is a tuple of (state, actions_to_reach_state)
    fringe = deque([(startState, [])])

    # Initialize a set to keep track of visited states
    visited = set()

    # Continue searching while there are states in the fringe
    while fringe:
        # Popleft (remove and return) the leftmost item from the fringe
        # This ensures FIFO (First In, First Out) behavior for BFS
        currentState, actions = fringe.popleft()

        # Check if the current state has not been visited
        if currentState not in visited:
            # Mark the current state as visited
            visited.add(currentState)

            # Check if the current state is the goal state
            if problem.isGoalState(currentState):
                # If goal is reached, return the sequence of actions to get here
                return actions

            # Explore successors of the current state
            for state, action, cost in problem.getSuccessors(currentState):
                # If the successor state hasn't been visited
                if state not in visited:
                    # Append the successor state and updated action list to the fringe
                    fringe.append((state, actions + [action]))

    # If no solution is found, return an empty list
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    startState = problem.getStartState()
    #ucs using priority queue to prioritize the successors with the least cost
    fringe = util.PriorityQueue()
    visited = []

    #the fringe apart from the state, action, cost also has priority 0 here
    #same as the cost as i want the least total cost first
    fringe.push((startState, [], 0), 0)

    #kept popping till no more nodes in the fringe
    while not fringe.isEmpty():
        currentState, actions, costs = fringe.pop()
        if not currentState in visited:
            #updated visited status
            visited.append(currentState)
            #if this goal state return the actions to reach it
            if problem.isGoalState(currentState):
                return actions
            #push all successors, not in visited
            for state, action, cost in problem.getSuccessors(currentState):
                if not state in visited:
                    #cost to reflect total cost and prioritize the least as the priority queue is implemented using heapq pops smallest element first and pushes to maintain the order
                    fringe.push((state, actions + [action], costs + cost), costs + cost)
    raise NotImplementedError()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    startState = problem.getStartState()
    #a* using priority queue as to prioritize the successors with the least heuristic cost
    fringe = util.PriorityQueue()
    visited = []

    #the fringe apart from the state, action cost has combined cost 0 here we want the least combined cost first
    fringe.push((startState, [], 0), 0)

    #kept popping till no more nodes in the fringe
    while not fringe.isEmpty():
        currentState, actions, costs = fringe.pop()
        if not currentState in visited:
            #update visited status
            visited.append(currentState)
            #if the goal state return the actions to reach it
            if problem.isGoalState(currentState):
                return actions
            #push all successors, not in visited
            for state, action, cost in problem.getSuccessors(currentState):
                if not state in visited:
                    #update cost to reflect combined path and heuristic cost and prioritize the least for popping as the priority queue is implemented using heapq pops smallest element first and pushes to maintain the order
                    heuristicCost = costs + cost + heuristic(state, problem)
                    fringe.push((state, actions + [action], costs + cost), heuristicCost)
    raise NotImplementedError()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
