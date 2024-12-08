import numpy as np
import math
import heapq
import random

class Node:
    def __init__(self, state, parent, action):
        self.state = tuple(state)
        self.parent = parent
        self.action = action
        self.path_cost = 0

    def __lt__(self, other):
        return self.path_cost < other.path_cost

class PriorityQueue:
    def __init__(self):
        self.elements = []
    def empty(self):
        return len(self.elements) == 0
    def add(self, item, priority):
        heapq.heappush(self.elements,(priority,item))
    def remove(self):
        return heapq.heappop(self.elements)[1]

class StackFrontier:
    def __init__(self):
        self.frontier = []
    def add(self, node):
        self.frontier.append(node)
    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)
    def empty(self):
        return len(self.frontier) == 0
    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            return self.frontier.pop()

class QueueFrontier(StackFrontier):
    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node

# Heuristic functions
class Heuristic:
    def custom_heuristic(current, goal, map):
        # Combination of Manhattan and penalty for terrain
        distance = Heuristic.manhattan_heuristic(current, goal) 

        penalty = 0  
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for direction in directions:
            neighbors = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbors[0] < len(map) and 0 <= neighbors[1] < len(map[0]):
                if map[neighbors[0]][neighbors[1]] == -1:
                    penalty += 1

        return distance + penalty * 2
    
    def euclidean_heuristic(current, goal):
        return np.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)
    
    def manhattan_heuristic(current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

class Search:
    def __init__(self, map, start, goal):
        self.map = map
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.frontier = None
        self.explored = set()
        self.visit_count = np.zeros_like(map)

    def is_valid_state(self, state):
        y, x = state
        if 0 <= y < len(self.map) and 0 <= x < len(self.map[0]):
            return self.map[y][x] != -1  # Assuming -1 represents an obstacle
        return False
    
    def get_neighbors(self, node):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for direction in directions:
            neighbor_state = (node.state[0] + direction[0], node.state[1] + direction[1])
            if self.is_valid_state(neighbor_state):
                neighbor = Node(neighbor_state, node, direction)
                neighbor.path_cost = node.path_cost + 1  # Assuming uniform cost
                neighbors.append(neighbor)
        return neighbors

    def reconstruct_path(self, node):
        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        path.reverse()
        return path

    def random_search(self):
        self.explored = set()
        self.frontier = StackFrontier()
        self.frontier.add(Node(self.start, None, None))  # Add start node
        while not self.frontier.empty():
            node = self.frontier.remove()
            self.visit_count[node.state[0], node.state[1]] += 1  # Increment visit count
            if node.state == self.goal:
                print(f"Number of explored nodes random: {len(self.explored)}")
                return self.reconstruct_path(node)
            self.explored.add(node.state)
            neighbors = self.get_neighbors(node)
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if not self.frontier.contains_state(neighbor.state) and neighbor.state not in self.explored:
                    self.frontier.add(neighbor)
        print(f"Number of explored nodes random: {len(self.explored)}")
        return None

    def breadth_first_search(self):
        self.explored = set()
        self.frontier = QueueFrontier()
        self.frontier.add(Node(self.start, None, None))  # Add start node
        while not self.frontier.empty():
            node = self.frontier.remove()
            self.visit_count[node.state[0], node.state[1]] += 1  # Increment visit count
            if node.state == self.goal:
                print(f"Number of explored cells BFS: {len(self.explored)}")
                return self.reconstruct_path(node)

            self.explored.add(node.state)
            for neighbor in self.get_neighbors(node):
                if not self.frontier.contains_state(neighbor.state) and neighbor.state not in self.explored:
                    self.frontier.add(neighbor)
        print(f"Number of explored cells BFS: {len(self.explored)}")
        return None

    def depth_first_search(self):
        self.explored = set()
        self.frontier = StackFrontier()
        self.frontier.add(Node(self.start, None, None))  # Add start node to the frontier
        while not self.frontier.empty():
            node = self.frontier.remove()
            self.visit_count[node.state[0], node.state[1]] += 1  # Increment visit count
            if node.state == self.goal:
                print(f"Number of explored cells DFS: {len(self.explored)}")
                return self.reconstruct_path(node)

            self.explored.add(node.state)
            for neighbor in self.get_neighbors(node):
                if not self.frontier.contains_state(neighbor.state) and neighbor.state not in self.explored:
                    self.frontier.add(neighbor)
        print(f"Number of explored cells DFS: {len(self.explored)}")
        return None

    def a_star_search(self, heuristic_func):
        self.explored = set()
        self.frontier = PriorityQueue()
        start_node = Node(self.start, None, None)
        self.frontier.add(start_node, 0)
        came_from = {}
        cost_so_far = {}
        came_from[self.start] = None
        cost_so_far[self.start] = 0

        while not self.frontier.empty():
            current = self.frontier.remove()

            if current.state == self.goal:
                print(f"Number of explored nodes A*: {len(self.explored)}")
                return self.reconstruct_path(current)

            self.visit_count[current.state[0], current.state[1]] += 1  # Increment visit count
            self.explored.add(current.state)  # Add to explored set
            for neighbor in self.get_neighbors(current):
                new_cost = cost_so_far[current.state] + 1  # Assuming uniform cost
                if neighbor.state not in cost_so_far or new_cost < cost_so_far[neighbor.state]:
                    cost_so_far[neighbor.state] = new_cost
                    if heuristic_func == Heuristic.custom_heuristic:
                        priority = new_cost + heuristic_func(neighbor.state, self.goal, self.map)
                    else:
                        priority = new_cost + heuristic_func(neighbor.state, self.goal)
                    self.frontier.add(neighbor, priority)
                    came_from[neighbor.state] = current.state

        print(f"Number of explored cells A*: {len(self.explored)}")
        return None
    
    def greedy_search(self, heuristic_func):
        self.explored = set()
        self.frontier = PriorityQueue()
        start_node = Node(self.start, None, None)

        self.frontier.add(start_node, 0)
        came_from = {}
        came_from[self.start] = None

        while not self.frontier.empty():
            current = self.frontier.remove()

            if current.state == self.goal:
                print(f"Number of explored cells greedy: {len(self.explored)}")
                return self.reconstruct_path(current)
                print(f"Number of explored cells greedy: {len(self.explored)}")

            self.visit_count[current.state[0], current.state[1]] += 1  # Increment visit count
            self.explored.add(current.state)  # Add to explored set
            for neighbor in self.get_neighbors(current):
                if neighbor.state not in came_from:
                    priority = heuristic_func(neighbor.state, self.goal)
                    self.frontier.add(neighbor, priority)
                    came_from[neighbor.state] = current.state

        print(f"Number of explored cells greedy: {len(self.explored)}")
        return None
