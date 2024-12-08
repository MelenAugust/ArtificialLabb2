""" Functions for generating 2D grid maps, for AI Lab 2 - path planning.
"""
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from search_algorithm import Search, Node , Heuristic as HS

percentOfObstacle = 0.9  # 30% - 60%, random

def generateMap2d(size_):

    '''Generates a random 2d map with obstacles (small blocks) randomly distributed. 
       You can specify any size of this map but your solution has to be independent of map size

    Parameters:
    -----------
    size_ : list
        Width and height of the 2d grid map, e.g. [60, 60]. The height and width of the map shall be greater than 20.

    Returns:
    --------
        map2d : array-like, shape (size_[0], size_[1])
           A 2d grid map, cells with a value of 0: Free cell; 
                                                -1: Obstacle;
                                                -2: Start point;
                                                -3: Goal point;
    '''
    
    size_x, size_y = size_[0], size_[1]

    map2d = np.random.rand(size_y, size_x)
    perObstacles_ = percentOfObstacle
    map2d[map2d <= perObstacles_] = 0
    map2d[map2d > perObstacles_] = -1

    yloc, xloc = [np.random.randint(0, size_x-1, 2), np.random.randint(0, size_y-1, 2)]
    while (yloc[0] == yloc[1]) and (xloc[0] == xloc[1]):
        yloc, xloc = [np.random.randint(0, size_x-1,2), np.random.randint(0, size_y-1, 2)]

    map2d[xloc[0]][yloc[0]] = -2
    map2d[xloc[1]][yloc[1]] = -3

    return map2d

# Generate 2d grid map with rotated-H-shape object
def generateMap2d_obstacle(size_):
    '''Generates a random 2d map with a rotated-H-shape object in the middle and obstacles (small blocks) randomly distributed. 
       You can specify any size of this map but your solution has to be independent of map size

    Parameters:
    -----------
    size_ : list
        Width and height of the 2d grid map, e.g. [60, 60]. The height and width of the map shall be greater than 40.

    Returns:
    --------
        map2d : array-like, shape (size_[0], size_[1])
           A 2d grid map, cells with a value of 0: Free cell; 
                                               -1: Obstacle;
                                               -2: Start point;
                                               -3: Goal point;
                                            
       [ytop, ybot, minx] : list
           information of the rotated-H-shape object
           ytop - y coordinate of the top horizontal wall/part
           ybot - y coordinate of the bottom horizontal wall/part
           minx - X coordinate of the vertical wall 
    '''
    
    size_x, size_y = size_[0], size_[1]
    map2d = generateMap2d(size_)

    map2d[map2d==-2] = 0
    map2d[map2d==-3] = 0

    # add special obstacle
    xtop = [np.random.randint(5, 3*size_x//10-2), np.random.randint(7*size_x//10+3, size_x-5)]
    ytop = np.random.randint(7*size_y//10 + 3, size_y - 5)
    xbot = np.random.randint(3, 3*size_x//10-5), np.random.randint(7*size_x//10+3, size_x-5)
    ybot = np.random.randint(5, size_y//5 - 3)


    map2d[ybot, xbot[0]:xbot[1]+1] = -1
    map2d[ytop, xtop[0]:xtop[1]+1] = -1
    minx = (xbot[0]+xbot[1])//2
    maxx = (xtop[0]+xtop[1])//2
    if minx > maxx:
        tempx = minx
        minx = maxx
        maxx = tempx
    if maxx == minx:
        maxx = maxx+1

    map2d[ybot:ytop, minx:maxx] = -1
    startp = [np.random.randint(0, size_x//2 - 4), np.random.randint(ybot+1, ytop-1)]

    map2d[startp[1], startp[0]] = -2
    goalp = [np.random.randint(size_x//2 + 4, size_x - 3), np.random.randint(ybot+1, ytop-1)]

    map2d[goalp[1],goalp[0]] = -3
    #return map2d, [startp[1], startp[0]], [goalp[1], goalp[0]], [ytop, ybot]
    return map2d, [ytop, ybot, minx]


# helper function for plotting the result
def plotMap(map2d_, path_, title_ =''):
    
    '''Plots a map (image) of a 2d matrix with a path from start point to the goal point. 
        cells with a value of 0: Free cell; 
                             -1: Obstacle;
                             -2: Start point;
                             -3: Goal point;
    Parameters:
    -----------
    map2d_ : array-like
        an array with Real Numbers
        
    path_ : array-like
        an array of 2d corrdinates (of the path) in the format of [[x0, y0], [x1, y1], [x2, y2], ..., [x_end, y_end]]
        
    title_ : string
        information/description of the plot

    Returns:
    --------

    '''
    
    import matplotlib.cm as cm
    plt.interactive(False)
    
    colors_nn = int(map2d_.max())
    colors = cm.winter(np.linspace(0, 1, colors_nn))

    colorsMap2d = [[[] for x in range(map2d_.shape[1])] for y in range(map2d_.shape[0])]
    # Assign RGB Val for starting point and ending point
    locStart, locEnd = np.where(map2d_ == -2), np.where(map2d_ == -3)
    
    colorsMap2d[locStart[0][0]][locStart[1][0]] = [.0, .0, .0, 1.0]  # black
    colorsMap2d[locEnd[0][0]][locEnd[1][0]] = [.0, .0, .0, .0]  # white

    # Assign RGB Val for obstacle
    locObstacle = np.where(map2d_ == -1)
    for iposObstacle in range(len(locObstacle[0])):
        colorsMap2d[locObstacle[0][iposObstacle]][locObstacle[1][iposObstacle]] = [1.0, .0, .0, 1.0]
    # Assign 0
    locZero = np.where(map2d_ == 0)

    for iposZero in range(len(locZero[0])):
        colorsMap2d[locZero[0][iposZero]][locZero[1][iposZero]] = [1.0, 1.0, 1.0, 1.0]

    # Assign Expanded nodes
    locExpand = np.where(map2d_>0)

    for iposExpand in range(len(locExpand[0])):
        _idx_ = int(map2d_[locExpand[0][iposExpand]][locExpand[1][iposExpand]]-1)
        colorsMap2d[locExpand[0][iposExpand]][locExpand[1][iposExpand]] = colors[_idx_]

    for irow in range(len(colorsMap2d)):
        for icol in range(len(colorsMap2d[irow])):
            if colorsMap2d[irow][icol] == []:
                colorsMap2d[irow][icol] = [1.0, 0.0, 0.0, 1.0]

    plt.figure()
    plt.title(title_)
    plt.imshow(colorsMap2d, interpolation='nearest')
    plt.colorbar()
    
    if path_ is not None:
        # Ensure path_ is a NumPy array
        path_ = np.array(path_)
        #print("Original path:", path_)
        path = path_.tolist()  # Convert to list of coordinate pairs
        #print("Formatted path:", path)
        x_coords, y_coords = zip(*path)
        plt.plot(y_coords, x_coords, color='magenta', linewidth=2.5)  # Correct indexing

    plt.show()

# helper function for plotting a heat map of the explored nodes
def plotHeatMap(map2d_, path_, visit_count, title_=''):
    import matplotlib.cm as cm
    plt.interactive(False)
    
    colors_nn = int(map2d_.max())
    colors = cm.winter(np.linspace(0, 1, colors_nn))

    colorsMap2d = [[[] for x in range(map2d_.shape[1])] for y in range(map2d_.shape[0])]
    # Assign RGB Val for starting point and ending point
    locStart, locEnd = np.where(map2d_ == -2), np.where(map2d_ == -3)
    
    colorsMap2d[locStart[0][0]][locStart[1][0]] = [.0, .0, .0, 1.0]  # black
    colorsMap2d[locEnd[0][0]][locEnd[1][0]] = [.0, .0, .0, .0]  # white

    # Assign RGB Val for obstacle
    locObstacle = np.where(map2d_ == -1)
    for iposObstacle in range(len(locObstacle[0])):
        colorsMap2d[locObstacle[0][iposObstacle]][locObstacle[1][iposObstacle]] = [1.0, .0, .0, 1.0]
    # Assign 0
    locZero = np.where(map2d_ == 0)

    for iposZero in range(len(locZero[0])):
        colorsMap2d[locZero[0][iposZero]][locZero[1][iposZero]] = [1.0, 1.0, 1.0, 1.0]

    # Assign Expanded nodes
    locExpand = np.where(map2d_ > 0)
    for iposExpand in range(len(locExpand[0])):
        _idx_ = int(map2d_[locExpand[0][iposExpand]][locExpand[1][iposExpand]] - 1)
        colorsMap2d[locExpand[0][iposExpand]][locExpand[1][iposExpand]] = colors[_idx_]

    for irow in range(len(colorsMap2d)):
        for icol in range(len(colorsMap2d[irow])):
            if colorsMap2d[irow][icol] == []:
                colorsMap2d[irow][icol] = [1.0, 0.0, 0.0, 1.0]

    plt.figure()
    plt.title(title_)
    plt.imshow(colorsMap2d, interpolation='nearest')

    # Overlay the visit_count heat map
    masked_visit_count = np.ma.masked_where(visit_count == 0, visit_count)
    plt.imshow(masked_visit_count, cmap='hot', interpolation='nearest', alpha=0.6)

    plt.colorbar()

    if path_ is not None:
        # Ensure path_ is a NumPy array
        path_ = np.array(path_)
        #print("Original path:", path_)
        path = path_.tolist() 
        #print("Formatted path:", path)
        x_coords, y_coords = zip(*path)
        plt.plot(y_coords, x_coords, color='magenta', linewidth=2.5)  # Correct indexing

    plt.show()

# helper function to find the start and goal points in the map
def find_coord(map2d_, start, goal):
    for y in range(len(map2d_)):
        for x in range(len(map2d_[y])):
            if map2d_[y][x] == -2:
                start = (y, x)
            elif map2d_[y][x] == -3:
                goal = (y, x)

    if start is None or goal is None:
        raise ValueError("Start or goal point not found in the map")

    return start, goal

#uncomment the following main function to run the simulation.py file to generate
#100 iterations of the path planning problem and save the results to a CSV file
'''
def main():
    start = None
    goal = None
    
    _map_ = generateMap2d([60, 60])
    map_h_object, info = generateMap2d_obstacle([60, 60])

    maps = [_map_, map_h_object]

    algorithms = {
        "A* (Custom)": lambda: Search(map2d, start, goal).a_star_search(HS.custom_heuristic),
        "A* (Euclidean)": lambda: Search(map2d, start, goal).a_star_search(HS.euclidean_heuristic),
        "A* (Manhattan)": lambda: Search(map2d, start, goal).a_star_search(HS.manhattan_heuristic),
        "Greedy (Euclidean)": lambda: Search(map2d, start, goal).greedy_search(HS.euclidean_heuristic),
        "Greedy (Manhattan)": lambda: Search(map2d, start, goal).greedy_search(HS.manhattan_heuristic),
        "Random Search": lambda: Search(map2d, start, goal).random_search(),
        "BFS": lambda: Search(map2d, start, goal).breadth_first_search(),
        "DFS": lambda: Search(map2d, start, goal).depth_first_search(),
    }

    results = []

    for map2d in maps:
        start, goal = find_coord(map2d, start, goal)
        for name, func in algorithms.items():
            search_instance = Search(map2d, start, goal)
            time_start = time.time()
            path = func()
            time_end = time.time()
            duration = time_end - time_start
            result = {
                "algorithm": name,
                "time": duration,
                "path_length": len(path) if path else 0,
                "explored_nodes": len(search_instance.explored),
            }
            results.append(result)
    return results
'''

#uncomment the following code to run in the terminal and produce images of the maze

# create a map with obstacles randomly distributed
#  0 - Free cell
# -1 - Obstacle
# -2 - Start point
# -3 - Goal point

_map_ = generateMap2d([60, 60])
plt.clf()
plt.imshow(_map_)
plt.show()

# map with rotated-H shape obstacle and obstacles randomly distributed
map_h_object, info = generateMap2d_obstacle([60, 60])

# environment information
print("map info: ")
print("y top: ", info[0])
print("t bot: ", info[1])
print("x wall: ", info[2])

plt.clf()
plt.imshow(map_h_object)
plt.show()

#define Start and Goal Points
start = None
goal = None

maps = [_map_, map_h_object]

algorithms = {
    "A* (Custom)": lambda: Search(map2d, start, goal).a_star_search(HS.custom_heuristic),
    "A* (Euclidean)": lambda: Search(map2d, start, goal).a_star_search(HS.euclidean_heuristic),
    "A* (Manhattan)": lambda: Search(map2d, start, goal).a_star_search(HS.manhattan_heuristic),
    "Greedy (Euclidean)": lambda: Search(map2d, start, goal).greedy_search(HS.euclidean_heuristic),
    "Greedy (Manhattan)": lambda: Search(map2d, start, goal).greedy_search(HS.manhattan_heuristic),
    "Random Search": lambda: Search(map2d, start, goal).random_search(),
    "BFS": lambda: Search(map2d, start, goal).breadth_first_search(),
    "DFS": lambda: Search(map2d, start, goal).depth_first_search(),
}

for map2d in maps:
    start, goal = find_coord(map2d, start, goal)
    for name, func in algorithms.items():
        search_instance = Search(map2d, start, goal)
        time_start = time.time()
        path = func()
        time_end = time.time()
        duration = time_end - time_start
        print(f"{name} took {duration:.4f} seconds")
        if path:
            print(f"Path length for {name}: {len(path)}\n")
        plotMap(map2d, path, title_=f'Path {name}')
        #print(f"Number of explored nodes in total {name}: {len(search_instance.explored)}")
        #plotHeatMap(map2d, path, search_instance.visit_count, title_=f'Explored Paths Heat Map {name}')
