import numpy as np

from collections import deque
import heapq

def generate_map(x_size, y_size, th, start_x, start_y, end_x, end_y):
  map = np.random.rand(x_size, y_size)
  map[map < th] = 0
  map[map >= th] = 1
  map[start_x, start_y] = -1
  map[end_x, end_y] = -1

  return map


def get_neighbors(map, x, y):
    neighbors = []
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < map.shape[0] and 0 <= ny < map.shape[1] and map[nx][ny] != 1:
            neighbors.append((nx, ny))
    return neighbors

def get_path_neighbors(map, x, y):
    neighbors = []
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < map.shape[0] and 0 <= ny < map.shape[1] and map[nx][ny] == 1:
            neighbors.append((nx, ny))
    return neighbors

def a_star(map, start_x, start_y, end_x, end_y):
    """Run A* algorithm to find the path from start to end."""
    start = (start_x, start_y)
    end = (end_x, end_y)

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    open_list = []
    heapq.heappush(open_list, (0, start))
    g_costs = {start: 0}
    parent = {start: None}

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == end:
            break
        for neighbor in get_neighbors(map, *current):
            tentative_g_cost = g_costs[current] + 1
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic(neighbor, end)
                heapq.heappush(open_list, (f_cost, neighbor))
                parent[neighbor] = current

    # Reconstruct the path
    path = []
    node = end
    while node != start:
        path.append(node)
        node = parent[node]
    path.reverse()
    return [(start_x, start_y)] + path

def check_disconnected_points(path_map):
    """Find all disconnected points in the path (points with only one neighbor)."""
    
    rows, cols, _ = path_map.shape  # Now this should always work
    count = 0

    for r in range(rows):
        for c in range(cols):
            if path_map[r][c] == 1:  # If it's part of the path
                neighbors = get_path_neighbors(path_map, r, c)
                if len(neighbors) == 1:  # Point with only one neighbor
                    count += 1
    
    if count > 2:
        return True
    return False

def get_path(map, model, start_x, start_y, end_x, end_y):
    path = a_star(map, start_x, start_y, end_x, end_y)

    return path

def get_path_map(map, model, start_x, start_y, end_x, end_y):
    path = a_star(map, start_x, start_y, end_x, end_y)

    new_map = map.copy()
    for x, y in path:
        new_map[x, y] = -1

    return new_map