import numpy as np
from robot import robot
from map_builder import get_path, get_path_map

class manager:
    def __init__(self, robots, map):
        self.robots = robots
        self.map = map

        #self.maps = segment_map(map)

    def update_robots(self):
        for robot in self.robots:
            robot.update()
    
    # CONSTRUCTION ---------------------------------------------------------

    def get_house_positions(self, map, house_x, house_y, num_parts, index):
        """
        Generate the perimeter (outside blocks) of the house, and split them into equal parts.
        
        Parameters:
            map (numpy.ndarray): The 2D array representing the map.
            house_x (int): Width of the house (number of columns).
            house_y (int): Height of the house (number of rows).
            num_parts (int): The number of parts to split the house perimeter into.
            index (int): The index of the part to return.
            
        Returns:
            list: A list of tuples representing the part of the house perimeter.
        """
        # Get the dimensions of the map
        n, m = map.shape

        # Calculate the center of the map
        center_x, center_y = m // 2, n // 2
        
        # Calculate the top-left corner of the house
        start_x = center_x - house_x // 2
        start_y = center_y - house_y // 2
        
        # Generate the perimeter of the house (outer boundary)
        house_positions = []

        # Top edge: Going from left to right, one unit above the house
        for x in range(start_x - 1, start_x + house_x + 1):
            if 0 <= x < m and start_y - 1 >= 0:  # Ensure we stay within map bounds
                house_positions.append((start_y - 1, x))

        # Right edge: Going from top to bottom, one unit to the right of the house
        for y in range(start_y - 1, start_y + house_y):
            if 0 <= y < n and start_x + house_x < m:  # Ensure we stay within map bounds
                house_positions.append((y, start_x + house_x))

        # Bottom edge: Going from right to left, one unit below the house
        for x in range(start_x + house_x, start_x - 1, -1):
            if 0 <= x < m and start_y + house_y < n:  # Ensure we stay within map bounds
                house_positions.append((start_y + house_y, x))

        # Left edge: Going from bottom to top, one unit to the left of the house
        for y in range(start_y + house_y, start_y - 1, -1):
            if 0 <= y < n and start_x - 1 >= 0:  # Ensure we stay within map bounds
                house_positions.append((y, start_x - 1))

        # Now split the house positions into num_parts
        total_positions = len(house_positions)
        if num_parts <= 0:
            raise ValueError("num_parts must be greater than 0.")
        if index < 0 or index >= num_parts:
            raise ValueError(f"index must be between 0 and {num_parts-1} (inclusive).")
        
        # Calculate the number of positions per part
        part_size = total_positions // num_parts
        remaining = total_positions % num_parts  # How many positions to distribute
        
        # Calculate the start and end for the current part
        start_index = index * part_size + min(index, remaining)
        end_index = start_index + part_size + (1 if index < remaining else 0)
        
        # Return the current part of the house perimeter
        return house_positions[start_index:end_index]


    def get_house_path(self, map, house_x, house_y, num_parts, index):
        """
        Generate a path around the house (with one unit distance outside the house),
        ensuring the path smoothly transitions around corners, and split these path indexes 
        into num_parts equal parts. Return the part corresponding to the given index.
        
        Parameters:
            map (numpy.ndarray): The 2D array representing the map.
            house_x (int): Width of the house (number of columns).
            house_y (int): Height of the house (number of rows).
            num_parts (int): The number of parts to split the path into.
            index (int): The index of the part to return.
            
        Returns:
            list: A list of tuples representing the part of the construction path.
        """
        
        # Get the dimensions of the map
        n, m = map.shape

        # Calculate the center of the map
        center_x, center_y = m // 2, n // 2
        
        # Calculate the top-left corner of the house
        start_x = center_x - house_x // 2
        start_y = center_y - house_y // 2
        
        # Calculate the bottom-right corner of the house
        end_x = start_x + house_x
        end_y = start_y + house_y
        
        # Generate the path around the house, offset by one unit outside the house
        path_outside = []

        # Top edge: Going from left to right, one unit above the house (outward)
        for x in range(start_x - 1, end_x + 1):
            if 0 <= x < m and start_y - 2 >= 0:  # Check bounds
                path_outside.append((start_y - 2, x))

        # Smooth transition at top-right corner (Move down to the right edge)
        if end_x + 1 < m and start_y - 2 >= 0:
            path_outside.append((start_y - 1, end_x + 1))  # Move down to the right edge

        # Right edge: Going from top to bottom, one unit to the right of the house
        for y in range(start_y - 1, end_y + 1):
            if 0 <= y < n and end_x + 1 < m:  # Check bounds
                path_outside.append((y, end_x + 1))

        # Smooth transition at bottom-right corner (Move left to the bottom edge)
        if end_y + 1 < n and end_x + 1 < m:
            path_outside.append((end_y + 1, end_x))  # Move left to the bottom edge

        # Bottom edge: Going from right to left, one unit below the house (outward)
        for x in range(end_x, start_x - 2, -1):
            if 0 <= x < m and end_y + 1 < n:  # Check bounds
                path_outside.append((end_y + 1, x))

        # Smooth transition at bottom-left corner (Move up to the left edge)
        if end_y + 1 < n and start_x - 2 >= 0:
            path_outside.append((end_y, start_x - 2))  # Move up to the left edge

        # Left edge: Going from bottom to top, one unit to the left of the house
        for y in range(end_y, start_y - 2, -1):
            if 0 <= y < n and start_x - 2 >= 0:  # Check bounds
                path_outside.append((y, start_x - 2))

        # Smooth transition at top-left corner (Move right to the top edge)
        if start_y - 2 >= 0 and start_x - 2 >= 0:
            path_outside.append((start_y - 1, start_x - 1))  # Move right to the top edge

        # Now split the full circular path into num_parts, ensuring it's connected
        total_indexes = len(path_outside)
        if num_parts <= 0:
            raise ValueError("num_parts must be greater than 0.")
        if index < 0 or index >= num_parts:
            raise ValueError(f"index must be between 0 and {num_parts-1} (inclusive).")
        
        # Calculate the number of indexes per part
        part_size = total_indexes // num_parts
        remaining = total_indexes % num_parts  # How many indexes to distribute
        
        # Calculate the start and end for the current part
        start_index = index * part_size + min(index, remaining)
        end_index = start_index + part_size + (1 if index < remaining else 0)
        
        # Return the current part of the circular path
        return path_outside[start_index:end_index]

    
    def are_neighbors(self, x1, y1, x2, y2):
        """
        Check if two points (x1, y1) and (x2, y2) are neighbors.
        Points are neighbors if they are adjacent in one of the four cardinal directions.

        Parameters:
            x1, y1 (int): Coordinates of the first point.
            x2, y2 (int): Coordinates of the second point.
            
        Returns:
            bool: True if the points are neighbors, False otherwise.
        """
        # Points are neighbors if they are adjacent in any of the four cardinal directions
        if (abs(x1 - x2) == 1 and y1 == y2) or (abs(y1 - y2) == 1 and x1 == x2):
            return True
        return False
    
    def get_construction_paths(self):
        for i in range(len(self.robots)):
            robot = self.robots[i]

            path_house = self.get_house_path(self.map, 20, 20, len(self.robots), i)
            position_house = self.get_house_positions(self.map, 20, 20, len(self.robots), i)
            
            dummy_map = self.map.copy()
            for pos in position_house:
                x, y = pos
                dummy_map[x, y] = 1

            paths = []
            for j in range(len(path_house) - 1):
                x_curr = path_house[j][0]
                y_curr = path_house[j][1]

                x_next = path_house[j + 1][0]
                y_next = path_house[j + 1][1]

                paths += get_path(dummy_map, None, x_curr, y_curr, x_next, y_next)

            path_to_house = get_path(dummy_map, None, robot.x, robot.y, path_house[0][0], path_house[0][1])

            path = path_to_house + paths
            #print(path)

            robot.path = path
            robot.curr_path_index = 0


    # DEMOLITION -----------------------------------------------------------

    def get_split_indexes(self, map, num_parts, index):
        """
        Compute the row index ranges for the index-th horizontal slice 
        of a 2D array map divided into num_parts equal horizontal slices.
        
        Parameters:
            map (numpy.ndarray): The 2D array to split.
            num_parts (int): Total number of parts to split the array into.
            index (int): The part index (0-based).
            
        Returns:
            tuple: A tuple ((row_start, row_end), (col_start, col_end)) representing
                the ranges for rows and columns in the specified split.
        """
        # Get dimensions of the array
        n, m = map.shape

        # Validate inputs
        if num_parts <= 0:
            raise ValueError("num_parts must be greater than 0.")
        if index < 0 or index >= num_parts:
            raise ValueError(f"index must be between 0 and {num_parts-1} (inclusive).")

        # Calculate the number of rows per slice
        rows_per_part = n // num_parts
        remaining_rows = n % num_parts  # Rows that need to be distributed

        # Calculate the row range for the current slice
        row_start = index * rows_per_part + min(index, remaining_rows)
        row_end = row_start + rows_per_part + (1 if index < remaining_rows else 0)

        # The column range is the same for all slices
        col_start = 0
        col_end = m

        return (row_start, row_end), (col_start, col_end)
    

    def get_demolition_paths(self):
        for i in range(len(self.robots)):
            robot = self.robots[i]

            x_indexes, y_indexes = self.get_split_indexes(self.map, len(self.robots), i)
            map_area = self.map[x_indexes[0] : x_indexes[1],
                                y_indexes[0] : y_indexes[1]]
            
            obs_indexes = np.argwhere(map_area == 1)
            obs_xs, obs_ys = obs_indexes[:, 0], obs_indexes[:, 1]
            obs_xs += x_indexes[0]
            obs_ys += y_indexes[0]

            old_obs_x = obs_xs[0]
            old_obs_y = obs_ys[0]

            dummy_map = self.map.copy()
            dummy_map[old_obs_x, old_obs_y] = 0

            path = []
            curr_path = get_path(dummy_map, None, robot.x, robot.y, obs_xs[0], obs_ys[0])
            path.extend(curr_path)
            
            for j in range(1, len(obs_xs)):
                obs_x = obs_xs[j]
                obs_y = obs_ys[j]

                dummy_map[old_obs_x, old_obs_y] = 0
                dummy_map[obs_x, obs_y] = 0
                curr_path = get_path(dummy_map, None, old_obs_x, old_obs_y, obs_xs[j], obs_ys[j])
                old_obs_x = obs_x
                old_obs_y = obs_y
                path.extend(curr_path)

            #print(path)
            robot.path = path
            robot.curr_path_index = 0
    
    # MAPPING -------------------------------------------------------------

    def calculate_map_from_hits(self, robot):
        map_gen = np.zeros_like(self.map)

        for i in range(len(robot.history)):
            hits = robot.history[i][0]

            if len(hits) == 0:
                continue

            obs_x, obs_y = hits[0]

            map_gen[obs_x, obs_y] = 1

        return map_gen

    def get_mapping_paths(self):
        for i in range(len(self.robots)):
            robot = self.robots[i]

            # Get the x and y indexes from your split function
            x_indexes, y_indexes = self.get_split_indexes(self.map, len(self.robots), i)

            # Unpack the start and end indexes for rows and columns from x_indexes and y_indexes
            row_start, row_end = x_indexes
            col_start, col_end = y_indexes

            # Initialize the path for the robot
            path = []

            # Traverse the rows within the slice range
            for row in range(row_start, row_end):
                # Zigzag within columns
                if row % 2 == 0:  # Even row index: left to right
                    path.extend([(row, col) for col in range(col_start, col_end)])
                else:  # Odd row index: right to left
                    path.extend([(row, col) for col in range(col_end - 1, col_start - 1, -1)])

            # Assign the computed path to the robot and reset the current path index
            robot.path = path
            robot.curr_path_index = 0

    def get_frames(self, path_method):
        path_method()
        prev_frame = np.zeros_like(self.map)

        house_positions = self.get_house_positions(self.map, 20, 20, 1, 0)

        map_gen = np.zeros_like(self.map)
        frames = []

        longest_path = 0
        for robot in self.robots:
            #print(robot)
            #print(robot.path)
            curr_length = len(robot.path)
            if curr_length > longest_path:
                longest_path = curr_length

        for _ in range(longest_path):
            frame = self.map.copy()
            for robot in self.robots:
                if robot.type == 'demolition':
                    if self.map[robot.x, robot.y] == 1:
                        self.map[robot.x, robot.y] = 0

                        frame = self.map.copy()

                if robot.type == 'exploration':
                    #frame = np.zeros_like(self.map)
                    frame[frame != -2] = 0
                    map_scan = self.calculate_map_from_hits(robot)
                    map_gen[map_scan == 1] = 1
                    frame[map_gen == 1] = 1

                if robot.type == 'construction':
                    frame = self.map.copy()

                    for pos in house_positions:
                        x, y = pos
                        if self.are_neighbors(x, y, robot.x, robot.y) == True:
                            self.map[x, y] = 1
                            break

                    for r in self.robots:
                        frame[r.x, r.y] = -2

                frame[robot.x, robot.y] = -2

                robot.update()


            if np.array_equal(prev_frame, frame):
                break

            prev_frame = frame

            self.update_robots()
            frames.append(frame)

        return np.array(frames)