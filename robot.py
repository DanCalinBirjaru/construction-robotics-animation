import numpy as np
from sensor import sensor

class robot:
    def __init__(self, id, type, path, real_map):
        self.id = id

        self.type = type

        self.path = path

        self.curr_path_index = 0

        self.x = self.path[self.curr_path_index][0]
        self.y = self.path[self.curr_path_index][1]

        self.sensor = sensor(real_map, self.x, self.y)
        
        hits = self.sensor.cast_rays(num_rays = 100, max_dist = 3)
        if hits == None:
            self.history = [(None, self.x, self.y)]
        else:
            self.history = [(hits, self.x, self.y)]

        self.done = False
    
    def update(self):
        if self.curr_path_index == len(self.path) - 1:
            self.curr_path_index = self.curr_path_index
            self.done = True

        else:
            self.curr_path_index += 1

        self.x = self.path[self.curr_path_index][0]
        self.y = self.path[self.curr_path_index][1]

        hits = self.sensor.cast_rays(num_rays = 100, max_dist = 10)
        if hits == None:
            self.history.append((None, self.x, self.y))
        else:
            self.history.append((hits, self.x, self.y))

        self.sensor.update_position(self.x, self.y)
        self.hits = self.sensor.cast_rays(num_rays = 10, max_dist = 3)

    def __str__(self):
        return f'{self.id}: \n type: {self.type} \n position: [{self.x}, {self.y}]'