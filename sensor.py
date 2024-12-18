import numpy as np

class sensor:
    def __init__(self, real_map, x, y):
        self.map = real_map
        self.x = x
        self.y = y

    def cast_rays(self, num_rays, max_dist):
        angles = np.linspace(0, 2 * np.pi, num_rays)

        rows, cols = self.map.shape

        for angle in angles:
            x_curr = self.x
            y_curr = self.y

            dx = np.sin(angle)
            dy = np.cos(angle)

            dist = 0

            hit = False
            hits = []

            while dist < max_dist:
                x_curr += dx
                y_curr += dy
                dist += 0.1 #small arbitrary value

                 # Check if the ray is out of bounds
                grid_x, grid_y = int(round(x_curr)), int(round(y_curr))
                if grid_x < 0 or grid_x >= cols or grid_y < 0 or grid_y >= rows:
                    break

                # Check if the ray hits an obstacle
                if self.map[grid_y, grid_x] == 1:
                    hit = True
                    break

            if hit:
                hits.append((grid_x, grid_y))

        return hits
    
    def update_position(self, x, y):
        self.x = x
        self.y = y