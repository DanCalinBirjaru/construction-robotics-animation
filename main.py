import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from map_builder import generate_map
from robot import robot
from manager import manager

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from combine_videos import combine_videos

# Create the parser
parser = argparse.ArgumentParser(description="A script that generates the animation.")

# Arguments
parser.add_argument('--map_size_x', type=int, help="Map length.", required=True)
parser.add_argument('--map_size_y', type=int, help="Map width.", required=True)

parser.add_argument('--num_robots', type=int, help="Number of robots.", required=True)

parser.add_argument('--output_name', type=str, help="Output file name.", required=True)
parser.add_argument('--fps', type=int, help="Number of frames per second.", required=True)

# Parse arguments
args = parser.parse_args()

# Access the variable
map_size_x = args.map_size_x
map_size_y = args.map_size_y

num_robots = args.num_robots

output_name = args.output_name
fps = args.fps

# Function to map values to RGB colors
def map_to_rgb(frame):
    # Create a blank RGB image (shape: (height, width, 3))
    rgb_image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    
    # Map the values in the frame to specific colors
    rgb_image[frame == -2] = [255, 0, 0]         # Red for -2
    rgb_image[frame == 0] = [68, 4, 84]          # '#440154' for 0
    rgb_image[frame == 1] = [255, 237, 35]       # '#FFED23' for 1
    rgb_image[frame == -1] = [0, 128, 0]          # Green for -1
    
    return rgb_image

# Function to create the animation
def create_animation(frames, output_file='animation.mp4', fps=15):
    frame_count, height, width = frames.shape

    # Create the figure and axis for the animation
    fig, ax = plt.subplots(figsize=(6, 6))

    # Initially plot the first frame by converting it to an RGB image
    im = ax.imshow(map_to_rgb(frames[0]))
    ax.axis('off')  # Hide axes for a cleaner image

    # Function to update the plot at each frame
    def update(frame):
        # Update the image data for the next frame
        im.set_data(map_to_rgb(frames[frame]))
        #ax.set_title(f"Frame {frame + 1}")  # Optionally add a title to each frame
        return [im]  # Return the image object for blitting

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(frame_count), interval=1000 / fps, blit=True)

    # Save the animation as an MP4 file using ffmpeg writer
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps)
    anim.save(output_file, writer=writer)
    print(f"Animation saved as {output_file}")

def do_the_work(map, type):
    print(f'\nWorking on {type}.')
    robots_x = np.linspace(0, map_size_x, num_robots + 1, dtype = int)[:-1]
    robots_y = np.zeros(num_robots, dtype = int)

    robots = []
    print('\nGenerating robots...')
    for i in range(len(robots_x)):
        path = [(robots_x[i], robots_y[i])]
        r = robot(i, type, path, map)
        robots.append(r)
        #print(r)

    print('Generating animation manager...')
    _manager = manager(robots, map)

    print('Generating frames...')
    if type == 'exploration':
        frames = _manager.get_frames(_manager.get_mapping_paths)
    
    elif type == 'demolition':
        frames = _manager.get_frames(_manager.get_demolition_paths)

    elif type == 'construction':
        frames = _manager.get_frames(_manager.get_construction_paths)

    else:
        print('Unsupported type for path generation.')

    print('Generating animation...')

    create_animation(frames, f'output/{type}.mp4', fps)

print(f'Generating map of size ({map_size_x}, {map_size_y})...')
map = generate_map(map_size_x, map_size_y, 0.9, 0, 0, 0, 0)

do_the_work(map, 'exploration')
do_the_work(map.T, 'demolition')
do_the_work(np.zeros_like(map), 'construction')

print('\nCombining animations...')
output_folder = 'output/'
combine_videos(output_folder, output_name)