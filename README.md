# construction-robotics-animation
 A simulation based animation generator for construction robotics. It features a 3 stage simulation: mapping with simulated drones and slam, demolition with ground based robots and A* pathfinding and construction of predefined structures.

# How to run the code?
 I recommend you run the code via conda:

 ```
 conda create -n construction python=3.11
 conda activate construction
 ```

 Once the environment is created and activated, you can install all the necesarry packages:

 ```
 conda install conda-forge::numpy
 conda install conda-forge::matplotlib
 conda install conda-forge::moviepy
 ```

 You can then run the code by using the main script:

 ```
 python main.py --map_size_x 70 --map_size_y 70 --num_robots 3 --output_name animation.mp4 --fps 60
 ```

 Your video animation will then be found inside the output directory/folder.
