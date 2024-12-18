import os
import sys
from moviepy.editor import VideoFileClip, concatenate_videoclips

def combine_videos(output_folder, output_name="combined_video.mp4"):
    # Define the specific order of the videos
    video_order = ['exploration.mp4', 'demolition.mp4', 'construction.mp4']
    
    # List to store video clips
    clips = []
    
    # Iterate over the order and append the video clips to the list
    for video_name in video_order:
        video_path = os.path.join(output_folder, video_name)
        
        if os.path.exists(video_path):
            clip = VideoFileClip(video_path)
            clips.append(clip)
        else:
            print(f"Warning: {video_name} does not exist in the folder.")
    
    # Combine all video clips
    if clips:
        final_clip = concatenate_videoclips(clips)
        
        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Define output file path with the provided output name in the output folder
        output_path = os.path.join(output_folder, output_name)
        
        # Write the final video to a file
        final_clip.write_videofile(output_path, codec='libx264')
        
        print(f"Videos combined successfully into {output_path}")
        
        # Close the clips to free resources
        final_clip.close()
        for clip in clips:
            clip.close()

    # Optionally, delete the original mp4 files from the folder
    for video_name in video_order:
        video_path = os.path.join(output_folder, video_name)
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"Deleted {video_name} from the folder.")
        else:
            print(f"Warning: {video_name} was not found to delete.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_folder = sys.argv[1]  # The output folder path
        output_name = sys.argv[2] if len(sys.argv) > 2 else "combined_video.mp4"
        combine_videos(output_folder, output_name)
    else:
        print("Please provide the path to the output folder as an argument.")
