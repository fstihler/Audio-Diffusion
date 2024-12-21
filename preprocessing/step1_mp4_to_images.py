from preprocessing.preprocessing import *
import os

if __name__ == '__main__':
    video_path = 'data/videos_root/00002/00002.mp4'
    images_path = os.path.dirname(video_path)
    returned = split_video_into_images(video_path=video_path, images_path=images_path)
    print(f'Split video function returned {returned}')