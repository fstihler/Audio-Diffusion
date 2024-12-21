from preprocessing.preprocessing import *
import os

if __name__ == '__main__':
    video_path = 'data/videos_root/00002/00002.mp4'
    audio_dirpath = os.path.dirname(video_path)
    returned = convert_video_into_audio(video_path=video_path, audio_dirpath=audio_dirpath)
    print(f'Convert video to audio function returned {returned}')