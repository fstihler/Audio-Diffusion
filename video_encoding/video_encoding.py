import os
from preprocessing.preprocessing import VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import av
import numpy as np
from transformers import AutoImageProcessor, VideoMAEModel
from huggingface_hub import hf_hub_download
import torch
import sys


# For VideoMAE
def read_video_pyav(container, indices, padding=0):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    stacked = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    print(f'Stacked frames: {stacked.shape}')
    if padding > 0:
        try:
            frame_shape = (padding, *frame.to_ndarray(format="rgb24").shape)
        except Exception as e:
            print(f'Cannot find frames in video contained: {e}')
            sys.exit()
        padded_frames = np.zeros(frame_shape)
        print(f'Padded frames: {padded_frames.shape}')
        stacked = np.concatenate([stacked, padded_frames], axis=0)
        print(f'Stacked shape: {stacked.shape}')
    return stacked


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    """
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def get_video_encoding(mp4_path: str, image_processor, model, padding=True, verbose=True):
    """
    Initialize image_processor as and model as follows:

    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    :param mp4_path:
    :param image_processor:
    :param model:
    :return:
    """
    video_path = mp4_path
    container = av.open(video_path)

    # sample 16 frames
    # indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
    total_frame_count = container.streams.video[0].frames
    groups_of_sixteen_frames = total_frame_count // 16
    if verbose:
        print(f'Total frames: {total_frame_count}')
    end_index = 0  # In case video has less than 16 frames
    for i in range(groups_of_sixteen_frames):
        start_index = i * 16
        end_index = ((i + 1) * 16) - 1
        indices = np.linspace(start_index, end_index, num=16)
        print(f'Encoding frames: {indices}')
        video = read_video_pyav(container, indices)

        # prepare video for the model
        inputs = image_processor(list(video), return_tensors="pt")

        # forward pass
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        if i == 0:
            stacked_hidden_states = last_hidden_states
        else:
            stacked_hidden_states = torch.cat([stacked_hidden_states, last_hidden_states], dim=1)
        print(f'Stacked encoded video now has shape {stacked_hidden_states.shape}')
    missing_frames = (total_frame_count - 1) - end_index
    if padding and missing_frames > 0:
        start_index = end_index + 1
        end_index = total_frame_count - 1
        indices = np.linspace(start_index, end_index, num=missing_frames)
        print(f'Encoding final frames + padding: {indices} + padding')
        video = read_video_pyav(container, indices, padding=16 - missing_frames)
        # prepare video for the model
        inputs = image_processor(list(video), return_tensors="pt")
        # forward pass
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        if groups_of_sixteen_frames == 0:
            stacked_hidden_states = last_hidden_states
        else:
            stacked_hidden_states = torch.cat([stacked_hidden_states, last_hidden_states], dim=1)
        print(f'Stacked encoded video now has shape {stacked_hidden_states.shape}')

    if verbose:
        print(f'Missing frames: {(total_frame_count - 1) - end_index}')
    return stacked_hidden_states



if __name__ == '__main__':
    # Load data
    # videos_root = os.path.join('data/videos_root')
    # annotation_file = os.path.join(videos_root, 'annotations.txt')
    #
    # preprocess = transforms.Compose([
    #     ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    #     # transforms.Resize(299),  # image batch, resize smaller edge to 299
    #     # transforms.CenterCrop(299),  # image batch, center crop to square 299x299
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    #
    # dataset = VideoFrameDataset(
    #     root_path=videos_root,
    #     annotationfile_path=annotation_file,
    #     num_segments=54,
    #     frames_per_segment=1,
    #     imagefile_template='{:06d}.jpg',
    #     transform=preprocess,
    #     test_mode=False
    # )
    #
    # sample = dataset[0]
    # frame_tensor = sample[0]
    #
    # np.random.seed(0)

    # file_path = hf_hub_download(
    #     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
    # )
    video_path = 'data/videos_root/00002/00002.mp4'
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

    video_encoding = get_video_encoding(video_path, image_processor, model, verbose=True)

    # Saving
    # encoded_data_directory = os.path.join('data', 'video_encodings')
    # if not os.path.exists(encoded_data_directory):
    #     os.mkdir(encoded_data_directory)
    # filename = os.path.basename(video_path)[:-4]

    # # Option 1: Save to npy: TOO HEAVY (15 MB)
    # stacked_hidden_states_array = stacked_hidden_states.detach().numpy()
    # filename = os.path.basename(video_path)[:-4]
    # filepath = os.path.join(encoded_data_directory, f'{filename}.npy')
    # np.save(filepath, stacked_hidden_states_array)

    # Option 2: Save to pt: too heavy (15 MB)
    # filepath = os.path.join(encoded_data_directory, f'{filename}.pt')
    # torch.save(stacked_hidden_states, filepath)

    # start_index = groups_of_sixteen_frames * 16
    # end_index = container.streams.video[0].frames
    # indices = np.linspace(start_index, end_index, num=end_index - start_index + 1)
    # print(indices)
    # video = read_video_pyav(container, indices)
    #
    # # prepare video for the model
    # inputs = image_processor(list(video), return_tensors="pt")
    #
    # # forward pass
    # outputs = model(**inputs)
    # last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states.shape)
    # stacked_hidden_states = torch.cat([stacked_hidden_states, last_hidden_states], dim=1)
    # print(stacked_hidden_states.shape)