import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def load_data(opts, mode='train', split_ratio=0.8, transform=None, batch_size=4, shuffle=True):

    dataset = MotionToMuscleDataset(opts, mode=mode, split_ratio=split_ratio, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def get_valid_pairs(data_dir, folder_names):
    valid_pairs = []
    for folder in folder_names:
        motion_dir = os.path.join(data_dir, folder, "motion")
        muscle_dir = os.path.join(data_dir, folder, "muscle")

        for subject in os.listdir(muscle_dir):
            subject_muscle_path = os.path.join(muscle_dir, subject)
            if not os.path.isdir(subject_muscle_path):
                continue
            
            subject_motion_path = os.path.join(motion_dir, subject)
            if not os.path.exists(subject_motion_path):
                continue

            for pose_folder in os.listdir(subject_muscle_path):
                muscle_motion_path = os.path.join(subject_muscle_path, pose_folder)
                motion_file_path = os.path.join(subject_motion_path, pose_folder + ".npz")
                muscle_file_path = os.path.join(muscle_motion_path, "muscle_forces.pkl")
                if os.path.exists(motion_file_path) and os.path.exists(muscle_file_path):
                    valid_pairs.append((motion_file_path, muscle_file_path))
    return valid_pairs

def get_info(motion_path):
    metadata_path = '/home/xuqianxu/muscles/mint_metadata.csv'
    df = pd.read_csv(metadata_path)

    # Extract target dataset and subject from the motion path
    target_dataset = motion_path.split('/')[5]
    target_subject = motion_path.split('/')[7].split('.')[0]

    # Filter the DataFrame for the matching row
    match = df[(df['dataset'] == target_dataset) & (df['subject'] == target_subject)]
    if match.empty:
        raise ValueError(f"No metadata found for dataset={target_dataset}, subject={target_subject}")
    
    row = match.iloc[0]
    gender_str = row['gender'].lower()
    gender = 1 if gender_str == 'female' else 0
    return gender, row['height_cm'], row['weight_kg']

import numpy as np

def load_motion_data(npz_path):
    data = np.load(npz_path)
    full_pose = data["poses"]  # [N, 152]
    #trans= data['trans']
    pose_66 = full_pose[:, :66] 

    # Discard parameters 31–36
    indices_to_keep = [i for i in range(66) if i < 30 or i > 35]
    pose_filtered = pose_66[:, indices_to_keep]  # [N, 60]

    # Reshape into [N, 20, 3] since 60 / 3 = 20 joints
    pose_reshaped = pose_filtered.reshape(-1, 20, 3)
    #trans_reshaped = trans.reshape(-1, 1, 3)
    mocap_rate = data["mocap_framerate"]
    #pose_reshaped = np.concatenate([trans_reshaped, pose_reshaped], axis=1)
    return pose_reshaped, mocap_rate

def load_muscle_data(pkl_path):
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)  
    muscle_times = df.index.to_numpy()         
    muscle_forces = df.to_numpy()             
    return muscle_times, muscle_forces

def align_motion_with_muscle(pose, mocap_rate, muscle_times, muscle_forces):
    indices = np.round(muscle_times * mocap_rate).astype(int)
    if indices[-1] >= pose.shape[0]:
        valid_length = np.searchsorted(indices, pose.shape[0], side='right')
        indices = indices[:valid_length]
        muscle_forces = muscle_forces[:valid_length, :]
    aligned_pose = pose[indices, :, :] 
    return aligned_pose, muscle_forces

class MotionToMuscleDataset(Dataset):
    def __init__(self, opts, mode='train', split_ratio=0.8, transform=None,
                 window_size=64, stride=32):
        
        self.data_dir = opts.data_dir
        self.folder_names = opts.folder_names
        self.eval_only = opts.eval_only
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        all_valid_pairs = get_valid_pairs(self.data_dir, self.folder_names)
        np.random.shuffle(all_valid_pairs)
        split = int(len(all_valid_pairs) * split_ratio)
        if mode == 'train':
            valid_pairs = all_valid_pairs[:split]
        elif mode == 'test':
            valid_pairs = all_valid_pairs[split:]
        else:
            raise ValueError("mode must be 'train' or 'test'")

        self.windows = []

        for motion_path, muscle_path in valid_pairs:
            gender, height, weight = get_info(motion_path)
            demo = np.array([gender, height, weight], dtype=np.float32)

            pose, mocap_rate = load_motion_data(motion_path) 
            muscle_times, muscle_forces = load_muscle_data(muscle_path)
            aligned_pose, aligned_muscle = align_motion_with_muscle(pose, mocap_rate, muscle_times, muscle_forces)

            if self.transform:
                sample = self.transform({"pose": aligned_pose, "muscle": aligned_muscle})
                aligned_pose, aligned_muscle = sample["pose"], sample["muscle"]

            total_len = aligned_pose.shape[0]
            sample_id = os.path.relpath(motion_path, self.data_dir)
            for start in range(0, total_len - window_size + 1, stride):
                pose_window = aligned_pose[start:start+window_size]
                muscle_window = aligned_muscle[start:start+window_size]
                self.windows.append((pose_window, muscle_window, demo, sample_id))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        pose, muscle, demo, sample_id = self.windows[idx] 
        pose = torch.tensor(pose, dtype=torch.float32)   
        muscle = torch.tensor(muscle, dtype=torch.float32) 
        demo = torch.tensor(demo, dtype=torch.float32) 
        if self.eval_only:
            return pose, muscle, demo, sample_id # sample_id used for visualization
        else:
            return pose, muscle, demo