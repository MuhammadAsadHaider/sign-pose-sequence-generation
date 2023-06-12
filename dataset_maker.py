import numpy as np
import os
from tqdm import tqdm

raw_data_folders_path = "F:/Datasets/features"
dataset_path = "F:/Datasets/H2S"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

raw_data_folders = os.listdir(raw_data_folders_path)
for raw_data_folder in tqdm(raw_data_folders):
    right_hand_file = os.path.join(raw_data_folders_path, raw_data_folder, 'right_hand.npy')
    left_hand_file = os.path.join(raw_data_folders_path, raw_data_folder, 'left_hand.npy')

    right_hand_kp = np.load(right_hand_file)
    left_hand_kp = np.load(left_hand_file)

    right_hand_kp_flattened = right_hand_kp.reshape(right_hand_kp.shape[0], -1)
    left_hand_kp_flattened = left_hand_kp.reshape(left_hand_kp.shape[0], -1)

    features = np.concatenate((right_hand_kp_flattened, left_hand_kp_flattened), axis=1)
    np.save(os.path.join(dataset_path, raw_data_folder + '.npy'), features)

    # verify that the features are correct
    # loaded_features = np.load(os.path.join(dataset_path, raw_data_folder + '.npy'))
    # loaded_right_hand_kp = loaded_features[:, :63]
    # loaded_left_hand_kp = loaded_features[:, 63:]
    # assert np.array_equal(right_hand_kp_flattened, loaded_right_hand_kp)
    # assert np.array_equal(left_hand_kp_flattened, loaded_left_hand_kp)




