import numpy as np
import os
from tqdm import tqdm

dataset_path = "F:/Datasets/H2S"

dataset_files = os.listdir(dataset_path)

accumalator = np.zeros((1, 126))

i = 0
max_length = 0
for dataset_file in tqdm(dataset_files):
    dataset = np.load(os.path.join(dataset_path, dataset_file))
    accumalator = np.concatenate((accumalator, dataset), axis=0)
    max_length = max(max_length, dataset.shape[0])
    i += 1

accumalator = accumalator[1:, :]
mean = np.mean(accumalator, axis=0)
std = np.std(accumalator, axis=0)

np.save(os.path.join(dataset_path, 'mean.npy'), mean)
np.save(os.path.join(dataset_path, 'std.npy'), std)

print('mean: ', mean.shape)
print('std: ', std.shape)
print('max_length: ', max_length)
