import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from utils import (load_as_float, undistort_image, pose_vec2mat,
                   cam_to_rover_coeffs, convert_date_string_to_unix_seconds)


train_runs = [('run1_base_hr', 42, 2550),
              ('run2_base_hr', 12, 2437),
              ('run3_base_hr', 5, 1821),
              ('run4_base_hr', 34, 2263)]
val_runs = [('run5_base_hr', 53, 3451)]
test_runs = [('run6_base_hr', 3, 3521)]


class CPETDepth(Dataset):
    def __init__(self, root, split, seed=0, scale=4):
        """Sequence dataset. Loads sequences of images at the set length and down-samples
            the images resolution.
        Args:
            root: root path to dataset
            split: data split -- one of ['train', 'val', 'test']
            scale: down-sample factor of height and width of images
            seed: random seed
        """
        osj = os.path.join

        # fix seed
        np.random.seed(seed)
        random.seed(seed)

        # data split config
        runs = None
        if split == 'train':
            runs = train_runs
        elif split == 'val':
            runs = val_runs
        elif split == 'test':
            runs = test_runs

        # get monocular image and pose paths
        run_paths = [os.path.join(root, run[0], 'mono_image') for run in runs]
        run_filter = [(k[1], k[2]) for k in runs]
        pose_paths = [osj(root, run[0], 'global-pose-utm.txt') for run in runs]
        gt_pose = [np.loadtxt(p, dtype=np.float64, delimiter=',', skiprows=1) for p in pose_paths]

        # construct sequence of target and reference frame samples at pose frequency
        samples = []
        for run_path, (k_s, k_f), pose in zip(run_paths, run_filter, gt_pose):
            # get image filenames
            img_paths = sorted(os.listdir(run_path))
            full_img_paths = [osj(run_path, img_path) for img_path in img_paths][k_s:k_f]

            # get first reference frame (earliest timestamp)
            ref_filename = os.path.basename(full_img_paths[0])
            ref_filename = os.path.splitext(ref_filename)[0]
            unix_time_seconds = convert_date_string_to_unix_seconds(ref_filename[12:])

            # find starting pose based on smallest time delta to first reference frame
            min_diff = float('inf')
            first_idx = 0
            for j in range(0, len(pose)):
                curr_diff = abs(unix_time_seconds - pose[j][0])
                if curr_diff < min_diff:
                    min_diff = curr_diff
                    first_idx = j
                else:
                    break
            pose = pose[first_idx:, ...]

            # match each pose estimate to the nearest image frame (downsampling image frames)
            run_samples = []
            prev_idx = 0
            for i in range(len(pose)):

                # find closest camera time
                pose_time = pose[i, 0]
                min_diff = float('inf')

                for j in range(prev_idx, len(full_img_paths)):
                    curr_filename = os.path.basename(full_img_paths[j])
                    curr_filename = os.path.splitext(curr_filename)[0]
                    unix_time_seconds = convert_date_string_to_unix_seconds(curr_filename[12:])

                    # check if closest
                    curr_diff = abs(pose_time - unix_time_seconds)
                    if curr_diff < min_diff:
                        min_diff = curr_diff
                        prev_idx = j
                    else:
                        break

                # store camera frame and nearest pose
                sample = {'img': full_img_paths[prev_idx], 'pose': pose[i, :]}
                run_samples.append(sample)

                # reached last moving camera frame
                prev_idx += 1
                if prev_idx >= len(full_img_paths):
                    break

            # add run to samples
            samples.append(run_samples)

        # create (target, reference, transform) sequence
        cam_to_rover = pose_vec2mat(cam_to_rover_coeffs, 'quat')
        sequence = []
        for i in range(len(samples)):
            run_samples = samples[i]
            for j in range(1, len(run_samples)):
                sample = {}
                sample['tgt_img'] = run_samples[j]['img']
                sample['ref_img'] = run_samples[j-1]['img']

                H_tgt_to_ref = np.identity(4)
                tgt_pose = run_samples[j]['pose']
                ref_pose = run_samples[j-1]['pose']

                # target cam to global
                tgt_rover_to_global = pose_vec2mat(tgt_pose[1:], 'quat')
                tgt_mono_to_global = tgt_rover_to_global @ cam_to_rover

                # ref cam to global
                ref_rover_to_global = pose_vec2mat(ref_pose[1:], 'quat')
                ref_mono_to_global = ref_rover_to_global @ cam_to_rover

                # compute relative translation
                tgt_to_ref = np.linalg.inv(ref_mono_to_global) @ tgt_mono_to_global
                
                sample['tgt_to_ref'] = tgt_to_ref.astype(np.float32)
                sequence.append(sample)

        random.shuffle(sequence)
        self.samples = sequence
        print("Loading {} set with {} samples".format(split, len(self.samples)))

        # image resolution scale
        self.scale = scale

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        s = self.scale

        # load and undistort target image
        tgt_img = load_as_float(sample['tgt_img'])
        tgt_img = undistort_image(tgt_img)
        h, w = tgt_img.shape
        # downsample and normalize target image resolution
        tgt_img = np.expand_dims(cv2.resize(tgt_img, (w // s, h // s)), 0) / 255.0

        # load and undistort reference images
        ref_img = sample['ref_img']
        ref_img = load_as_float(ref_img)
        ref_img = undistort_image(ref_img)
        # downsample and normalize reference image resolution
        ref_img = np.expand_dims(cv2.resize(ref_img, (w // s, h // s)), 0) / 255.0

        return tgt_img, ref_img, sample['tgt_to_ref']
