import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from SfMLearnerMars.utils import (load_as_float, undistort_image, convert_date_string_to_unix_seconds)


train_runs = [('run1_base_hr', 42, 2550),
              ('run2_base_hr', 12, 2437),
              ('run3_base_hr', 5, 1821),
              ('run4_base_hr', 34, 2263)]
val_runs = [('run5_base_hr', 53, 3451)]
test_runs = [('run6_base_hr', 3, 3521)]


class CEPT(Dataset):
    def __init__(self, root, split, sequence_length=3, seed=0, scale=4):
        """Sequence dataset. Loads sequences of images at the set length and down-samples
            the images resolution.
        Args:
            root: root path to dataset
            split: data split -- one of ['train', 'val', 'test']
            sequence_length: length of each sequence (centered around target image)
            scale: down-sample factor of height and width of images
            seed: random seed
        """
        osj = os.path.join

        # fix seed
        np.random.seed(seed)
        random.seed(seed)

        # data split config
        self.use_gt_pose = False
        self.pose_aligned = False
        runs = None
        if split == 'train':
            runs = train_runs
        else:
            self.use_gt_pose = True
            if split == 'val':
                runs = val_runs
            elif split == 'test':
                runs = test_runs

        # want samples within a range of target frame
        sequence_shift = (sequence_length - 1) // 2
        shifts = list(range(-sequence_shift, sequence_shift+1))
        shifts.pop(sequence_shift)

        # construct sequence samples
        samples = []
        run_paths = [os.path.join(root, run[0], 'mono_image') for run in runs]
        run_filter = [(k[1], k[2]) for k in runs]
        for run_path, (k_s, k_f) in zip(run_paths, run_filter):
            # get image filenames
            img_paths = sorted(os.listdir(run_path))
            full_img_paths = [osj(run_path, img_path) for img_path in img_paths]
            run_samples = []

            for i in range(sequence_shift, len(full_img_paths)-sequence_shift):
                # store target and reference images
                sample = {'tgt_img': full_img_paths[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(full_img_paths[i+j])
                run_samples.append(sample)

            # filter out stationary samples
            samples.extend(run_samples[k_s:k_f])

        if split == 'train':
            random.shuffle(samples)

        self.samples = samples[:50]

        # align ground truth pose with target frames
        self.gt_pose = None
        if self.use_gt_pose:
            pose_path = osj(root, runs[0][0], 'global-pose-utm.txt')
            self.gt_pose = np.loadtxt(pose_path, dtype=np.float64, delimiter=',', skiprows=1)
            self.align_pose_with_targets()

        print("Loading {} set with {} samples - using ground truth pose: {}".format(
            split, len(self.samples), self.use_gt_pose
        ))

        # image resolution scale
        self.scale = scale

    def align_pose_with_targets(self):
        """For each ground truth pose, find the nearest target frame in terms of time.
        This is function is only called for validation and test sequences, as it assumes
        self.samples and self.gt_pose correspond to a single run.
        """

        assert (self.gt_pose is not None)
        assert (self.gt_pose.shape[1] == 8)
        pose = self.gt_pose
        samples = self.samples

        # find timestamp of first target frame    
        tgt_filename = os.path.basename(samples[0]['tgt_img'])
        tgt_filename = os.path.splitext(tgt_filename)[0]
        unix_time_seconds = convert_date_string_to_unix_seconds(tgt_filename[12:])

        # find starting pose based on smallest time delta to first target frame
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

        # match each pose estimate to the nearest target image frame and store index
        run_samples_idx = []
        prev_idx = 0
        for i in range(len(pose)):

            # find closest camera time
            pose_time = pose[i, 0]
            min_diff = float('inf')

            for j in range(prev_idx, len(samples)):
                curr_filename = os.path.basename(samples[j]['tgt_img'])
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
            run_samples_idx.append(prev_idx)

            # reached last moving camera frame
            prev_idx += 1
            if prev_idx >= len(samples):
                break

        # should be a target frame index for each pose
        assert (pose.shape[0] == len(run_samples_idx))
        self.gt_pose = pose
        self.pose_tgt_idx = np.array(run_samples_idx)
        self.pose_aligned = True

    def get_gt_pose(self):
        """Return global utm pose corresponding to a validation / test sequence. 
        ALso provide a list of target frame indices associated with each pose.
        """
        if self.use_gt_pose and self.pose_aligned:
            return self.gt_pose, self.pose_tgt_idx

        return None

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
        ref_imgs = []
        for ref in sample['ref_imgs']:
            ref_img = load_as_float(ref)
            ref_img = undistort_image(ref_img)
            # downsample and normalize reference image resolution
            ref_img = np.expand_dims(cv2.resize(ref_img, (w // s, h // s)), 0) / 255.0
            ref_imgs.append(ref_img)

        return tgt_img, ref_imgs
