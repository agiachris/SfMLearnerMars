import os
import time
import argparse

import numpy as np
import torch
import torch.optim
import torch.utils.data

from SfMLearnerMars.dataset import CEPTDataset
from SfMLearnerMars.models import (DispNet, PoseNet)
from SfMLearnerMars.losses import ViewSynthesisLoss
from SfMLearnerMars.utils import (Visualizer, compute_ate)

# experiment settings
parser = argparse.ArgumentParser(description="Test SfM on CEPT Dataset")
parser.add_argument('--exp-name', type=str, required=True, help='experiment name')
parser.add_argument('--disp-net', type=str, required=True, help='path to pre-trained disparity net weights')
parser.add_argument('--pose-net', type=str, required=True, help='path to pre-trained pose net weights')
parser.add_argument('--dataset-dir', type=str, default='./input', help='path to data root')
parser.add_argument('--output-dir', type=str, default='./output/', help='experiment directory')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')

# hyper-parameters
parser.add_argument('--batch-size', default=4, type=int, help='mini-batch size')

# loss weights
parser.add_argument('-p', '--photo-loss-weight', default=1, type=float, help='weight for photometric loss')
parser.add_argument('-m', '--mask-loss-weight', default=0, type=float, help='weight for explainabilty mask')
parser.add_argument('-s', '--smooth-loss-weight', default=0.01, type=float, help='weight for disparity smoothness loss')

# training details
parser.add_argument('--sequence-length', default=3, type=int, help='sequence length for training')
parser.add_argument('--rotation-mode', choices=['euler', 'quat'], default='euler', type=str,
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', choices=['zeros', 'border'], default='zeros', type=str,
                    help='padding mode for image warping : this is important for photometric differentiation when '
                         'going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')

# logging
parser.add_argument('--vis-per-epoch', default=100, type=int, help='visuals per epoch to save')
parser.add_argument('--skip-freq', default=5, type=int, help='sample frequency for visualization')


epo = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    args = parser.parse_args()
    exp_path = os.path.join(args.output_dir, args.exp_name)
    log_path = os.path.join(exp_path, 'logs')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(exp_path):
        print('Error: Experiment already exists, please rename --exp-name')
        exit()
    os.makedirs(log_path)
    print("All experiment outputs will be saved within:", exp_path)

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # get models, load pre-trained disparity network
    disp_net = DispNet.DispNet(1).to(device)
    disp_net.load_state_dict(torch.load(args.disp_net))
    disp_net.eval()
    pose_net = PoseNet.PoseNet(1, args.sequence_length - 1).to(device)
    pose_net.load_state_dict(torch.load(args.pose_net))
    pose_net.eval()

    # get sequence dataset
    test_set = CEPTDataset.CEPT(args.dataset_dir, 'test', args.sequence_length, args.seed)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # custom view synthesis loss and depth smoothness loss
    criterion = ViewSynthesisLoss(device, args.rotation_mode, args.padding_mode)
    w_synth, w_smooth = args.photo_loss_weight, args.smooth_loss_weight

    # visualizer
    visualizer = Visualizer(exp_path, device)

    print("Commencing testing on sequence 6...")
    # commence experiment
    # run test epoch and acquire pose estimation metrics
    start_time = time.time()
    l_test, ate, gt_traj, pred_traj, rate = test(disp_net, pose_net, test_loader, criterion, w_synth, w_smooth)
    total_time = time.time() - start_time
    visualizer.generate_random_visuals(disp_net, pose_net, test_loader, criterion,
                                       args.vis_per_epoch, epo, 'test', args.skip_freq)
    visualizer.generate_trajectories(gt_traj, pred_traj, epo, 'test', overlay=True)

    print_str = "ATE - {:.3f} | view synth loss - {:.3f} | smooth loss - {:.3f}".format(ate, l_test[1], l_test[2])
    print_str += "samples / sec - {:.5f} | total time - {:.3f}".format(rate, total_time)
    print(print_str)

    # save current stats
    np.savetxt(os.path.join(log_path, 'test_loss.txt'), l_test)
    np.savetxt(os.path.join(log_path, 'val_ate.txt'), np.array([ate]))
    np.savetxt(os.path.join(log_path, 'time_log.txt'), np.array([total_time]))


def test(disp_net, pose_net, test_loader, criterion, w1, w2):
    """Evaluate the current models over the validation sequence. Track pose estimation
    metrics (ATE) based on scale alignment of predicted poses with ground truth utm pose.
    """

    # track predicted pose deltas
    sequence_pose = []

    # track losses independently
    total_loss = np.zeros(3)
    total_time = 0
    for i, sample in enumerate(test_loader, 0):
        tgt_img, ref_imgs = sample
        tgt_img = tgt_img.to(device)
        ref_imgs = [ref_img.to(device) for ref_img in ref_imgs]

        # predict disparities at multiple scale spaces with DispNet
        start_time = time.time()
        disparities = [disp_net(tgt_img)]
        depth = [1 / disp for disp in disparities]

        # predict poses with PoseNet
        _, poses = pose_net(tgt_img, ref_imgs)
        total_time += time.time() - start_time

        # append target to reference frame pose estimates
        for pose_pred in poses.detach().cpu().numpy():
            sequence_pose.append(pose_pred[0, :])

        # compute photometric loss and smoothness loss
        view_synthesis_loss, warped_imgs, diff_imgs = \
            criterion.photometric_reconstruction_loss(tgt_img, depth, ref_imgs, poses)
        smoothness_loss = criterion.smoothness_loss(depth)

        # scale and fuse losses
        loss = w1 * view_synthesis_loss + w2 * smoothness_loss

        total_loss[0] += loss.item()
        total_loss[1] += view_synthesis_loss.item()
        total_loss[2] += smoothness_loss.item()

    # stack predicted poses
    sequence_pose = np.stack(sequence_pose)

    # get ground truth pose and corresponding target frame indices
    gt_pose, tgt_idx = test_loader.dataset.get_gt_pose()

    # compute ATE metric and acquire aligned trajectories
    ate, ate_mean, traj_gt, traj_pred = compute_ate(gt_pose, sequence_pose, tgt_idx)

    # compute forward pass rate
    rate = test_loader.dataset.__len__() / total_time

    return total_loss / i, ate_mean, traj_gt, traj_pred, rate
