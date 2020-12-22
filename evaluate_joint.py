import os
import time
import shutil
import argparse
import datetime


import numpy as np
import torch
import torch.optim
import torch.utils.data

from SfMLearnerMars.dataset import CPETDataset
from SfMLearnerMars.models import (DispNet, PoseNet)
from SfMLearnerMars.losses import ViewSynthesisLoss
from SfMLearnerMars.utils import (Visualizer, compute_ate_horn, compute_ate_umeyama)

from rob501_project import parser


# local experiment settings - configure exp name, network weights, and test sequence
# parser = argparse.ArgumentParser(description="Test SfM on CPET Dataset")
# parser.add_argument('--exp-name', type=str, required=True, help='experiment name')
# parser.add_argument('--disp-net', type=str, required=True, help='path to pre-trained disparity net weights')
# parser.add_argument('--pose-net', type=str, required=True, help='path to pre-trained pose net weights')
# parser.add_argument('--run-sequence', type=str, required=True, help='run of the CPET to evaluate on',
#                     choices=['run1', 'run2', 'run3', 'run4', 'run5', 'run6', 'val', 'test'])

# docker call settings - default exp name, network weights, and test sequence
parser.add_argument('--exp-name', type=str, default='SfMLearnerMars_official', help='experiment name')
parser.add_argument('--disp-net', type=str, help='path to pre-trained disparity net weights',
                    default='src/SfMLearnerMars/exp/disp_net_19')
parser.add_argument('--pose-net', type=str, help='path to pre-trained pose net weights',
                    default='src/SfMLearnerMars/exp/pose_net_19')
parser.add_argument('--run-sequence', type=str, default='test', help='run of the CPET to evaluate on',
                    choices=['run1', 'run2', 'run3', 'run4', 'run5', 'run6', 'val', 'test'])

parser.add_argument('--dataset-dir', type=str, default='./input', help='path to data root')
parser.add_argument('--output-dir', type=str, default='./output', help='experiment directory')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')

# hyper-parameters
parser.add_argument('--batch-size', default=32, type=int, help='mini-batch size')
parser.add_argument('-p', '--photo-loss-weight', default=1, type=float, help='weight for photometric loss')
parser.add_argument('-m', '--mask-loss-weight', default=0, type=float, help='weight for explainabilty mask')
parser.add_argument('-s', '--smooth-loss-weight', default=0.01, type=float, help='weight for disparity smoothness loss')

# evaluation details
parser.add_argument('--sequence-length', default=3, type=int, help='sequence length for training')
parser.add_argument('--rotation-mode', choices=['euler', 'quat'], default='euler', type=str,
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', choices=['zeros', 'border'], default='zeros', type=str,
                    help='padding mode for image warping : this is important for photometric differentiation when '
                         'going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')

# logging
parser.add_argument('--vis-per-epoch', default=50, type=int, help='visuals per epoch to save')
parser.add_argument('--skip-freq', default=3, type=int, help='sample frequency (over batches) for visualization')


epo = 0
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device('cpu')    # no GPU deployment on docker


def main(input_dir=None, output_dir=None, run_sequence=''):
    args = parser.parse_args()

    # docker call - change default input and output directories
    if input_dir is not None and output_dir is not None:
        args.dataset_dir = input_dir
        args.output_dir = output_dir
    if run_sequence != '':
        args.run_sequence = run_sequence
        run_sequence = '_' + run_sequence

    exp_path = os.path.join(args.output_dir, args.exp_name + run_sequence)
    log_path = os.path.join(exp_path, 'logs')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(exp_path):
        print('Error: Experiment already exists, over-writing experiment')
        shutil.rmtree(exp_path)

    os.makedirs(log_path)
    print("All experiment outputs will be saved within:", exp_path)

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # get models, load pre-trained disparity network and pose network
    disp_net = DispNet.DispNet(1).to(device)
    disp_net.load_state_dict(torch.load(args.disp_net, map_location='cpu'))
    disp_net.eval()
    pose_net = PoseNet.PoseNet(1, args.sequence_length - 1).to(device)
    pose_net.load_state_dict(torch.load(args.pose_net, map_location='cpu'))
    pose_net.eval()

    # get sequence dataset
    test_set = CPETDataset.CPET(args.dataset_dir, args.run_sequence, args.sequence_length, args.seed)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # custom view synthesis loss and depth smoothness loss
    criterion = ViewSynthesisLoss(device, args.rotation_mode, args.padding_mode)
    w1, w2 = args.photo_loss_weight, args.smooth_loss_weight

    # visualizer
    visualizer = Visualizer(exp_path, device)

    print("Commencing testing on {} sequence...".format(args.run_sequence))

    # run test epoch, acquire pose estimation metrics (ATE) from Horn's Method and Umeyama Method
    start_time = time.time()
    l_test, horn, umeyama, rate = test(disp_net, pose_net, test_loader, criterion, visualizer, args.skip_freq, w1, w2)
    total_time = time.time() - start_time

    # visualize estimated and ground truth trajectories in BEV / 3D - Horn's alignment
    visualizer.generate_trajectories(horn[2], horn[3], "Horns", epo, args.run_sequence)
    visualizer.generate_3d_trajectory(horn[2], horn[3], "Horns", epo, args.run_sequence)

    # visualize estimated and ground truth trajectories in BEV / 3D - Umeyama alignment
    visualizer.generate_trajectories(umeyama[2], umeyama[3], "Umeyama", epo, args.run_sequence)
    visualizer.generate_3d_trajectory(umeyama[2], umeyama[3], "Umeyama", epo, args.run_sequence)

    # visualize trajectories independently - Umeyama
    visualizer.generate_trajectory(umeyama[2], 'gt', 'True', epo, args.run_sequence)
    visualizer.generate_trajectory(umeyama[3], 'pred', 'Estimated', epo, args.run_sequence)

    print_str = "ATE (Umeyama) - {:.3f} | ATE (Horn's) - {:.3f}".format(umeyama[1], horn[1])
    print_str += " | view synth loss - {:.3f} | smooth loss - {:.3f}".format(l_test[1], l_test[2])
    print_str += " | Hz - {:.5f} | total time - {}".format(rate, datetime.timedelta(seconds=total_time))
    print(print_str)

    # save current stats
    np.savetxt(os.path.join(log_path, '{}_loss.txt'.format(args.run_sequence)), l_test)
    np.savetxt(os.path.join(log_path, '{}_ate_mean_horn.txt'.format(args.run_sequence)), np.array([horn[1]]))
    np.savetxt(os.path.join(log_path, '{}_ate_full_horn.txt'.format(args.run_sequence)), horn[0])
    np.savetxt(os.path.join(log_path, '{}_ate_mean_umeyama.txt'.format(args.run_sequence)), np.array([umeyama[1]]))
    np.savetxt(os.path.join(log_path, '{}_ate_full_umeyama.txt'.format(args.run_sequence)), umeyama[0])
    np.savetxt(os.path.join(log_path, 'time_log.txt'), np.array([total_time]))
    print('-----')


def test(disp_net, pose_net, test_loader, criterion, visualizer, skip_freq, w1, w2):
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

        # append relative frame pose estimates across batch. Target-to-reference
        for pose_pred in poses.detach().cpu().numpy():
            sequence_pose.append(pose_pred[0, :])

        # compute photometric loss and smoothness loss
        view_synthesis_loss, warped_imgs, diff_imgs = \
            criterion.photometric_reconstruction_loss(tgt_img, depth, ref_imgs, poses)
        smoothness_loss = criterion.smoothness_loss(depth)

        # save visuals
        if i % skip_freq == 0:
            visualizer.save_sample(tgt_img, ref_imgs, depth, warped_imgs, i)

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
    horn_result = compute_ate_horn(gt_pose, sequence_pose, tgt_idx)
    umeyama_result = compute_ate_umeyama(gt_pose, sequence_pose, tgt_idx)

    # compute forward pass rate
    rate = test_loader.dataset.__len__() / total_time

    return total_loss / i, horn_result, umeyama_result, rate
