import os
import time
import argparse
import datetime

import numpy as np
import torch
import torch.optim
import torch.utils.data

from dataset import CPETDataset
from models import (DispNet, PoseNet)
from losses import ViewSynthesisLoss
from utils import (Visualizer, compute_ate_horn, model_checkpoint, generate_curve)


# experiment settings
parser = argparse.ArgumentParser(description="Train SfM on CPET Dataset")
parser.add_argument('--exp-name', type=str, required=True, help='experiment name')
parser.add_argument('--disp-net', type=str, default=None, help='path to pre-trained disparity net weights')
parser.add_argument('--dataset-dir', type=str, required=True, help='path to data root')
parser.add_argument('--output-dir', type=str, default='./exp', help='experiment directory')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')

# hyper-parameters
parser.add_argument('--epochs', default=50, type=int,  help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, help='mini-batch size')
parser.add_argument('--learning-rate', default=2e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, help='beta parameters for adam')
parser.add_argument('--weight-decay', default=0, type=float, help='weight decay')
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
parser.add_argument('--save-freq', default=1, type=int, help='model checkpoint frequency')
parser.add_argument('--vis-per-epoch', default=20, type=int, help='visuals per epoch to save')


epo = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    exp_path = os.path.join(args.output_dir, args.exp_name)
    log_path = os.path.join(exp_path, 'logs')
    checkpoint_path = os.path.join(exp_path, 'checkpoints')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(exp_path):
        print('Error: Experiment already exists, please rename --exp-name')
        exit()
    os.makedirs(log_path)
    os.mkdir(checkpoint_path)
    print("All experiment outputs will be saved within:", exp_path)

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # get models and load pre-trained disparity network
    disp_net = DispNet.DispNet(1).to(device)
    disp_net.init_weights()
    if args.disp_net is not None:
        disp_net.load_state_dict(torch.load(args.disp_net, map_location='cpu'))
    disp_net.train()
    pose_net = PoseNet.PoseNet(1, args.sequence_length-1).to(device)
    pose_net.init_weights()
    pose_net.train()

    # joint optimizer (pose and depth)
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.learning_rate},
        {'params': pose_net.parameters(), 'lr': args.learning_rate}
    ]
    optim = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)

    # get sequential dataset
    train_set = CPETDataset.CPET(args.dataset_dir, 'train', args.sequence_length, args.seed)
    val_set = CPETDataset.CPET(args.dataset_dir, 'val', args.sequence_length, args.seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # custom view synthesis loss and depth smoothness loss
    criterion = ViewSynthesisLoss(device, args.rotation_mode, args.padding_mode)
    w_synth, w_smooth = args.photo_loss_weight, args.smooth_loss_weight

    # visualizer
    visualizer = Visualizer(exp_path, device)

    # commence experiment
    print("Experiment commencing on 4 train seq and 1 validation seq for {} epochs...".format(args.epochs))
    start_time = time.time()

    # track losses and absolute trajectory error
    train_loss = np.zeros((args.epochs, 3))
    val_loss = np.zeros((args.epochs, 3))
    val_ate_mean = np.zeros(args.epochs)
    total_time = np.zeros(args.epochs)

    for epo in range(args.epochs):

        # run training epoch and generate / save random visualizations
        l_train = train_epoch(disp_net, pose_net, train_loader, criterion, optim, w_synth, w_smooth)
        train_loss[epo, :] = l_train[:]
        visualizer.generate_random_visuals(disp_net, pose_net, train_loader, criterion,
                                           args.vis_per_epoch, epo, 'train')

        # run validation epoch and acquire pose estimation metrics. Plot trajectories
        l_val, ate, ate_mean, gt_traj, pred_traj = validate(disp_net, pose_net, val_loader,
                                                            criterion, w_synth, w_smooth)
        val_loss[epo, :] = l_val[:]
        val_ate_mean[epo] = ate_mean

        # visualization of disparity maps, BEV trajectories, and 3D trajectories
        visualizer.generate_random_visuals(disp_net, pose_net, val_loader, criterion, args.vis_per_epoch, epo, 'val')
        visualizer.generate_trajectory(pred_traj, 'pred', 'Estimated', epo, 'val')
        visualizer.generate_trajectories(gt_traj, pred_traj, "Horns", epo, 'val')
        visualizer.generate_3d_trajectory(gt_traj, pred_traj, "Horns", epo, 'val')
        if epo == 0:
            visualizer.generate_trajectory(gt_traj, 'gt', 'True', epo, 'val')

        total_time[epo] = time.time() - start_time
        print_str = "epo - {}/{} | train_loss - {:.3f} | val_loss - {:.3f} | ".format(
            epo, args.epochs, train_loss[epo, 0], val_loss[epo, 0])
        print_str += "val_ate - {:.3f} | total_time - {}".format(ate_mean, datetime.timedelta(seconds=total_time[epo]))
        print(print_str)

        # save models
        if (epo+1) % args.save_freq == 0:
            model_checkpoint(disp_net, 'disp_net_' + str(epo+1), checkpoint_path)
            model_checkpoint(pose_net, 'pose_net_' + str(epo+1), checkpoint_path)

        # save current stats
        np.savetxt(os.path.join(log_path, 'train_loss.txt'), train_loss)
        np.savetxt(os.path.join(log_path, 'val_loss.txt'), val_loss)
        np.savetxt(os.path.join(log_path, 'val_ate_mean.txt'), val_ate_mean)
        np.savetxt(os.path.join(log_path, 'time_log.txt'), total_time)

    # generate metric curves
    generate_curve([train_loss[:, 0], val_loss[:, 0]], ['train', 'val'], 'loss',
                   'Train vs Val Combined Loss', log_path)
    generate_curve([train_loss[:, 1], val_loss[:, 1]], ['train', 'val'], 'photometric loss',
                   'Train vs Val Photometric Reconstruction Loss', log_path)
    generate_curve([train_loss[:, 2], val_loss[:, 2]], ['train', 'val'], 'depth smooth loss',
                   'Train vs Val Depth Smoothness Loss', log_path)
    generate_curve([val_ate_mean], ['val'], 'ATE', 'Validation Absolute Trajectory Error', log_path)


def train_epoch(disp_net, pose_net, train_loader, criterion, optim, w1, w2):
    """Run a single epoch over the training sequences.
    Args:
        disp_net: unsupervised multi-scale disparity prediction deep CNN
        pose_net: unsupervised pose prediction deep CNN
        train_loader: pytorch dataloader for training set
        criterion: ViewSynthesisLoss object for computing photometric and smoothness loss
        optim: joint pose and depth prediction optimizer
        w1: photometric loss weight
        w2: smoothness loss weight
    """

    # track losses independently
    total_loss = np.zeros(3)

    for i, sample in enumerate(train_loader, 0):
        tgt_img, ref_imgs = sample
        tgt_img = tgt_img.to(device)
        ref_imgs = [ref_img.to(device) for ref_img in ref_imgs]

        # predict disparities at multiple scale spaces with DispNet
        disparities = disp_net(tgt_img)
        depth = [1 / disp for disp in disparities]

        # predict poses with PoseNet (explainability mask not used)
        _, poses = pose_net(tgt_img, ref_imgs)

        # compute photometric loss and smoothness loss
        view_synthesis_loss, warped_imgs, diff_imgs = \
            criterion.photometric_reconstruction_loss(tgt_img, depth, ref_imgs, poses)
        smoothness_loss = criterion.smoothness_loss(depth)

        # scale and fuse losses
        loss = w1 * view_synthesis_loss + w2 * smoothness_loss

        # gradient update
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss[0] += loss.item()
        total_loss[1] += view_synthesis_loss.item()
        total_loss[2] += smoothness_loss.item()

    return total_loss / i


def validate(disp_net, pose_net, val_loader, criterion, w1, w2):
    """Evaluate the current models over the validation sequence. Track pose estimation
    metrics (ATE) based on scale alignment of predicted poses with ground truth utm pose.
    """
    disp_net.eval()
    pose_net.eval()

    # track relative pose estimates
    sequence_pose = []

    # track losses independently
    total_loss = np.zeros(3)
    for i, sample in enumerate(val_loader, 0):
        tgt_img, ref_imgs = sample
        tgt_img = tgt_img.to(device)
        ref_imgs = [ref_img.to(device) for ref_img in ref_imgs]

        # predict disparities at multiple scale spaces with DispNet
        disparities = [disp_net(tgt_img)]
        depth = [1 / disp for disp in disparities]

        # predict poses with PoseNet
        _, poses = pose_net(tgt_img, ref_imgs)

        # append relative frame pose estimates across batch. Target-to-reference
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

    # stack predicted poses -- [N, 6]
    sequence_pose = np.stack(sequence_pose)

    # get ground truth pose and corresponding target frame indices
    gt_pose, tgt_idx = val_loader.dataset.get_gt_pose()
    
    # compute ATE metric and acquire aligned trajectories -- [M, 3]
    ate, ate_mean, traj_gt, traj_pred = compute_ate_horn(gt_pose, sequence_pose, tgt_idx)

    disp_net.train()
    pose_net.train()
    return total_loss / i, ate, ate_mean, traj_gt, traj_pred


if __name__ == '__main__':
    arguments = parser.parse_args()
    main(arguments)
