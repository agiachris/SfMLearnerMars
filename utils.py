import os
import cv2
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt


# globals
intrinsics = np.array([[904.04572636, 0, 645.74398382],
                       [0, 907.01811462, 512.14951996],
                       [0, 0, 1]], dtype=np.float32)

# Description, Symbol, trans_x [m], trans_y [m], trans_z [m], quat_x, quat_y, quat_z, quat_w
cam_to_rover_coeffs = np.array([0.305,-0.003,0.604,-0.579,0.584,-0.407,0.398], dtype=np.float32)

# k1, k2, p1, p2, k3
distortion_coeffs = np.array([-0.3329137, 0.10161043, 0.00123166, -0.00096204, -0])


def convert_date_string_to_unix_seconds(date_and_time):
    """Convert a date & time to a unix timestamp in seconds.

    Input:
        date_and_time (str): date and time (Toronto time zone) in the
            'YYYY_MM_DD_hh_mm_ss_microsec' format
    Return:
        float: equivalent unix timestamp, in seconds
    """
    # Extract microseconds part and convert it to seconds
    microseconds_str = date_and_time.split('_')[-1]
    seconds_remainder = float('0.' + microseconds_str)

    # Extract date without microseconds and convert to unix timestamp
    # The added 'GMT-0400' indicates that the provided date is in the
    # Toronto (eastern) timezone during daylight saving
    date_to_sec_str = date_and_time.replace('_' + microseconds_str, '')
    seconds_to_sec = \
        datetime.datetime.strptime(date_to_sec_str + \
                                   ' GMT-0400', "%Y_%m_%d_%H_%M_%S GMT%z").timestamp()

    # Add the microseconds remainder
    return seconds_to_sec + seconds_remainder


def load_as_float(path):
    """Load grayscale image tensor.
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)


def undistort_image(img):
    """Undistort grascale image with intrinsic and distortion coefficients.
    """
    return cv2.undistort(img, intrinsics, distortion_coeffs)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [3, 3]
    """

    x, y, z = angle

    cosz = np.cos(z)
    sinz = np.sin(z)
    zmat = np.array([[cosz, -sinz, 0],
                     [sinz, cosz, 0],
                     [0, 0, 1]], dtype=np.float32)

    cosy = np.cos(y)
    siny = np.sin(y)
    ymat = np.array([[cosy, 0, siny],
                     [0, 1, 0],
                     [-siny, 0, cosy]], dtype=np.float32)

    cosx = np.cos(x)
    sinx = np.sin(x)
    xmat = np.array([[1, 0, 0],
                     [0, cosx, -sinx],
                     [0, sinx, cosx]], dtype=np.float32)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: coeff of quaternion of rotation. -- size = [4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [3, 3]
    """
    x, y, z, w = quat
    w2, x2, y2, z2 = w**2, x**2, y**2, z**2
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = np.array([[w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz],
                       [2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx],
                       [2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2]], dtype=np.float32)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """Convert translation and rotation parameters to transformation matrix.

    Args:
        vec: parameters in order of (tx, ty, tz, ex, ey, ez) if rotation mode is Euler,
        (tx, ty, tz, qx, qy, qz, qw) if rotation mode in Quaternion.
        rotation_mode: rotation mode - 'euler' or 'quat'
    Returns:
        A transformation matrix -- [4, 4]
    """
    trans = vec[:3]
    angle = vec[3:]
    if rotation_mode == 'euler':
        assert (len(angle) == 3)
        rot_mat = euler2mat(angle)

    elif rotation_mode == 'quat':
        assert (len(angle) == 4)
        rot_mat = quat2mat(angle)

    # homogenous transform
    h_mat = np.identity(4)
    h_mat[:3, :3] = rot_mat
    h_mat[:3, 3] = trans
    return h_mat


def absolute_from_relative(relative_poses):
    """Convert a sequence of relative pose estimates into a full trajectory.
    Args:
        relative_poses: relative poses -- np.array [N, 6]
    Returns:
        absolute_poses: absolute poses -- [N, 4, 4]
    """
    absolute_poses = np.zeros((relative_poses.shape[0], 4, 4))
    # construct absolute pose sequence recursively
    h_ref_tgt = np.identity(4)
    for i in range(relative_poses.shape[0]):
        h_ref_tgt = h_ref_tgt @ pose_vec2mat(relative_poses[i, :], rotation_mode='euler')
        absolute_poses[i, ...] = h_ref_tgt.copy()

    return absolute_poses


def align_trajectories(trans_gt, trans_pred):
    """Apply Horn's Closed Form method to compute the optimal rotation and translation
    which minimize the average displacements of a set of point correspondences in k-dimensions.

    Args:
        trans_gt: np.array of ground truth positions -- [3, N]
        trans_pred: np.array of predicted positions -- [3, N]
    Returns:
        trans_pred_aligned: np.array of aligned predicted trajectory -- [3, N]
    """
    # align trajectories (predictions are scale ambiguous) using Horn's Method (closed-form)
    # https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/

    # zero center positions
    trans_gt_zero_center = trans_gt - trans_gt.mean(1).reshape(3, 1)
    trans_pred_zero_center = trans_pred - trans_pred.mean(1).reshape(3, 1)

    W = np.zeros((3, 3))
    for col in range(trans_gt.shape[1]):
        W += np.outer(trans_pred_zero_center[:, col], trans_gt_zero_center[:, col])

    # singular value decomposition
    U, d, Vh = np.linalg.svd(W.transpose())
    S = np.identity(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1

    # compute aligned predicted trajectory
    rot = U @ S @ Vh
    trans = trans_gt.mean(1).reshape(3, 1) - rot @ trans_pred.mean(1).reshape(3, 1)
    trans_pred_aligned = rot @ trans_pred + trans

    return trans_pred_aligned


def scale_trajectory(trans_gt, trans_pred):
    """Compute the optimal scale factor between the predicted and ground truth
    trajectory. The scale factor is the ratio of ground-truth and estimated positions
    over the squared estimated positions.

    Args:
        trans_gt: np.array of ground truth positions -- [3, N]
        trans_pred: np.array of predicted positions -- [3, N]
    Returns:
        trans_pred_scaled: np.array of scaled predicted trajectory -- [3, N]
    """
    # align first frame and compute scale factor
    trans_pred += (trans_gt[:, 0] - trans_pred[:, 0]).reshape(3, 1)
    scale = np.sum(trans_gt * trans_pred) / np.sum(trans_pred * trans_pred)
    return trans_pred * scale


def compute_ate(gt_pose, pred_pose, tgt_idx):
    """Compute the absolute trajectory error of the predicted pose estimates
    against the ground truth UTM pose.

    Args:
        gt_pose: np.array of ground truth poses -- [N, 8].
                 Ground truth poses are represented as [t, x, y, z, qx, qy, qz, qw]
        pred_pose: np.array of predicted poses -- [M, 6].
                   Predicted poses are represented as [dx, dy, dz, dex, dey, dez] where
                   (dx, dy, dz) are the relative positions and (dex, dey, dez) are the relative euler rotations
        tgt_idx: np.array of target frame indices corresponding to ground truth poses -- [N]
    Returns:
        ate: absolute trajectory error in 3 dimensions
        ate_mean: mean of x-y-z absolute trajectory errors
        trans_gt: np.array of ground truth position sequence in UTM frame -- [3, N]
        trans_pred_scaled: np.array scaled and aligned estimated position in UTM frame -- [3, N]
    """

    # get homogenous prediction and ground truth sequences
    homogenous_pred_pose = absolute_from_relative(pred_pose)
    homogenous_pred_pose = np.linalg.inv(homogenous_pred_pose[0, ...]) @ homogenous_pred_pose

    # downsample predicted poses to align with ground truth
    homogenous_pred_pose = homogenous_pred_pose[tgt_idx]
    assert (homogenous_pred_pose.shape[0] == gt_pose.shape[0])

    # create [3, N] translational trajectories
    trans_pred = homogenous_pred_pose[:, :3, 3].T
    trans_gt = gt_pose[:, 1:4].T
    rover_to_utm = trans_gt[:, 0].reshape(3, 1)
    trans_gt = trans_gt - rover_to_utm

    # first alignment and scaling
    trans_pred_aligned = align_trajectories(trans_gt, trans_pred)
    trans_pred_scaled = scale_trajectory(trans_gt, trans_pred_aligned)

    # second alignment and scaling
    trans_pred_aligned = align_trajectories(trans_gt, trans_pred_scaled)
    trans_pred_scaled = scale_trajectory(trans_gt, trans_pred_aligned)

    # compute absolute trajectory error
    alignment_error = trans_pred_scaled - trans_gt
    ate = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0))
    ate_mean = ate.mean()

    # bring both trajectories back to utm coordinates
    trans_pred_scaled = trans_pred_scaled + rover_to_utm
    trans_gt = trans_gt + rover_to_utm

    return ate, ate_mean, trans_gt, trans_pred_scaled


def model_checkpoint(model, name, path):
    """Save the model state dictionary.
    """
    checkpoint_path = os.path.join(path, name)
    torch.save(model.state_dict(), checkpoint_path)


def generate_curve(data, labels, data_type, title, log_dir, epo=None):
    """Plot metric across epochs and save figure.
    """
    fig = plt.figure()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(data_type)
    if epo is None:
        n = data[0].shape[0]
        epochs = np.arange(1, n+1)
    else:
        n = epo
        epochs = np.arange(1, epo+1)
    for i, name in enumerate(labels):
        plt.plot(epochs, data[i][:n], label=name)
    plt.legend(loc='best')
    plt.savefig(log_dir + '/{}_curves.png'.format(data_type))
    plt.close(fig)


def generate_ate_curve(ate_xyz, log_dir):
    """Plot ATE in x-y-z directions across epochs in one figure.
    Args:
        ate_xyz: np.array of x-y-z ATE scores for each epoch -- [N, 3]
    """
    t = np.arange(1, ate_xyz.shape[0])
    fig, axes = plt.subplots(3)
    fig.suptitle("Val ATE in 3-Dimensions over Epochs")
    axes[0].plot(t, ate_xyz[:, 0])
    axes[1].plot(t, ate_xyz[:, 1], 'tab:orange')
    axes[2].plot(t, ate_xyz[:, 2], 'tab:red')

    # set labels and label outer for epochs
    axes[0].set_ylabel('ATE x-dim')
    axes[1].set_ylabel('ATE y-dim')
    axes[2].set_ylabel('ATE z-dim')
    axes[0].set_xlabel('Epochs')
    axes[1].set_xlabel('Epochs')
    axes[2].set_xlabel('Epochs')
    for ax in axes.flat:
        ax.label_outer()

    fig.savefig(log_dir + '/ATE_3dim.png')
    plt.close(fig)


class Visualizer:

    def __init__(self, output_dir, device):
        """Visualization class for disparities / depths, images, warped images, and trajectories.
        """
        self.device = device
        self.output_dir = output_dir + '/visuals'
        self.traj_dir = self.output_dir + '/trajectories'
        os.mkdir(self.output_dir)
        os.mkdir(self.traj_dir)

    def generate_trajectory(self, traj, label, name, epo, split):
        """Generate visualization of trajectory in bird's eye view frame (x-y plane).
        Args:
            traj: np.array of trajectory -- [3, N]
            label: label for the filename -- str
            name: name of the trajectory -- str
            epo: current epoch -- int
            split: data split -- str
        """
        color = np.arange(traj.shape[1]) / traj.shape[1]
        cmap = plt.get_cmap('viridis')

        # plot x-y plane trajectory
        fig = plt.figure()
        plt.scatter(traj[0, :], traj[1, :], cmap=cmap, c=color, label=label, s=1.75)
        plt.title("{} Trajectory in UTM Frame".format(name))
        plt.xlabel("Easting [m]")
        plt.ylabel("Northing [m]")
        plt.legend(loc='best')
        fig.savefig(self.traj_dir + '/epo{}_{}_{}_traj.png'.format(epo, split, label))
        plt.close(fig)

    def generate_trajectories(self, gt_traj, pred_traj, epo, split):
        """Generate visualization of ground truth trajectory and predicted trajectory in bird's
        eye view frame (x-y plane) which have been aligned in terms of scale, rotation, and translation.

        Args:
            gt_traj: np.array of ground truth position trajectory -- [3, N]
            pred_traj: np.array of predicted position trajectory -- [3, N]
            epo: current epoch -- int
            split: data split -- str
        """
        assert (pred_traj.shape == gt_traj.shape)
        color = np.arange(gt_traj.shape[1]) / gt_traj.shape[1]

        fig = plt.figure()
        # ground truth blue color scheme - prediction red color scheme
        cmap_gt = plt.get_cmap('winter')
        cmap_pred = plt.get_cmap('autumn')

        # overlay ground truth and predicted trajectories
        plt.scatter(gt_traj[0, :], gt_traj[1, :], cmap=cmap_gt, c=color, label='gt', s=1.75)
        plt.scatter(pred_traj[0, :], pred_traj[1, :], cmap=cmap_pred, c=color, label='pred', s=1.75)
        plt.title("Ground Truth vs Predicted Trajectory in UTM Frame")
        plt.xlabel("Easting [m]")
        plt.ylabel("Northing [m]")
        plt.legend(loc='best')
        fig.savefig(self.traj_dir + '/epo{}_{}_traj_overlap.png'.format(epo, split))
        plt.close(fig)

    def generate_random_visuals(self, disp_net, pose_net, dataloader, criterion, sample_size, epo, split, skip=1):
        """Randomly selects samples from a dataloader to produce visualizations of predicted
        disparity maps, inverse warped target frames. Also save original reference and target frames.
        Args:
            disp_net: PyTorch disparity network
            pose_net: PyTorch pose network
            dataloader: PyTorch dataloader
            criterion: required to compute inverse warped images
            sample_size: number of samples to consider -- int
            epo: epoch number -- int
            split: data split -- str
            skip: if the dataloader is not shuffled, skip samples for more variety -- int
        """
        disp_net.eval()
        pose_net.eval()

        # store samples and predictions
        samples = {'disp': [], 'tgt_img': [], 'ref_img': [], 'warp_img': []}
        for i, sample in enumerate(dataloader, 0):
            # skip sample
            if i > sample_size * skip:
                break
            elif i % skip != 0:
                continue

            tgt_img, ref_imgs = sample
            tgt_img = tgt_img.to(self.device)
            ref_imgs = [ref_img.to(self.device) for ref_img in ref_imgs]

            # predict disparities
            disparities = [disp_net(tgt_img)]
            depth = [1 / disp for disp in disparities]

            # predict poses
            _, poses = pose_net(tgt_img, ref_imgs)

            # compute photometric loss and smoothness loss
            view_synthesis_loss, warped_imgs, diff_imgs = \
                criterion.photometric_reconstruction_loss(tgt_img, depth, ref_imgs, poses)

            # extract sample images for visualization
            disp = self.normalize_depth_for_display(self.get_detach(depth[0][0, 0]))    # [1, B, 1, H, W]
            samples['disp'].append(disp)
            samples['tgt_img'].append(self.get_detach(tgt_img[0, 0]) * 255.0)           # [B, 1, H, W]
            samples['ref_img'].append(self.get_detach(ref_imgs[0][0, 0]) * 255.0)       # [Seq, B, 1, H, W]
            samples['warp_img'].append(self.get_detach(warped_imgs[0][0][0]) * 255.0)   # [1, Seq, 1, H, W]

        self.save_samples(samples, epo, split)
        disp_net.train()
        pose_net.train()

    def generate_random_visuals_depth(self, disp_net, dataloader, criterion, sample_size, epo, split):
        """Randomly selects samples from a dataloader to produce visuals of disparity maps.
        Args:
            disp_net: PyTorch disparity network
            dataloader: PyTorch dataloader
            criterion: required to compute inverse warped images
            sample_size: number of samples to consider -- int
            epo: epoch number -- int
            split: data split -- str
        """
        disp_net.eval()

        # store samples and predictions
        samples = {'disp': [], 'tgt_img': [], 'ref_img': [], 'warp_img': []}
        for i, sample in enumerate(dataloader, 0):
            if i > sample_size:
                break

            tgt_img, ref_img, pose = sample
            tgt_img = tgt_img.to(self.device)
            ref_img = ref_img.to(self.device)
            pose = pose.to(self.device)

            # predict disparities at multiple scale spaces with DispNet
            disparities = [disp_net(tgt_img)]
            depth = [1 / disp for disp in disparities]

            # compute photometric loss and smoothness loss
            view_synthesis_loss, warped_imgs, diff_imgs = \
                criterion.inverse_warp_loss(tgt_img, depth, ref_img, pose)

            # extract sample figures
            disp = self.normalize_depth_for_display(self.get_detach(depth[0][0, 0]))  # [1, B, 1, H, W]
            samples['disp'].append(disp)                                              
            samples['tgt_img'].append(self.get_detach(tgt_img[0, 0]) * 255.0)         # [B, 1, H, W]
            samples['ref_img'].append(self.get_detach(ref_img[0, 0]) * 255.0)         # [B, 1, H, W]
            samples['warp_img'].append(self.get_detach(warped_imgs[0][0]) * 255.0)    # [S, 1, H, W]

        self.save_samples(samples, epo, split)
        disp_net.train()

    def save_samples(self, samples, epo, split):
        """Saves samples to output directory.
        Args:
            samples: dict<str, list[np.array]> of sample types which hash to a list of images
            epo: epoch number -- int
            split: data split -- str
        """
        save_dir = os.path.join(self.output_dir, 'epo' + str(epo))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for img_type in samples:
            img_samples = samples[img_type]
            for i, img in enumerate(img_samples):
                save_path = save_dir + '/' + split + '_s' + str(i) + '_' + img_type
                cv2.imwrite(save_path + '.png', img)

    @staticmethod
    def normalize_depth_for_display(depth, pc=95, normalizer=None, cmap='gray'):
        # convert to disparity
        depth = 1./(depth + 1e-6)
        if normalizer is not None:
            depth = depth/normalizer
        else:
            depth = depth/(np.percentile(depth, pc) + 1e-6)
        depth = np.clip(depth, 0, 1)
        return depth * 255.0

    @staticmethod
    def get_detach(sample):
        """Convert tensor to detached numpy array in CPU.
        """
        return sample.detach().cpu().numpy()

    @staticmethod
    def gray2rgb(im, cmap='gray'):
        cmap = plt.get_cmap(cmap)
        rgba_img = cmap(im.astype(np.float32))
        rgb_img = np.delete(rgba_img, 3, 2)
        return rgb_img
