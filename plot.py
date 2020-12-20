import numpy as np
from utils import generate_curve


if __name__ == '__main__':
    # manually create plots (in case terminated training early)
    log_path = 'exp/joint_depth_pose/logs'
    train_loss = np.loadtxt(log_path + '/train_loss.txt')
    val_loss = np.loadtxt(log_path + '/val_loss.txt')
    val_ate = np.loadtxt(log_path + '/val_ate.txt')

    # generate metric curves
    generate_curve([train_loss[:, 0], val_loss[:, 0]], ['train', 'val'], 'loss',
                   'Train vs Val Combined Loss', log_path, 40)
    generate_curve([train_loss[:, 1], val_loss[:, 1]], ['train', 'val'], 'photometric loss',
                   'Train vs Val Photometric Reconstruction Loss', log_path, 40)
    generate_curve([train_loss[:, 2], val_loss[:, 2]], ['train', 'val'], 'depth smooth loss',
                   'Train vs Val Depth Smoothness Loss', log_path, 40)
    generate_curve([val_ate], ['val'], 'ATE', 'Validation Absolute Trajectory Error', log_path, 40)
