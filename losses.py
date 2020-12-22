import torch
import torch.nn.functional as F
from SfMLearnerMars.image_warping import ImageWarping
from SfMLearnerMars.utils import intrinsics


# Much of this code was adapted from: https://github.com/ClementPinard/SfmLearner-Pytorch
class ViewSynthesisLoss:

    def __init__(self, device, rotation_mode='euler', padding_mode='zeros', scale=4):
        """Custom loss class implementing differentiable photometric reconstruction
        loss (view synthesis) and smoothness loss (depth).
        """
        self.device = device
        self.warper = ImageWarping(rotation_mode, padding_mode)
        # downscale intrinsic matrix parameters to match dataloader transforms
        intrinsic = torch.from_numpy(intrinsics).to(device)
        self.intrinsic = torch.cat((intrinsic[:2, :] / scale, intrinsic[2:, :]), dim=0)

    def photometric_reconstruction_loss(self, tgt_img, depth, ref_imgs, poses):
        """Compute photometric reconstruction loss between reference images and a target image via
        view-synthesis framework. Loss is computed at multiple scale spaces.

        Args:
            tgt_img: target frame image -- [B, 1, H, W]
            depth: predicted depths -- [S, B, 1, H, W] - S is scale space
            ref_imgs: reference frame images -- [Seq, B, 1, H, W] - Seq is the sequence length
            poses: predicted poses -- [B, Seq, 6] - Seq is the sequence length
        Returns:
            reconstruction_loss: novel view synthesis loss
            warped_imgs: warped reference frame images for visualization
        """

        def one_scale(scale_depth):
            """Compute photometric loss at a single scale space."""

            # interpolate target and reference images to match scale space
            b, _, h, w = scale_depth.size()
            tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
            ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]

            # downscale intrinsic to match scale space
            downscale = tgt_img.size(2) / h
            intrinsic = self.intrinsic.clone().to(self.device)
            intrinsic_scaled = torch.cat((intrinsic[:2, :] / downscale, intrinsic[2:, :]), dim=0)

            warped_imgs = []
            diff_maps = []
            reconstruction_loss = 0

            # compute loss over each reference-target pair in the sequence
            for i, ref_img in enumerate(ref_imgs_scaled):
                pose = poses[:, i]  # [B, 2, 6] -> [B, 6]
                # batch inverse warp
                ref_warped_img, valid_points = self.warper.inverse_warp(
                    scale_depth[:, 0], ref_img, pose, intrinsic_scaled
                )
                assert (ref_warped_img.size() == tgt_img_scaled.size())

                # compute absolute pixel difference on valid points only
                masked_pixel_error = (tgt_img_scaled - ref_warped_img) * valid_points.unsqueeze(1).float()
                reconstruction_loss += masked_pixel_error.abs().mean()
                assert ((reconstruction_loss == reconstruction_loss).item() == 1)

                # store first in each batch
                warped_imgs.append(ref_warped_img[0])
                diff_maps.append(masked_pixel_error[0])

            return reconstruction_loss, warped_imgs, diff_maps

        # compute loss across batch for each scale space
        warped_results, diff_results = [], []
        total_loss = 0
        for d in depth:
            loss, warped, diff = one_scale(d)
            total_loss += loss
            warped_results.append(warped)
            diff_results.append(diff)

        return total_loss, warped_results, diff_results

    def inverse_warp_loss(self, tgt_img, depth, ref_img, poses):
        """Compute photometric reconstruction loss between a reference image and a target image via
        view-synthesis framework. Loss is computed at multiple scale spaces.

        Args:
            tgt_img: target frame image -- [B, 1, H, W]
            depth: predicted depths -- [S, B, 1, H, W] - S is scale space
            ref_img: reference frame images -- [B, 1, H, W] - Seq is the sequence length
            poses: ground truth homogenous pose -- [B, 4, 4] - Seq is the sequence length
        Returns:
            reconstruction_loss: novel view synthesis loss
            warped_imgs: warped reference frame images for visualization
        """

        def one_scale(scale_depth):
            """Compute photometric loss at a single scale space."""

            # interpolate target and reference images to match scale space
            b, _, h, w = scale_depth.size()
            tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
            ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area')

            # downscale intrinsic to match scale space
            downscale = tgt_img.size(2) / h
            intrinsic = self.intrinsic.clone().to(self.device)
            intrinsic_scaled = torch.cat((intrinsic[:2, :] / downscale, intrinsic[2:, :]), dim=0)

            # compute photometric reconstruction loss
            # batch inverse warp
            ref_warped_img, valid_points = self.warper.inverse_warp_gt_pose(
                scale_depth[:, 0], ref_img_scaled, poses, intrinsic_scaled
            )
            assert (ref_warped_img.size() == tgt_img_scaled.size())

            # compute absolute pixel difference on valid points only
            masked_pixel_error = (tgt_img_scaled - ref_warped_img) * valid_points.unsqueeze(1).float()
            reconstruction_loss = masked_pixel_error.abs().mean()
            assert ((reconstruction_loss == reconstruction_loss).item() == 1)

            return reconstruction_loss, ref_warped_img[0], masked_pixel_error[0]

        # compute loss across batch for each scale space
        warped_results, diff_results = [], []
        total_loss = 0
        for d in depth:
            loss, warped, diff = one_scale(d)
            total_loss += loss
            warped_results.append(warped)
            diff_results.append(diff)

        return total_loss, warped_results, diff_results

    def smoothness_loss(self, pred_depth):
        """Compute smoothness loss over depth image.

        Args:
            pred_depth: predicted depth maps -- [S, B, 1, H, W]
        """
        def gradient(pred):
            """Compute pixel gradients in x and y direction"""
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        loss = 0
        weight = 1.

        # compute smoothness loss for depth maps in each scale space
        for scaled_map in pred_depth:
            dx, dy = gradient(scaled_map)
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight
            weight /= 2.3

        return loss
