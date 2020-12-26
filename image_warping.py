import torch
import torch.nn.functional as F


# This code was adapted from: https://github.com/ClementPinard/SfmLearner-Pytorch
class ImageWarping:

    def __init__(self, rotation_mode='euler', padding_mode='zeros'):
        self.pixel_coords = None
        self.rotation_mode = rotation_mode
        self.padding_mode = padding_mode

    def inverse_warp(self, depth, ref_img, pose, intrinsic):
        """Inverse warp of pixels in the target image frame into the reference frame based on
        predicted depths and poses, bilinear interpolate the reference image pixel values to
        acquire reconstruction.

        Args:
            depth: predicted depth maps [B, H, W]
            ref_img: reference images [B, 1, H, W]
            pose: predicted pose [B, 6]
            intrinsic: intrinsic matrix [3, 3]
        """
        check_sizes(depth, 'depth', 'BHW')
        check_sizes(ref_img, 'img', 'B1HW')
        check_sizes(pose, 'pose', 'B6')

        # create new pixel coordinates to match scale space (e.g. size) of depth map
        self.pixel_coords = self.create_pixel_grid(depth)
        intrinsic_inv = intrinsic.inverse()

        # inverse projection
        cam_coords = self.pixel2cam(depth, intrinsic_inv)  # [B, 3, H, W]

        # get homogenous transformation
        pose_mat = self.pose_vec2mat(pose)  # [B, 3, 4]

        # get projection matrix for tgt camera frame to source pixel frame
        proj_cam_to_src_pixel = intrinsic.unsqueeze(0) @ pose_mat  # [B, 3, 4]

        # acquire reference frame coordinates
        rot, tr = proj_cam_to_src_pixel[..., :3], proj_cam_to_src_pixel[..., -1:]
        src_pixel_coords = self.cam2pixel(cam_coords, rot, tr)  # [B, H, W, 2]

        # interpolate pixel values in reference image and acquire valid pixel mask
        projected_img = F.grid_sample(ref_img, src_pixel_coords, padding_mode=self.padding_mode, align_corners=True)
        valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

        return projected_img, valid_points

    def inverse_warp_gt_pose(self, depth, ref_img, pose, intrinsic):
        """Inverse warp of pixels in the target image frame into the reference frame based on
        predicted depths and poses, bilinear interpolate the reference image pixel values to
        acquire reconstruction.

        Args:
            depth: predicted depth maps [B, H, W]
            ref_img: reference images [B, 1, H, W]
            pose: ground truth pose [B, 4, 4]
            intrinsic: intrinsic matrix [3, 3]
        """
        check_sizes(depth, 'depth', 'BHW')
        check_sizes(ref_img, 'img', 'B1HW')

        # create new pixel coordinates to match scale space (e.g. size) of depth map
        self.pixel_coords = self.create_pixel_grid(depth)
        intrinsic_inv = intrinsic.inverse()

        # inverse projection
        cam_coords = self.pixel2cam(depth, intrinsic_inv)  # [B, 3, H, W]

        # get homogenous transformation
        pose_mat = pose[:, :3, :]  # [B, 3, 4]

        # get projection matrix for tgt camera frame to source pixel frame
        proj_cam_to_src_pixel = intrinsic.unsqueeze(0) @ pose_mat  # [B, 3, 4]

        # acquire reference frame coordinates
        rot, tr = proj_cam_to_src_pixel[..., :3], proj_cam_to_src_pixel[..., -1:]
        src_pixel_coords = self.cam2pixel(cam_coords, rot, tr)  # [B, H, W, 2]

        # interpolate pixel values in reference image and acquire valid pixel mask
        projected_img = F.grid_sample(ref_img, src_pixel_coords, padding_mode=self.padding_mode, align_corners=True)
        valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

        return projected_img, valid_points

    def pixel2cam(self, depth, intrinsic_inv):
        """Transform coordinates in the pixel frame to the camera frame.
        Args:
            depth: depth maps [B, H, W]
            intrinsic_inv: [3, 3]
        Returns:
            array of (u,v,1) cam coordinates [B, 3, H, W]
        """

        b, h, w = depth.size()
        current_pixel_coords = self.pixel_coords[..., :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
        cam_coords = (intrinsic_inv.unsqueeze(0) @ current_pixel_coords).reshape(b, 3, h, w)
        return cam_coords * depth.unsqueeze(1)

    def cam2pixel(self, cam_coords, proj_c2p_rot, proj_c2p_tr):
        """Transform coordinates in the camera frame to the pixel frame.
        Args:
            cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
            proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
            proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
        Returns:
            array of [-1,1] coordinates -- [B, 2, H, W]
        """
        b, _, h, w = cam_coords.size()
        cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
        if proj_c2p_rot is not None:
            pcoords = proj_c2p_rot @ cam_coords_flat
        else:
            pcoords = cam_coords_flat

        if proj_c2p_tr is not None:
            pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
        X = pcoords[:, 0]
        Y = pcoords[:, 1]
        Z = pcoords[:, 2].clamp(min=1e-3)

        # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        X_norm = 2 * (X / Z) / (w - 1) - 1
        Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]

        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
        return pixel_coords.reshape(b, h, w, 2)

    def pose_vec2mat(self, vec):
        """Convert 6DoF parameters to transformation matrix.
        Args:
            vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
        Returns:
            A transformation matrix -- [B, 3, 4]
        """
        translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
        rot = vec[:, 3:]

        rot_mat = None
        if self.rotation_mode == 'euler':
            rot_mat = self.euler2mat(rot)  # [B, 3, 3]
        elif self.rotation_mode == 'quat':
            rot_mat = self.quat2mat(rot)  # [B, 3, 3]

        transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
        return transform_mat

    @staticmethod
    def create_pixel_grid(depth):
        """Create pixel grid corresponding to depth map size.
        Args:
            depth: depth maps [B, H, W]
        """
        _, h, w = depth.size()
        i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
        j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
        ones = torch.ones(1, h, w).type_as(depth)
        pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
        return pixel_coords

    @staticmethod
    def euler2mat(angle):
        """Convert euler angles to rotation matrix.

         Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

        Args:
            angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
        Returns:
            Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
        """
        B = angle.size(0)
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

        cosz = torch.cos(z)
        sinz = torch.sin(z)

        zeros = z.detach() * 0
        ones = zeros.detach() + 1
        zmat = torch.stack([cosz, -sinz, zeros,
                            sinz, cosz, zeros,
                            zeros, zeros, ones], dim=1).reshape(B, 3, 3)

        cosy = torch.cos(y)
        siny = torch.sin(y)

        ymat = torch.stack([cosy, zeros, siny,
                            zeros, ones, zeros,
                            -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

        cosx = torch.cos(x)
        sinx = torch.sin(x)

        xmat = torch.stack([ones, zeros, zeros,
                            zeros, cosx, -sinx,
                            zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

        rotMat = xmat @ ymat @ zmat
        return rotMat

    @staticmethod
    def quat2mat(quat):
        """Convert quaternion coefficients to rotation matrix.

        Args:
            quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
        """
        norm_quat = torch.cat([quat[:, :1].detach() * 0 + 1, quat], dim=1)
        norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

        B = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                              2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                              2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
        return rotMat


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))
