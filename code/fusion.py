'''
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
'''
import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import quaternion  # pip install numpy-quaternion

import transforms
import o3d_utility

from preprocess import load_gt_poses


class Map:
    def __init__(self):
        self.points = np.empty((0, 3))
        self.normals = np.empty((0, 3))
        self.colors = np.empty((0, 3))
        self.weights = np.empty((0, 1))
        self.last_updated = np.empty((0, 1)) # for filtering
        self.initialized = False

    def merge(self, indices, points, normals, colors, R, t):
        '''
        TODO: implement the merge function
        \param self The current maintained map
        \param indices Indices of selected points. Used for IN PLACE modification.
        \param points Input associated points, (N, 3)
        \param normals Input associated normals, (N, 3)
        \param colors Input associated colors, (N, 3)
        \param R rotation from camera (input) to world (map), (3, 3)
        \param t translation from camera (input) to world (map), (3, )
        \return None, update map properties IN PLACE
        '''

        # Transforming the input points and normals to the map coordinate system
        t_points = (R @ points.T + t).T
        t_normals = (R @ normals.T).T

        cur_points = self.points[indices] 
        cur_normals = self.normals[indices]
        cur_weights = self.weights[indices]
        cur_colors = self.colors[indices]
        
        # P and Q are the points and normals of the current map and the input map, c is confidence control / weights
        p = cur_points
        q = t_points
        n_p = cur_normals
        n_q = t_normals
        c = cur_weights
        r =  cur_colors

        alpha = np.ones_like(c)


        # Updating p
        p = ((c * p) + alpha * q) / (alpha+c)

        # Updating colors
        r = ((c * r) + alpha * colors) / (alpha+c)

        # Updating n_p
        n_p = ((c * n_p) + alpha * n_q) / (alpha+c)

        # Normalizing n_p
        n_p = n_p / np.linalg.norm(n_p, axis=1).reshape(-1, 1)

        

        # Updating c
        c = c + alpha

        # Assigning the updated values

        self.points[indices] = p
        self.normals[indices] = n_p
        self.weights[indices] = c
        self.colors[indices] = r

        pass

    def add(self, points, normals, colors, R, t, current_frame_index=0):
        '''
        TODO: implement the add function
        \param self The current maintained map
        \param points Input associated points, (N, 3)
        \param normals Input associated normals, (N, 3)
        \param colors Input associated colors, (N, 3)
        \param R rotation from camera (input) to world (map), (3, 3)
        \param t translation from camera (input) to world (map), (3, )
        \return None, update map properties by concatenation
        '''

        # Transforming the input points and normals to the map coordinate system
        t_points = (R @ points.T + t).T
        t_normals = (R @ normals.T).T

        # Concatenating the input points, normals, colors to the current map
        self.points = np.concatenate((self.points, t_points))
        self.normals = np.concatenate((self.normals, t_normals))
        self.colors = np.concatenate((self.colors, colors))

        # Assigning the weights
        new_weights = np.ones((len(points), 1))
        self.weights = np.concatenate((self.weights, new_weights))

        # Assigning the last updated frame index
        self.last_updated = np.concatenate((self.last_updated, np.full((len(points), 1), current_frame_index)))

        pass




    def filter_pass1(self, us, vs, ds, h, w):
        '''
        TODO: implement the filter function
        \param self The current maintained map, unused
        \param us Putative corresponding u coordinates on an image, (N, 1)
        \param vs Putative corresponding v coordinates on an image, (N, 1)
        \param ds Putative corresponding d depth on an image, (N, 1)
        \param h Height of the image projected to
        \param w Width of the image projected to
        \return mask (N, 1) in bool indicating the valid coordinates
        '''

        mask = np.zeros_like(us).astype(bool)
        valid_bounds = (us >= 0) & (us < w) & (vs >= 0) & (vs < h) & (ds > 0)
        mask = valid_bounds


        return mask

    def filter_pass2(self, points, normals, input_points, input_normals,
                     dist_diff, angle_diff):
        '''
        TODO: implement the filter function
        \param self The current maintained map, unused
        \param points Maintained associated points, (M, 3)
        \param normals Maintained associated normals, (M, 3)
        \param input_points Input associated points, (M, 3)
        \param input_normals Input associated normals, (M, 3)
        \param dist_diff Distance difference threshold to filter correspondences by positions
        \param angle_diff Angle difference threshold to filter correspondences by normals
        \return mask (N, 1) in bool indicating the valid correspondences
        '''
        mask = np.zeros((len(points))).astype(bool)
        dist = np.linalg.norm(input_points - points, axis=1)
        valid_dist = dist < dist_diff
        # angle_threshold = np.cos(np.deg2rad(angle_diff))
        angle_threshold = np.cos(angle_diff)
        angle_dif = np.sum(normals * input_normals, axis=1)
        valid_angle = angle_dif > angle_threshold

        mask = valid_dist & valid_angle


        return mask
    
    def remove_dormant_points(self, indices):
        '''
        \param self The current maintained map
        \param indices Indices of dormant points to be removed
        '''

        self.points = np.delete(self.points, indices, axis=0)
        self.normals = np.delete(self.normals, indices, axis=0)
        self.colors = np.delete(self.colors, indices, axis=0)
        self.weights = np.delete(self.weights, indices, axis=0)
        self.last_updated = np.delete(self.last_updated, indices, axis=0)

    def fuse(self,
             vertex_map,
             normal_map,
             color_map,
             intrinsic,
             T,
             dist_diff=0.03,
             angle_diff=np.deg2rad(5),
             current_frame_index=0):
        '''
        \param self The current maintained map
        \param vertex_map Input vertex map, (H, W, 3)
        \param normal_map Input normal map, (H, W, 3)
        \param intrinsic Intrinsic matrix, (3, 3)
        \param T transformation from camera (input) to world (map), (4, 4)
        \return None, update map properties on demand
        '''
        # Camera to world
        R = T[:3, :3]
        t = T[:3, 3:]

        # World to camera
        T_inv = np.linalg.inv(T)
        R_inv = T_inv[:3, :3]
        t_inv = T_inv[:3, 3:]

        if not self.initialized:
            points = vertex_map.reshape((-1, 3))
            normals = normal_map.reshape((-1, 3))
            colors = color_map.reshape((-1, 3))

            # TODO: add step
            self.add(points, normals, colors, R, t)
            self.initialized = True

        else:
            h, w, _ = vertex_map.shape

            # Transform from world to camera for projective association
            indices = np.arange(len(self.points)).astype(int)
            T_points = (R_inv @ self.points.T + t_inv).T
            R_normals = (R_inv @ self.normals.T).T

            # Projective association
            us, vs, ds = transforms.project(T_points, intrinsic)
            us = np.round(us).astype(int)
            vs = np.round(vs).astype(int)

            # TODO: first filter: valid projection
            mask = self.filter_pass1(us, vs, ds, h, w)
            # Should not happen -- placeholder before implementation
            if mask.sum() == 0:
                return
            # End of TODO

            indices = indices[mask]
            us = us[mask]
            vs = vs[mask]

            T_points = T_points[indices]
            R_normals = R_normals[indices]
            valid_points = vertex_map[vs, us]
            valid_normals = normal_map[vs, us]

            # TODO: second filter: apply thresholds
            mask = self.filter_pass2(T_points, R_normals, valid_points,
                                     valid_normals, dist_diff, angle_diff)
            # Should not happen -- placeholder before implementation
            if mask.sum() == 0:
                return
            # End of TODO

            indices = indices[mask]
            us = us[mask]
            vs = vs[mask]

            updated_entries = len(indices)

            merged_points = vertex_map[vs, us]
            merged_normals = normal_map[vs, us]
            merged_colors = color_map[vs, us]

            # TODO: Merge step - compute weight average after transformation
            self.merge(indices, merged_points, merged_normals, merged_colors,
                       R, t)
            # End of TODO

            associated_mask = np.zeros((h, w)).astype(bool)
            associated_mask[vs, us] = True
            new_points = vertex_map[~associated_mask]
            new_normals = normal_map[~associated_mask]
            new_colors = color_map[~associated_mask]

            # Keeping track of the last updated frame index for dormant point removal
            if indices.size > 0:
                # print(indices)
                self.last_updated[indices] = current_frame_index

            # Removing dormant points if they have not been updated for some frame count
            dormant_indices = np.where(current_frame_index - self.last_updated > 10)[0]
            if len(dormant_indices) > 0:
                self.remove_dormant_points(dormant_indices)
                print('Removed {} dormant points'.format(len(dormant_indices)))

            # TODO: Add step
            self.add(new_points, new_normals, new_colors, R, t, current_frame_index=current_frame_index)
            # End of TODO

            added_entries = len(new_points)
            print('updated: {}, added: {}, total: {}'.format(
                updated_entries, added_entries, len(self.points)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path', help='path to the dataset folder containing rgb/ and depth/',
        default='C:\CMU_Spring\SLAM_HW_ICP\living_room_traj2_frei_png')
    parser.add_argument('--start_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=1)
    parser.add_argument('--end_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=200)
    parser.add_argument('--downsample_factor', type=int, default=2)
    args = parser.parse_args()

    intrinsic_struct = o3d.io.read_pinhole_camera_intrinsic('code\intrinsics.json')
    intrinsic = np.array(intrinsic_struct.intrinsic_matrix)
    indices, gt_poses = load_gt_poses(
        os.path.join(args.path, 'livingRoom2.gt.freiburg'))
    # TUM convention
    depth_scale = 5000.0

    rgb_path = os.path.join(args.path, 'rgb')
    depth_path = os.path.join(args.path, 'depth')
    normal_path = os.path.join(args.path, 'normal')

    m = Map()

    down_factor = args.downsample_factor
    intrinsic /= down_factor
    intrinsic[2, 2] = 1

    for i in range(args.start_idx, args.end_idx + 1):
        print('Fusing frame {:03d}'.format(i))
        source_depth = o3d.io.read_image('{}/{}.png'.format(depth_path, i))
        source_depth = np.asarray(source_depth) / depth_scale
        source_depth = source_depth[::down_factor, ::down_factor]
        source_vertex_map = transforms.unproject(source_depth, intrinsic)

        source_color_map = np.asarray(
            o3d.io.read_image('{}/{}.png'.format(rgb_path,
                                                 i))).astype(float) / 255.0
        source_color_map = source_color_map[::down_factor, ::down_factor]

        source_normal_map = np.load('{}/{}.npy'.format(normal_path, i))
        source_normal_map = source_normal_map[::down_factor, ::down_factor]

        m.fuse(source_vertex_map, source_normal_map, source_color_map,
               intrinsic, gt_poses[i] , current_frame_index=i)

    global_pcd = o3d_utility.make_point_cloud(m.points,
                                              colors=m.colors,
                                              normals=m.normals)
    o3d.visualization.draw_geometries(
        [global_pcd.transform(o3d_utility.flip_transform)])
