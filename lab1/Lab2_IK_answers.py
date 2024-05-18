import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, _, path1, path2 = meta_data.get_path_from_root_to_end()
    joint_rotations = []
    original_rotations = []
    path_set = set()
    n = len(path)
    for i in range(n):
        path_set.add(path[i])
    for i in range(len(meta_data.joint_name)):
        joint_rotations.append(R.from_quat(joint_orientations[i]))
        original_rotations.append(joint_rotations[meta_data.joint_parent[i]].inv() * joint_rotations[i])
    cnt = 0
    end = path[-1]
    # end_parent = meta_data.joint_parent[end]
    # joint_rotations[end_parent] = R.from_euler('XYZ', [0, 0, 0])
    # joint_positions[end] = joint_positions[end_parent] + meta_data.joint_initial_position[end] - meta_data.joint_initial_position[end_parent]
    while cnt < 600:
        for i in range(n - 2, -1, -1):
            parent = path[i]
            if parent == 0:
                break
            diff1 = target_pose - joint_positions[parent]
            diff2 = joint_positions[end] - joint_positions[parent]
            diff1 = diff1 / np.linalg.norm(diff1)
            diff2 = diff2 / np.linalg.norm(diff2)
            axis = np.cross(diff2, diff1)
            angle = np.dot(diff1, diff2)
            # angle = max(0, angle)
            rotation = R.from_rotvec(axis * np.arccos(angle))
            for j in range(i, n - 1):
                index = path[j]
                joint_rotations[index] = rotation * joint_rotations[index]
                offset = meta_data.joint_initial_position[path[j + 1]] - meta_data.joint_initial_position[index]
                joint_positions[path[j + 1]] = joint_positions[index] + joint_rotations[index].apply(offset)
        len2 = len(path2) - 1
        for i in range(0, len2):
            parent = path2[i]
            diff1 = target_pose - joint_positions[parent]
            diff2 = joint_positions[end] - joint_positions[parent]
            diff1 = diff1 / np.linalg.norm(diff1)
            diff2 = diff2 / np.linalg.norm(diff2)
            axis = np.cross(diff2, diff1)
            angle = np.dot(diff1, diff2)
            rotation = R.from_rotvec(axis * np.arccos(angle))
            inv_rotation = R.from_rotvec(axis * np.arccos(angle)).inv()
            for j in range(i + 1, len(path2)):
                index = path2[j]
                joint_rotations[index] = rotation * joint_rotations[index]
                offset = meta_data.joint_initial_position[path2[j - 1]] - meta_data.joint_initial_position[index]
                joint_positions[index] = joint_positions[path2[j - 1]] - joint_rotations[index].apply(offset)
            for j in range(len2, n - 1):
                index = path[j]
                if index != 0:
                    joint_rotations[index] = rotation * joint_rotations[index]
                offset = meta_data.joint_initial_position[path[j + 1]] - meta_data.joint_initial_position[index]
                joint_positions[path[j + 1]] = joint_positions[index] + joint_rotations[index].apply(offset)
            # if i == 1:
            #     break
        dis = np.linalg.norm(joint_positions[end] - target_pose)
        # print(dis)
        if dis < 0.02:
            print(cnt)
            break
        cnt += 1
    for i in range(1, len(meta_data.joint_name)):
        if i not in path_set:
            rotation = joint_rotations[meta_data.joint_parent[i]]
            offset = meta_data.joint_initial_position[i] - meta_data.joint_initial_position[meta_data.joint_parent[i]]
            joint_positions[i] = joint_positions[meta_data.joint_parent[i]] + rotation.apply(offset)
            joint_rotations[i] = rotation * original_rotations[i]
        joint_orientations[i] = joint_rotations[i].as_quat()


    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    root_pose = joint_positions[0]
    rarget_pose = np.array([root_pose[0] + relative_x, target_height, root_pose[2] + relative_z])
    return part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, rarget_pose)
    # return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    return joint_positions, joint_orientations