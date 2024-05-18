import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data

def parse_bvh(file, joint_name : list, joint_parent : list, joint_offset : list, parent : int, index : int, name):
    # line = file.readline().split()
    # name = line[1]
    joint_name.append(name)
    joint_parent.append(parent)
    this_index = index
    file.readline()
    line = file.readline().split()
    offset = np.array([float(x) for x in line[1:]])
    joint_offset.append(offset)
    file.readline()
    while True :
        line = file.readline().split()
        if line[0] == "}":
            break
        elif line[0] == "End":
            file.readline()
            line = file.readline().split()
            offset = np.array([float(x) for x in line[1:]])
            joint_offset.append(offset)
            joint_name.append(name + "_end")
            joint_parent.append(this_index)
            index += 1
            file.readline()
            file.readline()
            break
        else:
            index = parse_bvh(file, joint_name, joint_parent, joint_offset, this_index, index + 1, line[1])
    return index


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    file = open(bvh_file_path, 'r')
    file.readline()
    file.readline()
    root_name = "RootJoint"
    parse_bvh(file, joint_name, joint_parent, joint_offset, -1, 0, root_name)
    return joint_name, joint_parent, np.array(joint_offset)


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []
    joint_rotations = []
    motion = motion_data[frame_id]
    joint_positions.append(motion[:3])
    rotation = R.from_euler('XYZ', motion[3:6], degrees=True)
    joint_rotations.append(rotation)
    joint_orientations.append(rotation.as_quat())
    index = 6
    for i in range(1, len(joint_name)):
        parent = joint_parent[i]
        position = joint_positions[parent] + joint_rotations[parent].apply(joint_offset[i])
        joint_positions.append(position)
        if joint_name[i].find("end") != -1:
            joint_orientations.append(joint_orientations[parent])
            # rotation = joint_rotations[parent]
            joint_rotations.append(0)
        else:
            rotation = R.from_euler('XYZ', motion[index:index+3], degrees=True)
            index += 3
            rotation = joint_rotations[parent] * rotation
            joint_rotations.append(rotation)
            joint_orientations.append(rotation.as_quat())
    return np.array(joint_positions), np.array(joint_orientations)


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    A_pose_motion_data = load_motion_data(A_pose_bvh_path)
    # motion_data = np.zeros_like(A_pose_motion_data)
    A_name, A_parent, A_offset = part1_calculate_T_pose(A_pose_bvh_path)
    T_name, T_parent, T_offset = part1_calculate_T_pose(T_pose_bvh_path)
    motion_data = []
    for frame in A_pose_motion_data:
        frame_data = []
        frame_data.extend(frame[:3])
        index = 3
        dict = {}
        for i in range(len(A_name)):
            if A_name[i].find("end") == -1:
                dict[A_name[i]] = (frame[index:index + 3])
                index += 3
        for i in range(len(T_name)):
            if T_name[i].find("end") == -1:
                if T_name[i] == "lShoulder":
                    rotate = R.from_euler('XYZ', dict["lShoulder"], degrees=True)
                    rotate = rotate * R.from_euler('XYZ', [0, 0, -45], degrees=True) #* rotate
                    frame_data.extend(rotate.as_euler('XYZ', degrees=True))
                elif T_name[i] == "rShoulder":
                    rotate = R.from_euler('XYZ', dict["rShoulder"], degrees=True)
                    rotate = rotate * R.from_euler('XYZ', [0, 0, 45], degrees=True) #* rotate
                    frame_data.extend(rotate.as_euler('XYZ', degrees=True))
                else:
                    frame_data.extend(dict[T_name[i]])
        motion_data.append(np.array(frame_data))
    return np.array(motion_data)
