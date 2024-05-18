# 以下部分均为可更改部分
import dataclasses

from answer_task1 import *
import smooth_utils

def compute_velocity(pos, dt):
    '''
    计算速度
    '''
    return (pos[1:] - pos[:-1]) / dt
@dataclasses
class motion_data:
    motion: BVHMotion
    # pos: np.ndarray
    pos20: np.ndarray
    pos40: np.ndarray
    vel: np.ndarray
    rot: np.ndarray
    avel: np.ndarray
    joint_translation : np.ndarray

def add_motion(filename):
    motion = BVHMotion(filename)
    joint_trans, _ = motion.batch_forward_kinematics_root_zero()
    pos = motion.joint_position[:, 0, :]
    vel = compute_velocity(pos, 1 / 60)
    pos20 = pos[20: ] - pos[: -20]
    pos40 = pos[40: ] - pos[: -40]
    rotations = motion.joint_rotation[:, 0, :]
    for i in range(rotations.shape[0]):
        rotations[i] = R.as_quat(decompose_rotation_with_yaxis(rotations[i])[0])
    ang_vel = smooth_utils.quat_to_avel(rotations)
    inv_R = R.from_quat(rotations[0]).inv()
    for i in range(rotations.shape[0]):
        rotations[i] = (inv_R * R.from_quat(rotations[i])).as_quat()
    return motion_data(motion, pos20, pos40, vel, rotations, ang_vel, joint_trans)

class CharacterController():
    def __init__(self, controller) -> None:
        self.motions = []
        self.motions.append(add_motion('motion_material/kinematic_motion/long_run.bvh'))
        self.motions.append(add_motion('motion_material/kinematic_motion/long_walk.bvh'))
        # self.motions.append(BVHMotion('motion_material/kinematic_motion/long_run.bvh'))
        # self.motions.append(BVHMotion('motion_material/kinematic_motion/long_walk.bvh'))
        # self.motion_pos = []
        # self.motion_velocity = []
        # self.motion_rotation = []
        # self.motion_ang_vel = []
        self.index = []
        for motion in self.motions:
            for i, name in enumerate(motion.motion.joint_name):
                if name == "lAnkle" or name == "rAnkle":
                    self.index.append(i)
        self.controller = controller
        self.cur_joint_pos = [self.motions[0].joint_translation[self.index[0]], self.motions[1].joint_translation[self.index[1]]]
        self.cur_root_pos = [0, 0, 0]
        self.cur_root_rot = [0, 0, 0, 1]
        self.cur_frame = 0
        pass

    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他
        
        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        def compute_cost(motion):
            cost = 0
            for i, idx in enumerate(self.index):
                cost += np.linalg.norm(motion.joint_translation[idx] - self.cur_joint_pos[i])
            cost += np.linalg.norm(motion.pos20)
            return cost
        # 一个简单的例子，输出第i帧的状态
        joint_name = self.motions[0].joint_name
        joint_translation, joint_orientation = self.motions[0].batch_forward_kinematics()
        joint_translation = joint_translation[self.cur_frame]
        joint_orientation = joint_orientation[self.cur_frame]
        
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        self.cur_frame = (self.cur_frame + 1) % self.motions[0].motion_length
        
        return joint_name, joint_translation, joint_orientation
    
    
    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''
        
        # 一个简单的例子，将手柄的位置与角色对齐
        controller.set_pos(self.cur_root_pos)
        controller.set_rot(self.cur_root_rot)
        
        return character_state
    # 你的其他代码,state matchine, motion matching, learning, etc.