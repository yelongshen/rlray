# lerobot dataset view and test.
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

print("List of available datasets:")
pprint(lerobot.available_datasets)

repo_id = 'lerobot/aloha_mobile_shrimp'
ds_meta = LeRobotDatasetMetadata(repo_id)

print(f"Total number of episodes: {ds_meta.total_episodes}")
print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
print(f"Frames per second used during data collection: {ds_meta.fps}")
print(f"Robot type: {ds_meta.robot_type}")
print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")


print("Tasks:")
print(ds_meta.tasks)
print("Features:")
print(ds_meta.features)

#'observation.images.cam_high': {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels'], 'video_info': {'video.fps': 50.0, 'video.codec': 'av1', 'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'has_audio': False}}, 

#'observation.images.cam_left_wrist': {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels'], 'video_info': {'video.fps': 50.0, 'video.codec': 'av1', 'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'has_audio': False}}, 

#'observation.images.cam_right_wrist': {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels'], 'video_info': {'video.fps': 50.0, 'video.codec': 'av1', 'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'has_audio': False}}, 

#'observation.state': {'dtype': 'float32', 'shape': (14,), 'names': {'motors': ['left_waist', 'left_shoulder', 'left_elbow', 'left_forearm_roll', 'left_wrist_angle', 'left_wrist_rotate', 'left_gripper', 'right_waist', 'right_shoulder', 'right_elbow', 'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate', 'right_gripper']}}, 
# Physical meaning:
# This typically represents the robot’s measured joint states, i.e., the joint positions (angles for revolute joints, linear displacements for prismatic joints).

#'observation.effort': {'dtype': 'float32', 'shape': (14,), 'names': {'motors': ['left_waist', 'left_shoulder', 'left_elbow', 'left_forearm_roll', 'left_wrist_angle', 'left_wrist_rotate', 'left_gripper', 'right_waist', 'right_shoulder', 'right_elbow', 'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate', 'right_gripper']}},
#Physical meaning:
#This usually encodes the torque or force applied at each joint, as sensed by the robot. In ROS, JointState.effort is exactly this: the measured effort that actuators are exerting to hold or move the joints.

#'action': {'dtype': 'float32', 'shape': (14,), 'names': {'motors': ['left_waist', 'left_shoulder', 'left_elbow', 'left_forearm_roll', 'left_wrist_angle', 'left_wrist_rotate', 'left_gripper', 'right_waist', 'right_shoulder', 'right_elbow', 'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate', 'right_gripper']}}, 'episode_index': {'dtype': 'int64', 'shape': (1,), 'names': None}, 'frame_index': {'dtype': 'int64', 'shape': (1,), 'names': None}, 'timestamp': {'dtype': 'float32', 'shape': (1,), 'names': None}, 'next.done': {'dtype': 'bool', 'shape': (1,), 'names': None}, 'index': {'dtype': 'int64', 'shape': (1,), 'names': None}, 'task_index': {'dtype': 'int64', 'shape': (1,), 'names': None}}