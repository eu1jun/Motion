import numpy as np
from Motion.InverseKinematics import animation_from_positions
from Motion import BVH

npy_file = 'results.npy'
bvh_file = 'results.bvh'
wrapper = np.load(npy_file, allow_pickle=True).item()
pos = wrapper['motion']
print(pos.shape)
pos = pos.transpose(0, 3, 1, 2) # samples x joints x coord x frames ==> samples x frames x joints x coord
parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SMPL_JOINT_NAMES = [
'Pelvis', # 0
'L_Hip', # 1
'R_Hip', # 2
'Spine1', # 3
'L_Knee', # 4
'R_Knee', # 5
'Spine2', # 6
'L_Ankle', # 7
'R_Ankle', # 8
'Spine3', # 9
'L_Foot', # 10
'R_Foot', # 11
'Neck', # 12
'L_Collar', # 13
'R_Collar', # 14
'Head', # 15
'L_Shoulder', # 16
'R_Shoulder', # 17
'L_Elbow', # 18
'R_Elbow', # 19
'L_Wrist', # 20
'R_Wrist', # 21
]
for i, p in enumerate(pos):
    print(f'starting anim no. {i}')
    anim, sorted_order, _ = animation_from_positions(p, parents)
    BVH.save(bvh_file.format(i), anim, names=np.array(SMPL_JOINT_NAMES)[sorted_order])