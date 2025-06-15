import os
import argparse
import numpy as np
from Motion.InverseKinematics import animation_from_positions
from Motion import BVH

def main():
    parser = argparse.ArgumentParser(
        description="Convert a .npy motion file to one or more .bvh files."
    )
    parser.add_argument(
        "npy_file",
        help="Path to the .npy file containing {'motion': np.array(...)}"
    )
    args = parser.parse_args()

    # 입력 npy 파일에서 모션 데이터 로드
    wrapper = np.load(args.npy_file, allow_pickle=True).item()
    pos = wrapper['motion']  # shape: (samples, joints, coords, frames)
    print(f"Loaded motion array with shape {pos.shape}")

    # 샘플 x joints x coord x frames → 샘플 x frames x joints x coord
    pos = pos.transpose(0, 3, 1, 2)

    # 부모 인덱스와 조인트 이름
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    SMPL_JOINT_NAMES = [
        'Pelvis','L_Hip','R_Hip','Spine1','L_Knee','R_Knee','Spine2','L_Ankle','R_Ankle',
        'Spine3','L_Foot','R_Foot','Neck','L_Collar','R_Collar','Head','L_Shoulder',
        'R_Shoulder','L_Elbow','R_Elbow','L_Wrist','R_Wrist'
    ]

    # 입력 파일명에서 확장자 떼고 베이스명 추출
    base = os.path.splitext(os.path.basename(args.npy_file))[0]

    # 각 샘플별로 BVH 저장
    for i, sample in enumerate(pos):
        print(f"Processing sample #{i}")
        anim, sorted_order, _ = animation_from_positions(sample, parents)
        # 파일명: <basename>_0.bvh, <basename>_1.bvh, ...
        out_bvh = f"{base}_{i}.bvh"
        BVH.save(out_bvh, anim, names=np.array(SMPL_JOINT_NAMES)[sorted_order])
        print(f"  -> saved {out_bvh}")

if __name__ == "__main__":
    main()
