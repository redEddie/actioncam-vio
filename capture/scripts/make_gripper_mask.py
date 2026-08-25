#!/usr/bin/env python3
"""Generate the static gripper-occlusion mask for SLAM (--mask of gopro_vio.slam).

카메라가 그리퍼에 강체 고정이라 그리퍼는 항상 같은 화면 영역을 차지한다.
여닫는 동안 손가락이 회전하며 쓸고 가는 영역 전체를 덮는 사다리꼴 마스크
(흰색 = 추적 제외). 기본값은 HERO7 + 자체 그리퍼 리그 실측(GX014232/233).

    python scripts/make_gripper_mask.py -o cameras/hero7black/calibration/gripper_mask.png
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--size", default="960x720", help="mask resolution WxH")
    ap.add_argument("--top-y", type=float, default=0.53,
                    help="top edge of trapezoid (fraction of height)")
    ap.add_argument("--top-x", type=float, nargs=2, default=(0.30, 0.70),
                    help="top edge x range (fraction of width)")
    ap.add_argument("--bottom-x", type=float, nargs=2, default=(0.02, 0.98),
                    help="bottom edge x range (fraction of width)")
    args = ap.parse_args()

    w, h = map(int, args.size.split("x"))
    mask = np.zeros((h, w), np.uint8)
    pts = np.array([
        [args.bottom_x[0] * w, h], [args.top_x[0] * w, args.top_y * h],
        [args.top_x[1] * w, args.top_y * h], [args.bottom_x[1] * w, h],
    ], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    cv2.imwrite(args.out, mask)
    print(f"masked fraction: {mask.mean() / 255:.2%} -> {args.out}")


if __name__ == "__main__":
    main()
