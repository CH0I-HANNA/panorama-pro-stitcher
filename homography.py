"""
homography.py
-------------
Homography Estimation and Image Warping module.

Pipeline:
  1. estimate_homography() : cv2.findHomography + RANSAC으로 호모그래피 추정
  2. chain_homographies()  : 다중 이미지에서 기준 프레임 기준 절대 호모그래피 계산
  3. compute_canvas_size() : 모든 워핑된 이미지를 담을 캔버스 크기 계산
  4. warp_image()          : cv2.warpPerspective로 이미지를 캔버스 좌표계에 워핑
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────
# 1. Homography Estimation with RANSAC
# ─────────────────────────────────────────────

def estimate_homography(
    kp1: list,
    kp2: list,
    matches: list,
    ransac_thresh: float = 4.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    매칭된 키포인트에서 RANSAC을 이용해 호모그래피 행렬 H를 추정한다.

    H는 image1의 좌표를 image2의 좌표로 변환하는 3×3 행렬이다.
      p2 ~ H @ p1  (동차 좌표 기준)

    RANSAC 과정:
      - 4개 포인트 쌍을 무작위 샘플링하여 후보 H 계산
      - 전체 매칭에 대해 재투영 오차(reprojection error) 계산
      - 오차 < ransac_thresh 인 매칭을 인라이어로 분류
      - 인라이어가 가장 많은 H를 최종 선택 후 인라이어 전체로 재추정

    Parameters
    ----------
    kp1, kp2       : 각 이미지의 키포인트 리스트
    matches        : DMatch 리스트 (queryIdx → kp1, trainIdx → kp2)
    ransac_thresh  : RANSAC 재투영 오차 임계값 (픽셀 단위, 기본 4.0)

    Returns
    -------
    H    : (3, 3) float64 호모그래피 행렬, 실패 시 None
    mask : (N, 1) uint8 인라이어 마스크, 실패 시 None
    """
    MIN_MATCHES = 4  # 호모그래피 추정에 최소 4쌍 필요

    if len(matches) < MIN_MATCHES:
        print(f"  [WARN] 매칭 수 부족: {len(matches)} < {MIN_MATCHES}")
        return None, None

    # 매칭된 좌표 추출
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # cv2.findHomography: 내부적으로 RANSAC 수행
    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000,
        confidence=0.995,
    )

    if H is None:
        print("  [WARN] 호모그래피 추정 실패 (충분한 인라이어 없음)")
        return None, None

    n_inliers = int(mask.sum()) if mask is not None else 0
    n_total   = len(matches)
    ratio     = n_inliers / n_total if n_total > 0 else 0
    print(f"  매칭: {n_total}개 → 인라이어: {n_inliers}개 ({ratio:.1%})")

    return H, mask


# ─────────────────────────────────────────────
# 2. Chain Homographies (Multi-image)
# ─────────────────────────────────────────────

def chain_homographies(
    pairwise_H: List[np.ndarray],
    ref_idx: int,
) -> List[np.ndarray]:
    """
    인접 이미지 쌍의 호모그래피를 체인하여
    각 이미지 → 기준 이미지(ref_idx) 좌표계로의 절대 호모그래피를 계산한다.

    pairwise_H[i] : image[i] → image[i+1] 변환 행렬

    체인 규칙:
      - i < ref : H_abs[i] = H_abs[i+1] @ pairwise_H[i]
                (i→i+1→...→ref)
      - i = ref : H_abs[ref] = I (항등 행렬)
      - i > ref : H_abs[i] = H_abs[i-1] @ inv(pairwise_H[i-1])
                (i→i-1→...→ref, 역변환 사용)

    Parameters
    ----------
    pairwise_H : 인접 쌍 호모그래피 리스트, 길이 = n_images - 1
    ref_idx    : 기준 이미지 인덱스

    Returns
    -------
    abs_H : 길이 n_images의 절대 호모그래피 리스트
    """
    n = len(pairwise_H) + 1  # 이미지 개수
    abs_H = [None] * n

    # 기준 이미지: 항등 변환
    abs_H[ref_idx] = np.eye(3, dtype=np.float64)

    # 기준 이미지 왼쪽 (역방향 체인)
    for i in range(ref_idx - 1, -1, -1):
        # image[i] → image[i+1] → ... → image[ref]
        abs_H[i] = abs_H[i + 1] @ pairwise_H[i]

    # 기준 이미지 오른쪽 (역행렬 체인)
    for i in range(ref_idx + 1, n):
        # pairwise_H[i-1]: image[i-1] → image[i]
        # inv(pairwise_H[i-1]): image[i] → image[i-1]
        H_inv = np.linalg.inv(pairwise_H[i - 1])
        abs_H[i] = abs_H[i - 1] @ H_inv

    return abs_H


# ─────────────────────────────────────────────
# 3. Canvas Size Calculation
# ─────────────────────────────────────────────

def compute_canvas_size(
    images: List[np.ndarray],
    abs_H: List[np.ndarray],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    모든 워핑된 이미지의 꼭짓점 좌표를 기준으로 캔버스 크기를 계산한다.

    각 이미지의 4개 꼭짓점을 호모그래피로 변환한 뒤
    최소/최대 좌표를 구해 캔버스 경계를 결정한다.

    Returns
    -------
    canvas_size : (width, height)
    offset      : (x_min, y_min) — 캔버스 원점 보정을 위한 오프셋
    """
    all_corners = []

    for img, H in zip(images, abs_H):
        h, w = img.shape[:2]
        # 이미지 4개 꼭짓점 (동차 좌표로 perspectiveTransform 입력)
        corners = np.float32([
            [0,   0  ],
            [w-1, 0  ],
            [w-1, h-1],
            [0,   h-1],
        ]).reshape(-1, 1, 2)

        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped_corners)

    all_corners = np.concatenate(all_corners, axis=0)  # (4N, 1, 2)

    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil( all_corners[:, 0, 0].max()))
    y_max = int(np.ceil( all_corners[:, 0, 1].max()))

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    return (canvas_w, canvas_h), (x_min, y_min)


# ─────────────────────────────────────────────
# 4. Image Warping
# ─────────────────────────────────────────────

def warp_image(
    image: np.ndarray,
    H: np.ndarray,
    canvas_size: Tuple[int, int],
    offset: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지를 호모그래피 H로 캔버스 좌표계에 워핑한다.

    오프셋 보정을 위해 평행이동 행렬 T를 H에 합성한다:
      H_final = T @ H

    여기서 T = [[1, 0, -x_off],
                [0, 1, -y_off],
                [0, 0,  1    ]]

    Parameters
    ----------
    image       : 원본 BGR 이미지
    H           : 절대 호모그래피 (image → 기준 프레임)
    canvas_size : (width, height)
    offset      : (x_min, y_min)

    Returns
    -------
    warped : 캔버스에 워핑된 BGR 이미지
    mask   : 유효 픽셀 위치를 나타내는 (H, W) uint8 마스크
    """
    canvas_w, canvas_h = canvas_size
    x_off, y_off = offset

    # 평행이동 행렬 (오프셋 보정)
    T = np.array([
        [1, 0, -x_off],
        [0, 1, -y_off],
        [0, 0,  1    ],
    ], dtype=np.float64)

    H_final = T @ H

    # 이미지 워핑
    warped = cv2.warpPerspective(
        image, H_final, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 유효 영역 마스크: 흰색 이미지를 동일하게 워핑하여 정확한 마스크 생성
    # (이미지 내용에 의존하지 않으므로 어두운 영역도 정확히 처리)
    h_img, w_img = image.shape[:2]
    white = np.ones((h_img, w_img), dtype=np.uint8) * 255
    mask = cv2.warpPerspective(
        white, H_final, (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = (mask > 127).astype(np.uint8)

    return warped, mask
