"""
projection.py
-------------
Cylindrical Projection module.

원통형 투영(Cylindrical Projection)은 넓은 시야각 파노라마에서
cv2.warpPerspective만 사용할 때 발생하는 "활 모양(bow-tie)" 왜곡을 줄인다.

수학적 원리:
  원통형 좌표계에서:
    θ = (x_src - cx) / f          (수평 각도)
    h = (y_src - cy) / f          (정규화 높이)

  역사상(inverse mapping): 원통 좌표 → 원본 이미지 좌표
    x_orig = f * tan(θ) + cx = f * tan((x_dst - cx) / f) + cx
    y_orig = f * h / cos(θ) + cy = (y_dst - cy) / cos((x_dst - cx) / f) + cy

  cv2.remap으로 역사상 적용 → 원통형으로 투영된 이미지 반환

초점 거리(f):
  - EXIF 데이터에서 읽거나 직접 지정
  - 경험적 추정: f ≈ image_width (약 60° FOV 기준)
  - 큰 f → 적은 왜곡 / 작은 f → 강한 원통 효과
"""

import cv2
import numpy as np
from typing import List, Optional


# ─────────────────────────────────────────────
# 1. Cylindrical Projection (Vectorized)
# ─────────────────────────────────────────────

def cylindrical_projection(
    image: np.ndarray,
    focal_length: float,
) -> np.ndarray:
    """
    이미지를 원통형 표면에 투영한다.

    NumPy 벡터화 연산으로 빠르게 역사상 맵을 생성하고
    cv2.remap으로 쌍선형 보간(bilinear interpolation) 적용.

    Parameters
    ----------
    image        : BGR 이미지
    focal_length : 초점 거리 (픽셀 단위)

    Returns
    -------
    projected : 원통형 투영된 BGR 이미지 (입력과 동일 크기)
    """
    h, w = image.shape[:2]
    cx = w / 2.0
    cy = h / 2.0
    f  = float(focal_length)

    # 출력 이미지의 각 픽셀 좌표 그리드 생성
    # map_x[y,x], map_y[y,x] → 원본 이미지에서 샘플링할 좌표
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    map_x_grid, map_y_grid = np.meshgrid(xs, ys)  # (H, W)

    # 원통형 좌표 → 원본 평면 좌표 역사상
    theta     = (map_x_grid - cx) / f          # 수평 각도
    cos_theta = np.cos(theta)

    # 원본 이미지 좌표 (역사상)
    src_x = (f * np.tan(theta) + cx).astype(np.float32)
    src_y = ((map_y_grid - cy) / cos_theta + cy).astype(np.float32)

    # remap: 역사상 맵 적용 → 원통형 투영 결과
    projected = cv2.remap(
        image, src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return projected


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    """
    원통형 투영 후 생기는 검은 테두리를 자동으로 제거한다.

    가로/세로 방향으로 비어있는(모두 0) 행/열을 제거한다.

    Parameters
    ----------
    image : BGR 이미지

    Returns
    -------
    cropped : 검은 테두리 제거된 이미지
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 각 행/열에서 하나라도 non-zero인 경우를 유효 행/열로 판단
    row_mask = gray.max(axis=1) > 0  # (H,)
    col_mask = gray.max(axis=0) > 0  # (W,)

    rows = np.where(row_mask)[0]
    cols = np.where(col_mask)[0]

    if rows.size == 0 or cols.size == 0:
        return image  # 유효 픽셀 없음 → 원본 반환

    r_start, r_end = rows[0],  rows[-1]  + 1
    c_start, c_end = cols[0],  cols[-1]  + 1

    return image[r_start:r_end, c_start:c_end]


# ─────────────────────────────────────────────
# 2. Focal Length Estimation
# ─────────────────────────────────────────────

def estimate_focal_length(
    images: List[np.ndarray],
    fov_degrees: float = 60.0,
) -> float:
    """
    이미지 크기와 추정 FOV(수평 시야각)를 기반으로 초점 거리를 추정한다.

    핀홀 카메라 모델:
      f = (image_width / 2) / tan(FOV_horizontal / 2)

    일반 스마트폰/DSLR 광각 렌즈의 수평 FOV ≈ 60~70°
    → 기본값 60° 사용 (이미지 폭과 거의 동일한 결과)

    Parameters
    ----------
    images      : 이미지 리스트 (크기 기준은 첫 번째 이미지)
    fov_degrees : 예상 수평 시야각 (도 단위)

    Returns
    -------
    focal_length : 픽셀 단위 초점 거리 (float)
    """
    h, w = images[0].shape[:2]
    fov_rad = np.radians(fov_degrees)
    focal_length = (w / 2.0) / np.tan(fov_rad / 2.0)
    return focal_length
