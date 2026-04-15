"""
blending.py
-----------
Image Blending module.

두 가지 블렌딩 방식 구현:

1. Feathering (Alpha Blending)
   - 각 이미지의 거리 변환(distance transform)을 가중치로 사용
   - 경계에서 멀수록 높은 가중치 → 겹치는 영역에서 부드러운 선형 보간
   - 구현 간단, 속도 빠름 / 고주파 텍스처 경계에서 약한 블러 발생 가능

2. Multi-band Blending (Laplacian Pyramid)
   - Laplacian 피라미드의 각 주파수 대역별로 독립적으로 블렌딩
   - 저주파(색상/밝기): 넓은 전환 영역 사용 → 자연스러운 노출 보정
   - 고주파(텍스처/엣지): 좁은 전환 영역 사용 → 선명한 디테일 유지
   - 결과: Feathering 대비 눈에 띄는 경계 세임(seam) 감소
"""

import cv2
import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────
# 공통 유틸: 거리 변환 기반 가중치 맵 생성
# ─────────────────────────────────────────────

def _compute_weight_maps(
    masks: List[np.ndarray],
) -> List[np.ndarray]:
    """
    각 이미지의 마스크에서 거리 변환 기반 가중치 맵을 계산한다.

    distanceTransform: 각 픽셀에서 가장 가까운 배경(0) 픽셀까지의 거리
    → 경계에서 멀수록(이미지 중심 방향) 높은 가중치

    모든 이미지의 가중치 맵을 정규화하여 픽셀별 합이 1이 되게 한다.

    Parameters
    ----------
    masks : 각 이미지의 (H, W) uint8 마스크 리스트

    Returns
    -------
    norm_weights : 정규화된 float32 가중치 맵 리스트
    """
    weight_maps = []
    for mask in masks:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, maskSize=5)
        weight_maps.append(dist.astype(np.float32))

    # 픽셀별 정규화: w_i(x,y) / sum_j(w_j(x,y))
    weight_stack = np.stack(weight_maps, axis=0)  # (N, H, W)
    weight_sum   = weight_stack.sum(axis=0, keepdims=True)  # (1, H, W)
    weight_sum   = np.where(weight_sum == 0, 1.0, weight_sum)  # 0 나눗셈 방지
    norm_weights = (weight_stack / weight_sum)  # (N, H, W)

    return [norm_weights[i] for i in range(len(masks))]


# ─────────────────────────────────────────────
# 1. Feathering Blend
# ─────────────────────────────────────────────

def feathering_blend(
    images_with_masks: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    거리 변환 가중치 기반 Feathering 블렌딩.

    result(x,y) = Σ_i [ w_i(x,y) * I_i(x,y) ]
    단, w_i는 이미지 i의 경계까지 거리에 비례하며 합이 1로 정규화됨.

    Parameters
    ----------
    images_with_masks : [(image, mask), ...] 리스트
                        image: (H, W, 3) uint8 BGR
                        mask : (H, W)    uint8 이진 마스크

    Returns
    -------
    result : (H, W, 3) uint8 블렌딩 결과
    """
    if not images_with_masks:
        return None

    images = [im for im, _ in images_with_masks]
    masks  = [mk for _, mk in images_with_masks]

    h, w = images[0].shape[:2]
    result     = np.zeros((h, w, 3), dtype=np.float64)
    weight_sum = np.zeros((h, w, 1), dtype=np.float64)

    norm_weights = _compute_weight_maps(masks)

    for img, w_map in zip(images, norm_weights):
        w_3c = w_map[:, :, np.newaxis]  # (H, W, 1) → 브로드캐스팅용
        result     += img.astype(np.float64) * w_3c
        weight_sum += w_3c

    # 유효 픽셀만 정규화 (weight_sum=0인 픽셀은 그대로 0)
    valid = (weight_sum[..., 0] > 0)
    result[valid] /= weight_sum[valid]

    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 2. Multi-band Blending (Laplacian Pyramid)
# ─────────────────────────────────────────────

def _build_gaussian_pyramid(
    image: np.ndarray,
    levels: int,
) -> List[np.ndarray]:
    """
    Gaussian 피라미드 생성.
    pyramid[0] = 원본 크기, pyramid[-1] = 가장 작은 크기
    """
    pyramid = [image.astype(np.float32)]
    cur = image.astype(np.float32)
    for _ in range(levels - 1):
        cur = cv2.pyrDown(cur)
        pyramid.append(cur)
    return pyramid


def _build_laplacian_pyramid(
    image: np.ndarray,
    levels: int,
) -> List[np.ndarray]:
    """
    Laplacian 피라미드 생성.
    L[i] = G[i] - pyrUp(G[i+1])   (밴드패스 필터, 고주파~중주파 성분)
    L[-1] = G[-1]                  (최저주파 성분, 기저 레벨)

    각 레벨은 특정 공간 주파수 대역의 에너지를 담는다.
    """
    gp = _build_gaussian_pyramid(image, levels)
    lp = []
    for i in range(levels - 1):
        # pyrUp의 출력 크기를 상위 레벨 크기와 명시적으로 맞춤 (홀수 크기 오류 방지)
        target_h, target_w = gp[i].shape[:2]
        up = cv2.pyrUp(gp[i + 1], dstsize=(target_w, target_h))
        lap = cv2.subtract(gp[i], up)  # subtract: float overflow 방지
        lp.append(lap)
    lp.append(gp[-1])  # 기저 레벨 (저주파)
    return lp


def multiband_blend(
    images_with_masks: List[Tuple[np.ndarray, np.ndarray]],
    levels: int = 6,
) -> np.ndarray:
    """
    Laplacian 피라미드 기반 Multi-band 블렌딩.

    알고리즘:
      1. 각 이미지의 Laplacian 피라미드 계산
      2. 각 이미지의 가중치 마스크 Gaussian 피라미드 계산
         (고주파 레벨→좁은 전환 / 저주파 레벨→넓은 전환 자동 적용)
      3. 각 피라미드 레벨에서: blended_L[i] += L_img[i] * G_weight[i]
      4. 블렌딩된 Laplacian 피라미드를 역으로 재구성(reconstruct)

    Parameters
    ----------
    images_with_masks : [(image, mask), ...] 리스트
    levels            : 피라미드 레벨 수 (클수록 넓은 영역 블렌딩, 기본 6)

    Returns
    -------
    result : (H, W, 3) uint8 블렌딩 결과
    """
    if not images_with_masks:
        return None
    if len(images_with_masks) == 1:
        return images_with_masks[0][0].copy()

    images = [im for im, _ in images_with_masks]
    masks  = [mk for _, mk in images_with_masks]
    h, w   = images[0].shape[:2]

    # 피라미드 레벨을 이미지 크기에 맞게 자동 제한
    max_levels = int(np.floor(np.log2(min(h, w)))) - 1
    levels = min(levels, max_levels)

    # 정규화된 가중치 맵 계산
    norm_weights = _compute_weight_maps(masks)

    # 블렌딩된 피라미드 초기화
    blended_pyramid = None

    for img, w_map in zip(images, norm_weights):
        # 이미지 Laplacian 피라미드
        lp_img = _build_laplacian_pyramid(img.astype(np.float32), levels)

        # 가중치 마스크 Gaussian 피라미드 (3채널 확장)
        w_map_3c = np.stack([w_map, w_map, w_map], axis=-1).astype(np.float32)
        gp_weight = _build_gaussian_pyramid(w_map_3c, levels)

        if blended_pyramid is None:
            blended_pyramid = [np.zeros_like(lp_img[lv]) for lv in range(levels)]

        for lv in range(levels):
            blended_pyramid[lv] += lp_img[lv] * gp_weight[lv]

    # 피라미드 재구성: 기저 레벨부터 역방향 누적
    result = blended_pyramid[-1]
    for lv in range(levels - 2, -1, -1):
        target_h, target_w = blended_pyramid[lv].shape[:2]
        result = cv2.pyrUp(result, dstsize=(target_w, target_h))
        result = cv2.add(result, blended_pyramid[lv])

    return np.clip(result, 0, 255).astype(np.uint8)
