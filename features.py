"""
features.py
-----------
Feature Detection, Description, and Matching module.

Pipeline:
  1. detect_and_describe() : SIFT 또는 ORB로 키포인트 및 디스크립터 추출
  2. match_features()       : FLANN(SIFT) 또는 BFMatcher(ORB) + Lowe's ratio test
  3. visualize_matches()    : 매칭 결과 시각화 (선택)
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# 1. Feature Detection & Description
# ─────────────────────────────────────────────

def detect_and_describe(image: np.ndarray, method: str = 'SIFT'):
    """
    이미지에서 키포인트와 디스크립터를 추출한다.

    Parameters
    ----------
    image  : BGR 이미지 (numpy array)
    method : 'SIFT' 또는 'ORB'

    Returns
    -------
    keypoints   : list[cv2.KeyPoint]
    descriptors : numpy array  (SIFT → float32, ORB → uint8)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method.upper() == 'SIFT':
        # SIFT: 스케일·회전 불변 특징점, 128-dim float32 디스크립터
        # nfeatures=0 → 제한 없음, contrastThreshold/edgeThreshold 기본값 사용
        detector = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04, edgeThreshold=10)

    elif method.upper() == 'ORB':
        # ORB: 바이너리 디스크립터, 빠른 속도, 특허 없음
        detector = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)

    else:
        raise ValueError(f"지원하지 않는 method: '{method}'. 'SIFT' 또는 'ORB'를 사용하세요.")

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


# ─────────────────────────────────────────────
# 2. Feature Matching
# ─────────────────────────────────────────────

def match_features(
    desc1: np.ndarray,
    desc2: np.ndarray,
    method: str = 'SIFT',
    ratio_thresh: float = 0.75,
) -> list:
    """
    두 디스크립터 집합 간 특징점 매칭을 수행한다.

    SIFT → FLANN (KD-Tree 기반, float32 디스크립터에 최적)
    ORB  → BFMatcher + Hamming 거리 (바이너리 디스크립터)

    두 경우 모두 Lowe's Ratio Test로 불명확한 매칭(애매한 대응)을 제거한다.

    Parameters
    ----------
    desc1, desc2   : 디스크립터 배열
    method         : 'SIFT' 또는 'ORB'
    ratio_thresh   : Lowe's ratio test 임계값 (낮을수록 엄격, 기본 0.75)

    Returns
    -------
    good_matches : list[cv2.DMatch]
    """
    if desc1 is None or desc2 is None:
        return []

    if method.upper() == 'SIFT':
        # FLANN: KD-Tree 인덱스 (float 디스크립터에 최적)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # BFMatcher + Hamming (ORB 바이너리 디스크립터)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # knnMatch(k=2): 각 쿼리 디스크립터에 대해 가장 가까운 2개 후보 반환
    matches = matcher.knnMatch(desc1, desc2, k=2)

    # Lowe's Ratio Test: 최근접 / 차근접 < ratio_thresh 인 경우만 통과
    # 비율이 낮으면 "뚜렷하게 가장 가까운" 매칭이므로 신뢰 가능
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return good_matches


# ─────────────────────────────────────────────
# 3. Visualization (Optional)
# ─────────────────────────────────────────────

def visualize_matches(
    img1: np.ndarray,
    kp1: list,
    img2: np.ndarray,
    kp2: list,
    matches: list,
    n_draw: int = 50,
    inlier_mask: np.ndarray = None,
) -> np.ndarray:
    """
    매칭 결과를 시각화한다.
    inlier_mask가 주어지면 RANSAC 인라이어(초록)와 아웃라이어(빨강)를 구분한다.

    Parameters
    ----------
    inlier_mask : findHomography()가 반환하는 mask 배열 (선택)

    Returns
    -------
    vis : 두 이미지를 나란히 놓고 매칭 선을 그린 BGR 이미지
    """
    if inlier_mask is not None:
        # 인라이어·아웃라이어 분리
        inliers  = [m for m, keep in zip(matches, inlier_mask.ravel()) if keep]
        outliers = [m for m, keep in zip(matches, inlier_mask.ravel()) if not keep]
        vis = cv2.drawMatches(
            img1, kp1, img2, kp2,
            inliers[:n_draw], None,
            matchColor=(0, 255, 0),        # 초록: 인라이어
            singlePointColor=(200, 200, 200),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        vis = cv2.drawMatches(
            img1, kp1, img2, kp2,
            outliers[:n_draw], None,
            matchColor=(0, 0, 255),        # 빨강: 아웃라이어
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG,
            outImg=vis,
        )
    else:
        vis = cv2.drawMatches(
            img1, kp1, img2, kp2,
            matches[:n_draw], None,
            matchColor=(0, 255, 0),
            singlePointColor=(200, 200, 200),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    return vis
