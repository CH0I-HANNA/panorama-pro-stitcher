"""
stitcher.py
-----------
PanoramaStitcher: 전통적인 컴퓨터 비전 파이프라인 기반 파노라마 스티칭

전체 파이프라인:
  ┌─────────────────────────────────────────────────────────────┐
  │ 0. [선택] Cylindrical Projection                            │
  │    → 넓은 FOV 파노라마에서 원근 왜곡 감소                   │
  ├─────────────────────────────────────────────────────────────┤
  │ 1. Feature Detection & Description (SIFT / ORB)             │
  │    → 각 이미지에서 스케일·회전 불변 키포인트 추출           │
  ├─────────────────────────────────────────────────────────────┤
  │ 2. Feature Matching (FLANN / BFMatcher + Ratio Test)        │
  │    → 인접 이미지 쌍 간 대응점 탐색                          │
  ├─────────────────────────────────────────────────────────────┤
  │ 3. Homography Estimation (cv2.findHomography + RANSAC)      │
  │    → 아웃라이어 제거 후 정밀한 변환 행렬 추정               │
  ├─────────────────────────────────────────────────────────────┤
  │ 4. Homography Chaining                                      │
  │    → 중앙 기준 이미지 좌표계로 절대 변환 행렬 계산          │
  ├─────────────────────────────────────────────────────────────┤
  │ 5. Image Warping (cv2.warpPerspective)                      │
  │    → 모든 이미지를 공통 캔버스에 투영                       │
  ├─────────────────────────────────────────────────────────────┤
  │ 6. Image Blending (Feathering / Multi-band)                 │
  │    → 경계 세임(seam) 없이 자연스러운 합성                   │
  └─────────────────────────────────────────────────────────────┘
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Tuple

from features   import detect_and_describe, match_features, visualize_matches
from homography import estimate_homography, chain_homographies, compute_canvas_size, warp_image
from blending   import feathering_blend, multiband_blend
from projection import cylindrical_projection, crop_black_borders, estimate_focal_length


class PanoramaStitcher:
    """
    모듈형 파노라마 스티처.

    Parameters
    ----------
    feature_method  : 특징점 알고리즘 ('SIFT' | 'ORB')
    match_ratio     : Lowe's ratio test 임계값 (0.6~0.8, 낮을수록 엄격)
    ransac_thresh   : RANSAC 재투영 오차 임계값 (픽셀)
    blend_mode      : 블렌딩 방식 ('multiband' | 'feathering' | 'simple')
    pyramid_levels  : Multi-band 블렌딩 피라미드 레벨 수
    use_cylindrical : 원통형 투영 전처리 여부
    focal_length    : 원통형 투영 초점 거리 (None이면 자동 추정)
    save_debug      : True면 중간 결과(매칭 이미지 등)를 debug/ 에 저장
    verbose         : 진행 로그 출력 여부
    """

    def __init__(
        self,
        feature_method:  str   = 'SIFT',
        match_ratio:     float = 0.75,
        ransac_thresh:   float = 4.0,
        blend_mode:      str   = 'multiband',
        pyramid_levels:  int   = 6,
        use_cylindrical: bool  = False,
        focal_length:    Optional[float] = None,
        save_debug:      bool  = False,
        verbose:         bool  = True,
    ):
        self.feature_method  = feature_method.upper()
        self.match_ratio     = match_ratio
        self.ransac_thresh   = ransac_thresh
        self.blend_mode      = blend_mode.lower()
        self.pyramid_levels  = pyramid_levels
        self.use_cylindrical = use_cylindrical
        self.focal_length    = focal_length
        self.save_debug      = save_debug
        self.verbose         = verbose

        # 결과 저장용 내부 상태
        self._debug_dir  = 'debug'
        self._all_kp     = []
        self._pairwise_H = []

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def stitch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        이미지 리스트를 받아 파노라마 이미지를 반환한다.

        Parameters
        ----------
        images : BGR 이미지 리스트 (왼쪽 → 오른쪽 순서 권장)

        Returns
        -------
        panorama : 합성된 파노라마 BGR 이미지
        """
        n = len(images)
        if n == 0:
            raise ValueError("이미지가 없습니다.")
        if n == 1:
            self._log("이미지 1장 → 스티칭 불필요, 원본 반환")
            return images[0].copy()

        self._log(f"\n{'='*55}")
        self._log(f"  파노라마 스티칭 시작: {n}장의 이미지")
        self._log(f"  특징점: {self.feature_method}  |  블렌딩: {self.blend_mode}")
        self._log(f"{'='*55}")

        # ── Step 0: Cylindrical Projection ──────────────────
        if self.use_cylindrical:
            images = self._apply_cylindrical(images)

        # ── Step 1: Feature Detection & Description ──────────
        self._log(f"\n[1/5] 특징점 검출 ({self.feature_method})...")
        all_kp, all_desc = self._detect_features(images)

        # ── Step 2 & 3: Matching + Homography Estimation ─────
        self._log("\n[2/5] 특징점 매칭 및 호모그래피 추정...")
        pairwise_H = self._match_and_estimate(images, all_kp, all_desc)

        # ── Step 4: Chain Homographies ───────────────────────
        self._log("\n[3/5] 절대 호모그래피 체인 계산...")
        ref_idx = n // 2
        abs_H   = chain_homographies(pairwise_H, ref_idx)
        self._log(f"  기준 이미지: #{ref_idx}")

        # ── Step 5: Warping ──────────────────────────────────
        self._log("\n[4/5] 이미지 워핑...")
        canvas_size, offset = compute_canvas_size(images, abs_H)
        self._log(f"  캔버스 크기: {canvas_size[0]} × {canvas_size[1]} px")

        warped_images, warped_masks = self._warp_all(images, abs_H, canvas_size, offset)

        # ── Step 6: Blending ─────────────────────────────────
        self._log(f"\n[5/5] 블렌딩 ({self.blend_mode})...")
        panorama = self._blend(warped_images, warped_masks)

        self._log("\n  완료!")
        return panorama

    # ──────────────────────────────────────────
    # Private Methods
    # ──────────────────────────────────────────

    def _apply_cylindrical(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """원통형 투영 전처리"""
        f = self.focal_length or estimate_focal_length(images)
        self._log(f"\n[0/5] 원통형 투영 (f = {f:.1f} px)...")
        projected = []
        for i, img in enumerate(images):
            proj = cylindrical_projection(img, f)
            projected.append(proj)
            self._log(f"  이미지 {i}: {img.shape[1]}×{img.shape[0]} → 투영 완료")
        return projected

    def _detect_features(
        self,
        images: List[np.ndarray],
    ) -> Tuple[list, list]:
        """모든 이미지에서 특징점 검출"""
        all_kp   = []
        all_desc = []
        for i, img in enumerate(images):
            kp, desc = detect_and_describe(img, self.feature_method)
            all_kp.append(kp)
            all_desc.append(desc)
            self._log(f"  이미지 {i}: {len(kp)}개 키포인트 검출")
        self._all_kp = all_kp
        return all_kp, all_desc

    def _match_and_estimate(
        self,
        images:    List[np.ndarray],
        all_kp:    list,
        all_desc:  list,
    ) -> List[np.ndarray]:
        """인접 쌍 매칭 + RANSAC 호모그래피 추정"""
        pairwise_H = []

        if self.save_debug:
            os.makedirs(self._debug_dir, exist_ok=True)

        for i in range(len(images) - 1):
            self._log(f"  쌍 ({i}, {i+1}):")

            # 특징점 매칭
            good_matches = match_features(
                all_desc[i], all_desc[i + 1],
                self.feature_method,
                self.match_ratio,
            )
            self._log(f"    ratio test 통과: {len(good_matches)}개 매칭")

            # RANSAC 호모그래피 추정
            H, mask = estimate_homography(
                all_kp[i], all_kp[i + 1],
                good_matches,
                self.ransac_thresh,
            )

            if H is None:
                raise RuntimeError(
                    f"이미지 쌍 ({i}, {i+1})의 호모그래피 추정 실패.\n"
                    f"  → 이미지 간 시야 겹침이 충분한지 확인하세요 (권장: 30~50%)."
                )

            pairwise_H.append(H)

            # 디버그: 매칭 결과 이미지 저장
            if self.save_debug:
                vis = visualize_matches(
                    images[i],   all_kp[i],
                    images[i+1], all_kp[i+1],
                    good_matches, inlier_mask=mask,
                )
                path = os.path.join(self._debug_dir, f'match_{i}_{i+1}.jpg')
                cv2.imwrite(path, vis)
                self._log(f"    매칭 시각화 저장: {path}")

        self._pairwise_H = pairwise_H
        return pairwise_H

    def _warp_all(
        self,
        images:      List[np.ndarray],
        abs_H:       List[np.ndarray],
        canvas_size: Tuple[int, int],
        offset:      Tuple[int, int],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """모든 이미지를 캔버스에 워핑"""
        warped_images = []
        warped_masks  = []
        for i, (img, H) in enumerate(zip(images, abs_H)):
            warped, mask = warp_image(img, H, canvas_size, offset)
            warped_images.append(warped)
            warped_masks.append(mask)
            self._log(f"  이미지 {i} 워핑 완료")
        return warped_images, warped_masks

    def _blend(
        self,
        warped_images: List[np.ndarray],
        warped_masks:  List[np.ndarray],
    ) -> np.ndarray:
        """블렌딩 방식에 따라 최종 합성"""
        pairs = list(zip(warped_images, warped_masks))

        if self.blend_mode == 'multiband':
            self._log(f"  Laplacian 피라미드 {self.pyramid_levels} 레벨...")
            result = multiband_blend(pairs, levels=self.pyramid_levels)

        elif self.blend_mode == 'feathering':
            self._log("  거리 변환 가중치 블렌딩...")
            result = feathering_blend(pairs)

        else:
            # simple: 나중 이미지가 겹치는 영역을 덮어씀
            self._log("  단순 오버레이 (경계 처리 없음)...")
            h, w = warped_images[0].shape[:2]
            result = np.zeros((h, w, 3), dtype=np.uint8)
            for img, mask in pairs:
                result[mask > 0] = img[mask > 0]

        return result


# ──────────────────────────────────────────────
# 편의 함수: 파일 경로 리스트로 직접 스티칭
# ──────────────────────────────────────────────

def stitch_from_paths(
    image_paths: List[str],
    **kwargs,
) -> np.ndarray:
    """
    이미지 파일 경로 리스트를 받아 파노라마를 반환하는 편의 함수.

    Parameters
    ----------
    image_paths : 이미지 파일 경로 리스트
    **kwargs    : PanoramaStitcher 생성자 인수

    Returns
    -------
    panorama : BGR 파노라마 이미지
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"이미지 로드 실패: {path}")
        images.append(img)

    stitcher = PanoramaStitcher(**kwargs)
    return stitcher.stitch(images)
