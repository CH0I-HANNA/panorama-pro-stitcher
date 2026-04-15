"""
main.py
-------
Panorama Pro Stitcher — CLI 진입점

사용 예시:
  # images/ 폴더의 모든 이미지를 SIFT + Multi-band 블렌딩으로 스티칭
  python main.py

  # 특정 이미지 파일들 지정
  python main.py --images img1.jpg img2.jpg img3.jpg

  # ORB + Feathering 블렌딩
  python main.py --feature ORB --blend feathering

  # 원통형 투영 활성화 (초점 거리 자동 추정)
  python main.py --cylindrical

  # 초점 거리 직접 지정 + 디버그 이미지 저장
  python main.py --cylindrical --focal-length 800 --debug

  # 결과 해상도 제한 (긴 쪽 기준)
  python main.py --max-size 4000
"""

import os
import sys
import glob
import argparse
import time

import cv2
import numpy as np


def imread_unicode(path: str) -> np.ndarray:
    """
    한글·유니코드 경로에서도 이미지를 읽는다.
    cv2.imread는 비ASCII 경로를 지원하지 않으므로
    np.fromfile + cv2.imdecode 방식을 사용한다.
    """
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img

from stitcher import PanoramaStitcher


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def load_images_from_dir(directory: str) -> tuple:
    """
    디렉토리에서 지원 포맷 이미지를 알파벳 순으로 로드한다.
    Returns: (images, paths)
    """
    paths = sorted([
        p for p in glob.glob(os.path.join(directory, '*'))
        if os.path.splitext(p)[1].lower() in SUPPORTED_EXTS
    ])
    if not paths:
        print(f"[ERROR] '{directory}' 에서 이미지를 찾을 수 없습니다.")
        print(f"  지원 포맷: {', '.join(SUPPORTED_EXTS)}")
        sys.exit(1)

    images = []
    for p in paths:
        img = imread_unicode(p)
        if img is None:
            print(f"  [WARN] 로드 실패 (건너뜀): {p}")
            continue
        images.append(img)

    return images, paths


def resize_if_needed(images: list, max_size: int) -> list:
    """
    이미지의 긴 쪽이 max_size를 초과하면 비율 유지하며 축소한다.
    """
    if max_size <= 0:
        return images

    resized = []
    for img in images:
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side > max_size:
            scale = max_size / long_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized.append(img)
    return resized


def print_image_info(images: list, paths: list):
    """로드된 이미지 정보 출력"""
    print(f"\n로드된 이미지 {len(images)}장:")
    for i, (img, path) in enumerate(zip(images, paths)):
        h, w = img.shape[:2]
        size_kb = os.path.getsize(path) / 1024
        print(f"  [{i}] {os.path.basename(path):30s}  {w}×{h}  ({size_kb:.0f} KB)")


# ──────────────────────────────────────────────
# CLI 인수 파싱
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Panorama Pro Stitcher — OpenCV 기반 수동 구현',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 입력
    parser.add_argument(
        '--images', '-i', nargs='+', metavar='FILE',
        help='스티칭할 이미지 파일 경로들 (미지정 시 --input-dir 사용)',
    )
    parser.add_argument(
        '--input-dir', '-d', default='images',
        help='이미지 디렉토리 (기본: images/)',
    )

    # 출력
    parser.add_argument(
        '--output', '-o', default='output/panorama.jpg',
        help='결과 파일 저장 경로 (기본: output/panorama.jpg)',
    )

    # 특징점
    parser.add_argument(
        '--feature', '-f', default='SIFT', choices=['SIFT', 'ORB'],
        help='특징점 알고리즘 (기본: SIFT)',
    )
    parser.add_argument(
        '--ratio', type=float, default=0.75,
        help="Lowe's ratio test 임계값 (기본: 0.75, 범위: 0.5~0.9)",
    )
    parser.add_argument(
        '--ransac', type=float, default=4.0,
        help='RANSAC 재투영 오차 임계값 px (기본: 4.0)',
    )

    # 블렌딩
    parser.add_argument(
        '--blend', '-b', default='multiband',
        choices=['multiband', 'feathering', 'simple'],
        help='블렌딩 방식 (기본: multiband)',
    )
    parser.add_argument(
        '--levels', type=int, default=6,
        help='Multi-band 블렌딩 피라미드 레벨 (기본: 6)',
    )

    # 원통형 투영
    parser.add_argument(
        '--cylindrical', '-c', action='store_true',
        help='원통형 투영 전처리 활성화 (넓은 FOV 파노라마에 권장)',
    )
    parser.add_argument(
        '--focal-length', type=float, default=None,
        help='원통형 투영 초점 거리 px (미지정 시 자동 추정)',
    )

    # 기타
    parser.add_argument(
        '--max-size', type=int, default=0,
        help='입력 이미지 최대 크기 px (긴 쪽 기준, 0=제한 없음)',
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='디버그 이미지 (매칭 결과 등) 저장',
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='진행 로그 숨김',
    )
    parser.add_argument(
        '--show', '-s', action='store_true',
        help='결과 창에 표시 (GUI 환경 필요)',
    )

    return parser.parse_args()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 이미지 로드 ────────────────────────────
    if args.images:
        paths  = args.images
        images = []
        for p in paths:
            img = imread_unicode(p)
            if img is None:
                print(f"[ERROR] 이미지 로드 실패: {p}")
                sys.exit(1)
            images.append(img)
    else:
        images, paths = load_images_from_dir(args.input_dir)

    if not args.quiet:
        print_image_info(images, paths)

    # ── 크기 제한 ──────────────────────────────
    if args.max_size > 0:
        original_sizes = [(img.shape[1], img.shape[0]) for img in images]
        images = resize_if_needed(images, args.max_size)
        resized = [(img.shape[1], img.shape[0]) for img in images]
        if not args.quiet and any(o != r for o, r in zip(original_sizes, resized)):
            print(f"\n  이미지를 최대 {args.max_size}px로 축소함")

    # ── 출력 디렉토리 생성 ─────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # ── 스티처 초기화 ──────────────────────────
    stitcher = PanoramaStitcher(
        feature_method  = args.feature,
        match_ratio     = args.ratio,
        ransac_thresh   = args.ransac,
        blend_mode      = args.blend,
        pyramid_levels  = args.levels,
        use_cylindrical = args.cylindrical,
        focal_length    = args.focal_length,
        save_debug      = args.debug,
        verbose         = not args.quiet,
    )

    # ── 스티칭 실행 ────────────────────────────
    t_start = time.time()
    try:
        panorama = stitcher.stitch(images)
    except RuntimeError as e:
        print(f"\n[ERROR] 스티칭 실패:\n  {e}")
        sys.exit(1)

    elapsed = time.time() - t_start

    # ── 결과 저장 ──────────────────────────────
    success = cv2.imwrite(args.output, panorama)
    if not success:
        print(f"[ERROR] 파일 저장 실패: {args.output}")
        sys.exit(1)

    ph, pw = panorama.shape[:2]
    if not args.quiet:
        print(f"\n결과 파노라마: {pw} × {ph} px")
        print(f"저장 경로: {os.path.abspath(args.output)}")
        print(f"소요 시간: {elapsed:.1f}초")

    # ── 결과 표시 ──────────────────────────────
    if args.show:
        # 화면에 맞게 축소하여 표시
        max_display = 1400
        display_scale = min(1.0, max_display / max(pw, ph))
        display_w = int(pw * display_scale)
        display_h = int(ph * display_scale)
        disp = cv2.resize(panorama, (display_w, display_h))
        cv2.imshow('Panorama Result', disp)
        print("\n  아무 키나 누르면 종료...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
