"""Measure the actual scanner and optionally compare every output pixel.

python scripts/benchmark_scanner.py --save <reference.npz>
python scripts/benchmark_scanner.py --compare <reference.npz> --repeat 3
"""
import argparse
from contextlib import ExitStack
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import subprocess
import statistics
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scanner import scan_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=ROOT / 'test_image.jpg')
    parser.add_argument('--save', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--reference-commit', help='Use the original convolution/NMS/hysteresis from this local Git commit')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f'Cannot read {args.image}')
    durations = []
    result = None
    reference_context = ExitStack()
    if args.reference_commit:
        conv_scope = {}
        canny_scope = {}
        for filename, scope in [('src/convolution.py', conv_scope), ('src/canny.py', canny_scope)]:
            source = subprocess.check_output(['git', 'show', f'{args.reference_commit}:{filename}'], cwd=ROOT)
            exec(compile(source, f'{args.reference_commit}:{filename}', 'exec'), scope)
        reference_context.enter_context(patch('src.canny.convolution2d', conv_scope['convolution2d']))
        reference_context.enter_context(patch('src.sharpening.convolution2d', conv_scope['convolution2d']))
        reference_context.enter_context(patch('src.canny.non_maximum_suppression', canny_scope['non_maximum_suppression']))
        reference_context.enter_context(patch('src.canny.hysteresis', canny_scope['hysteresis']))
        print(f'Reference implementations: {args.reference_commit}', flush=True)
    for index in range(args.repeat):
        print(f'Scanning {image.shape[1]}x{image.shape[0]} (run {index + 1})...', flush=True)
        profile = cProfile.Profile()
        start = time.perf_counter()
        if args.profile:
            profile.enable()
        result = scan_document(image, detection_width=800, output_width=1600, verbose=False)
        if args.profile:
            profile.disable()
        durations.append(time.perf_counter() - start)
        print(f'Elapsed: {durations[-1]:.4f}s', flush=True)
        if args.profile:
            summary = io.StringIO()
            pstats.Stats(profile, stream=summary).strip_dirs().sort_stats('cumtime').print_stats(14)
            print(summary.getvalue(), flush=True)
    reference_context.close()
    keys = ['corners', 'edge', 'nms', 'warped', 'sharpened']
    if args.save:
        np.savez_compressed(args.save, **{key: result[key] for key in keys})
        print(f'Reference saved: {args.save}', flush=True)
    if args.compare:
        with np.load(args.compare) as reference:
            for key in keys:
                np.testing.assert_array_equal(result[key], reference[key], err_msg=f'{key} changed')
        print('All corners, edges, NMS, warped image and final pixels are identical.', flush=True)
    print(json.dumps({
        'image': str(args.image), 'input_shape': list(image.shape),
        'output_shape': list(result['sharpened'].shape),
        'seconds': durations, 'median_seconds': statistics.median(durations),
        'final_sha256': hashlib.sha256(result['sharpened'].tobytes()).hexdigest(),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
