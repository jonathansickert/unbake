"""
Render TRELLIS.2 conditioning views for every .glb in the dataset with the original TRELLIS.2 setup
(blender_script/render_cond.py): 16 views on a Hammersley sphere, random FOV in [10, 70] deg, random
point lights, 1024 px, transparent background.

Run from the repo root:
    python data_toolkit/render_cond.py baked_env

Output: <root>/<sha256>/<glb name>_cond/000.png ... and transforms.json. All .glb files of an asset use
the same cameras. Existing outputs are skipped.
"""
import os
import sys
import glob
import json
import zlib
from subprocess import DEVNULL, call
import numpy as np

BLENDER_PATH = '/Applications/Blender.app/Contents/MacOS/Blender'
NUM_COND_VIEWS = 16


def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    """From TRELLIS.2 data_toolkit/utils.py."""
    u, v, inv_base = n / num_samples, 0.0, 0.5
    while n > 0:   # radical inverse in base 2
        v += (n % 2) * inv_base
        n //= 2
        inv_base /= 2
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


def _render_cond(file_path, out_dir, num_cond_views, seed):
    # Build conditional view cameras (as in TRELLIS.2), seeded so all .glb files of an asset match
    rng = np.random.RandomState(seed)
    offset = (rng.rand(), rng.rand())
    yaws, pitchs = zip(*[sphere_hammersley_sequence(i, num_cond_views, offset) for i in range(num_cond_views)])
    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    ks = rng.uniform(1 / radius_max ** 2, 1 / radius_min ** 2, num_cond_views)
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
    cond_views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    call([
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blender_script', 'render_cond.py'),
        '--',
        '--object', os.path.abspath(file_path),
        '--cond_views', json.dumps(cond_views),
        '--cond_resolution', '1024',
        '--cond_output_folder', os.path.abspath(out_dir),
        '--engine', 'CYCLES',
    ], stdout=DEVNULL, stderr=DEVNULL)
    return os.path.exists(os.path.join(out_dir, 'transforms.json'))


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'baked_env'
    glbs = sorted(glob.glob(os.path.join(root, '*', '*.glb')))
    for i, glb in enumerate(glbs):
        out_dir = os.path.splitext(glb)[0] + '_cond'
        if os.path.exists(os.path.join(out_dir, 'transforms.json')):
            continue
        sha256 = os.path.basename(os.path.dirname(glb))
        ok = _render_cond(glb, out_dir, NUM_COND_VIEWS, seed=zlib.crc32(sha256.encode()))
        print(f'[{i + 1}/{len(glbs)}] {glb}: {"done" if ok else "FAILED"}', flush=True)
