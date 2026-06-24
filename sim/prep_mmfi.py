"""Preprocess MM-Fi into compact per-sample multimodal tensors for the FL sim.

Task: human ACTION RECOGNITION (27 classes A01-A27) from three modalities the
paper uses -- depth image, skeleton (3D body keypoints), mmWave point cloud.
Each (subject, action) sequence (~297 frames) is cut into fixed windows; each
window becomes one classification sample. Subject id is kept for the natural
non-IID partition across vehicles (as with UCI HAR).

Layout expected under --root:
  <root>/E0x/S##/A##/ground_truth.npy        skeleton, [F,17,3]   (from E0x.zip)
  <root>/E0x/S##/A##/depth/frame###.png      depth 16-bit png     (from E0x.zip)
  <root>/filtered_mmwave/E0x/S##/A##/frame###.bin   mmwave [P,5]  (filtered_mmwave.zip)

Modalities present are auto-detected; missing ones are skipped (so this runs
on mmwave alone before the depth/skeleton zip finishes downloading).

Output: <out> .npz with keys xsk,xmm,xdp (only those present), y, subj, and
the per-modality shapes.  Run:
  ~/anaconda3/envs/cm-pfl/bin/python sim/prep_mmfi.py --root results/data/mmfi
"""
import argparse
import glob
import os
import numpy as np

W = 30            # frames per window
STRIDE = 15       # window hop -> ~ (F-W)/STRIDE + 1 samples per sequence
T_SK = 16         # skeleton frames kept per window (subsampled)
DEPTH_HW = 112    # depth resized to DEPTH_HW x DEPTH_HW
VOX = 32          # mmWave bird's-eye voxel grid resolution
N_ACT = 27


def list_subjects(root):
    """Return {subject: scene} for subjects whose E0x dir is present."""
    out = {}
    for scene in sorted(os.listdir(root)):
        sp = os.path.join(root, scene)
        if not (scene.startswith("E") and os.path.isdir(sp)):
            continue
        for subj in sorted(os.listdir(sp)):
            if subj.startswith("S") and os.path.isdir(os.path.join(sp, subj)):
                out[subj] = scene
    return out


def mmwave_dir(root, scene, subj, act):
    for cand in (os.path.join(root, "filtered_mmwave", scene, subj, act),
                 os.path.join(root, scene, subj, act, "mmwave")):
        if os.path.isdir(cand):
            return cand
    return None


def load_mmwave_frames(d):
    frames = []
    for b in sorted(glob.glob(os.path.join(d, "frame*.bin"))):
        raw = np.frombuffer(open(b, "rb").read(), dtype=np.float64)
        frames.append(raw.reshape(-1, 5).astype(np.float32) if raw.size else
                      np.zeros((0, 5), np.float32))
    return frames


def voxelize_bev(points, bounds, vox=VOX):
    """Window points [P,5] -> [2,vox,vox] BEV grid: log-count + mean intensity."""
    grid = np.zeros((2, vox, vox), np.float32)
    if len(points) == 0:
        return grid
    (xmn, xmx), (ymn, ymx) = bounds
    xi = np.clip(((points[:, 0] - xmn) / (xmx - xmn + 1e-6) * vox).astype(int), 0, vox - 1)
    yi = np.clip(((points[:, 1] - ymn) / (ymx - ymn + 1e-6) * vox).astype(int), 0, vox - 1)
    inten = points[:, 4]
    for k in range(len(points)):
        grid[0, xi[k], yi[k]] += 1.0
        grid[1, xi[k], yi[k]] += inten[k]
    cnt = np.maximum(grid[0], 1.0)
    grid[1] /= cnt
    grid[0] = np.log1p(grid[0])
    return grid


def subsample_idx(n, t):
    if n <= 0:
        return np.zeros(t, int)
    return np.linspace(0, n - 1, t).round().astype(int)


def process(root, out, max_subjects=None):
    subj_scene = list_subjects(root)
    have_depth_sk = len(subj_scene) > 0  # E0x dirs present -> depth+skeleton
    # mmwave can come from filtered_mmwave even without E0x: enumerate from there
    if not subj_scene:
        fm = os.path.join(root, "filtered_mmwave")
        for scene in sorted(os.listdir(fm)) if os.path.isdir(fm) else []:
            for subj in sorted(os.listdir(os.path.join(fm, scene))):
                subj_scene[subj] = scene
    subjects = sorted(subj_scene)[: max_subjects] if max_subjects else sorted(subj_scene)
    print(f"subjects={len(subjects)} depth/skeleton={'yes' if have_depth_sk else 'NO (mmwave only)'}")

    # global mmWave spatial bounds (robust percentiles over a sample)
    sx, sy = [], []
    for subj in subjects[:3]:
        for a in range(1, N_ACT + 1):
            d = mmwave_dir(root, subj_scene[subj], subj, f"A{a:02d}")
            if not d:
                continue
            for fr in load_mmwave_frames(d)[:30]:
                if len(fr):
                    sx.append(fr[:, 0]); sy.append(fr[:, 1])
    sx = np.concatenate(sx) if sx else np.array([0, 5.0])
    sy = np.concatenate(sy) if sy else np.array([-2.0, 2.0])
    bounds = ((np.percentile(sx, 1), np.percentile(sx, 99)),
              (np.percentile(sy, 1), np.percentile(sy, 99)))
    print("mmWave BEV bounds x=%.2f..%.2f y=%.2f..%.2f" %
          (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]))

    XSK, XMM, XDP, Y, SUBJ = [], [], [], [], []
    try:
        import cv2
    except ImportError:
        cv2 = None
    for si, subj in enumerate(subjects):
        scene = subj_scene[subj]
        for a in range(1, N_ACT + 1):
            act = f"A{a:02d}"
            mmd = mmwave_dir(root, scene, subj, act)
            mm = load_mmwave_frames(mmd) if mmd else []
            gt_path = os.path.join(root, scene, subj, act, "ground_truth.npy")
            sk = np.load(gt_path).astype(np.float32) if (have_depth_sk and os.path.exists(gt_path)) else None
            depth_dir = os.path.join(root, scene, subj, act, "depth")
            depth_files = sorted(glob.glob(os.path.join(depth_dir, "frame*.png"))) \
                if (have_depth_sk and os.path.isdir(depth_dir)) else []
            # number of frames = min across present modalities
            counts = [c for c in [len(mm), (len(sk) if sk is not None else 0),
                                  len(depth_files)] if c > 0]
            if not counts:
                continue
            F = min(counts)
            for s in range(0, max(F - W, 0) + 1, STRIDE):
                e = s + W
                if e > F:
                    break
                if mm:
                    pts = np.concatenate(mm[s:e], 0) if any(len(f) for f in mm[s:e]) \
                        else np.zeros((0, 5), np.float32)
                    XMM.append(voxelize_bev(pts, bounds))
                if sk is not None:
                    idx = s + subsample_idx(W, T_SK)
                    win = sk[idx]                      # [T_SK,17,3]
                    win = win - win[:, 0:1, :]         # center on root joint
                    XSK.append(win.reshape(T_SK, -1).T.astype(np.float32))  # [51,T_SK]
                if depth_files:
                    mid = depth_files[s + W // 2]
                    img = cv2.imread(mid, cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
                    img = cv2.resize(img, (DEPTH_HW, DEPTH_HW))
                    XDP.append(img[None].astype(np.float32))   # [1,H,W]
                Y.append(a - 1)
                SUBJ.append(int(subj[1:]))
        print(f"  [{si+1}/{len(subjects)}] {subj}: samples so far={len(Y)}", flush=True)

    data = dict(y=np.array(Y, np.int64), subj=np.array(SUBJ, np.int64))
    if XMM:
        data["xmm"] = np.stack(XMM).astype(np.float32)
    if XSK:
        data["xsk"] = np.stack(XSK).astype(np.float32)
    if XDP:
        data["xdp"] = np.stack(XDP).astype(np.float32)
    np.savez_compressed(out, **data)
    print("saved", out)
    for k, v in data.items():
        print(f"  {k}: {v.shape} {v.dtype}")
    print("classes present:", sorted(set(Y.tolist() if hasattr(Y, 'tolist') else Y)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/data/mmfi")
    ap.add_argument("--out", default="results/data/mmfi_cache.npz")
    ap.add_argument("--max-subjects", type=int, default=None)
    args = ap.parse_args()
    process(args.root, args.out, args.max_subjects)
