"""Aria field groups: captures of one smear field found by SIFT + RANSAC + overlap NCC (`overlap_groups`)."""
import os
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def _features(path: str, annotation_box: Optional[Sequence[float]]):
    """Grey image, its valid-pixel mask (annotation corner excluded) and SIFT keypoints / descriptors."""
    import cv2
    grey = np.asarray(Image.open(path).convert("L"))
    valid = np.ones(grey.shape, np.uint8)
    if annotation_box is not None:
        h, w = grey.shape
        y0, y1, x0, x1 = annotation_box
        valid[int(y0 * h):int(round(y1 * h)), int(x0 * w):int(round(x1 * w))] = 0
    keypoints, desc = cv2.SIFT_create(nfeatures=400).detectAndCompute(grey, valid)
    return grey.astype(np.float32), valid, np.float32([k.pt for k in keypoints]), desc


def _shared_field(fa, fb, min_inliers: int) -> Tuple[int, float, float]:
    """(RANSAC inliers, overlap fraction, overlap NCC) of image b warped onto image a by a similarity transform."""
    import cv2
    (ga, va, pa, da), (gb, vb, pb, db) = fa, fb
    if da is None or db is None or len(da) < 3 or len(db) < 3:
        return 0, 0.0, 0.0
    good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return 0, 0.0, 0.0
    M, inl = cv2.estimateAffinePartial2D(pb[[m.trainIdx for m in good]], pa[[m.queryIdx for m in good]],
                                         method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or int(inl.sum()) < min_inliers:
        return 0 if M is None else int(inl.sum()), 0.0, 0.0
    h, w = ga.shape
    warped = cv2.warpAffine(gb, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=-1)
    warped_valid = cv2.warpAffine(vb.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST) > 0.5
    overlap = (warped >= 0) & warped_valid & (va > 0)
    if overlap.sum() < 500:
        return int(inl.sum()), float(overlap.mean()), 0.0
    x, y = ga[overlap] - ga[overlap].mean(), warped[overlap] - warped[overlap].mean()
    return int(inl.sum()), float(overlap.mean()), float(x @ y / max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-6))


def overlap_groups(paths: Sequence[str], labels: Sequence[int], min_inliers: int, min_ncc: float, min_overlap: float,
                   annotation_box: Optional[Sequence[float]] = None, threads: int = 0) -> np.ndarray:
    """Group id per image: connected components of same-class pairs that show the same smear field.

    Aria et al. ship repeated captures of one field under nearby ids, shifted by a few to ~50 px, so a whole-frame
    similarity misses most of them. Two images are linked when SIFT + RANSAC finds a similarity transform with
    >= `min_inliers` inliers and, after warping, they share >= `min_overlap` of the frame with NCC > `min_ncc`
    (distinct fields of one class: overlap NCC p99 0.18). Every same-class pair is tested (~1.4M, minutes on 12
    cores); OpenCV runs single-threaded per call, so the groups are deterministic.
    """
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    cv2.setNumThreads(1)
    feats = [_features(p, annotation_box) for p in paths]
    labels = np.asarray(labels)
    pairs = [(i, j) for i in range(len(paths)) for j in range(i + 1, len(paths)) if labels[i] == labels[j]]
    with ThreadPoolExecutor(threads or os.cpu_count()) as ex:
        scores = list(ex.map(lambda ij: _shared_field(feats[ij[0]], feats[ij[1]], min_inliers), pairs, chunksize=256))

    parent = np.arange(len(paths))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (i, j), (inliers, overlap, ncc) in zip(pairs, scores):
        if inliers >= min_inliers and overlap >= min_overlap and ncc > min_ncc:
            parent[root(i)] = root(j)
    return np.array([root(i) for i in range(len(paths))])
