#!/usr/bin/env python3
"""
Raster → Vector (B/W) converter for pen plotters / wall painting robots.

Given a colorful raster image, this script produces a BLACK-AND-WHITE vector
SVG composed of pen-friendly polylines. Two vectorization modes are included:

1) centerline  – skeletonizes dark regions to produce single-stroke lines
2) outline     – traces filled region boundaries as polylines

Typical pipeline for plotter-friendly output:
- Preprocess → grayscale + threshold (adaptive or Otsu)
- (centerline) Thin foreground to 1px skeleton, trace to polylines
- (outline)    Find and simplify shape contours
- Scale to desired physical size (mm) and write SVG with round caps/joins

Usage examples:
  python raster_to_vector_plotter.py input.jpg out.svg \
      --mode centerline --width-mm 1000 --margin-mm 15 --simplify-epsilon-mm 0.6

  python raster_to_vector_plotter.py input.png out.svg \
      --mode outline --threshold otsu --width-mm 800 --simplify-epsilon-mm 0.8

Install deps (Python ≥3.9 recommended):
  pip install opencv-python scikit-image svgwrite numpy

Notes:
- For inverted art (light lines on dark bg), use --invert
- Increase --simplify-epsilon-mm to reduce nodes for faster plotting
- Use --min-length-mm to drop tiny fragments/noise
- If you know your plotter pen width, set --stroke-mm to match
"""

import argparse
import math
import os
from typing import Iterable, List, Sequence, Tuple, Dict, Set

import numpy as np
import cv2
import svgwrite

try:
    from skimage.morphology import skeletonize
except Exception as e:  # pragma: no cover
    skeletonize = None

Point = Tuple[int, int]
FloatPoint = Tuple[float, float]
Polyline = List[FloatPoint]


def imread_grayscale(path: str, max_side: int | None) -> np.ndarray:
    """Read image as grayscale uint8 [0..255]. Optionally downscale to max_side px.
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray


def threshold_bw(gray: np.ndarray,
                 method: str = "adaptive",
                 block_size: int = 35,
                 C: int = 10,
                 invert: bool = False,
                 blur_ksize: int = 3,
                 use_otsu: bool = False) -> np.ndarray:
    """Return binary image (uint8 0/255) foreground=white, background=black."""
    g = gray.copy()
    if blur_ksize and blur_ksize > 1 and blur_ksize % 2 == 1:
        g = cv2.GaussianBlur(g, (blur_ksize, blur_ksize), 0)

    if use_otsu or method.lower() == "otsu":
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        bs = block_size if block_size % 2 == 1 else block_size + 1
        th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, bs, C)

    if invert:
        th = cv2.bitwise_not(th)

    return th


# ------------------------------ Outline mode ---------------------------------

def find_outline_polylines(binary: np.ndarray,
                           simplify_eps_px: float,
                           min_len_px: float) -> List[Polyline]:
    """Trace contours of white regions and simplify to polylines."""
    # Ensure white = foreground
    # Remove speckle noise
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours of white regions (foreground)
    contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    polylines: List[Polyline] = []
    for cnt in contours:
        if len(cnt) < 2:
            continue
        # Simplify contour (closed). epsilon controls simplification in pixels.
        approx = cv2.approxPolyDP(cnt, simplify_eps_px, True)
        pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        # Close explicitly for consistency
        if len(pts) >= 2 and (pts[0][0] != pts[-1][0] or pts[0][1] != pts[-1][1]):
            pts.append(pts[0])
        if polyline_length_px(pts) >= min_len_px:
            polylines.append(pts)

    return polylines


# ---------------------------- Centerline mode --------------------------------

def neighbors8(p: Point) -> List[Point]:
    x, y = p
    return [
        (x-1, y-1), (x, y-1), (x+1, y-1),
        (x-1, y),             (x+1, y),
        (x-1, y+1), (x, y+1), (x+1, y+1),
    ]


def build_adjacency_from_skeleton(skel: np.ndarray) -> Dict[Point, List[Point]]:
    h, w = skel.shape
    adj: Dict[Point, List[Point]] = {}
    on = np.argwhere(skel > 0)
    on_set: Set[Point] = set((int(x), int(y)) for y, x in on)  # note argwhere returns (y,x)

    for x, y in on_set:
        nbrs: List[Point] = []
        for nx, ny in neighbors8((x, y)):
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) in on_set:
                nbrs.append((nx, ny))
        adj[(x, y)] = nbrs
    return adj


def degree(adj: Dict[Point, List[Point]], p: Point) -> int:
    return len(adj.get(p, []))


def rdp(points: Sequence[FloatPoint], eps: float) -> List[FloatPoint]:
    """Ramer–Douglas–Peucker for polylines (2D)."""
    if len(points) < 3 or eps <= 0:
        return list(points)

    def perp_dist(pt: FloatPoint, a: FloatPoint, b: FloatPoint) -> float:
        (x, y), (x1, y1), (x2, y2) = pt, a, b
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(x - x1, y - y1)
        return abs(dy * x - dx * y + x2*y1 - y2*x1) / math.hypot(dx, dy)

    def _rdp(pts: Sequence[FloatPoint]) -> List[FloatPoint]:
        if len(pts) < 3:
            return list(pts)
        a, b = pts[0], pts[-1]
        max_d, idx = 0.0, -1
        for i in range(1, len(pts) - 1):
            d = perp_dist(pts[i], a, b)
            if d > max_d:
                max_d, idx = d, i
        if max_d > eps:
            left = _rdp(pts[:idx+1])
            right = _rdp(pts[idx:])
            return left[:-1] + right
        else:
            return [a, b]

    return _rdp(list(points))


def trace_skeleton_to_polylines(skel: np.ndarray,
                                simplify_eps_px: float,
                                min_len_px: float) -> List[Polyline]:
    """Trace 8-connected skeleton pixels into polylines and simplify."""
    adj = build_adjacency_from_skeleton(skel)
    if not adj:
        return []

    # Edge-visited set: store undirected edges as frozenset of two nodes
    visited: Set[frozenset] = set()

    def take_edge(u: Point, v: Point):
        visited.add(frozenset((u, v)))

    def edge_seen(u: Point, v: Point) -> bool:
        return frozenset((u, v)) in visited

    starts: List[Point] = []
    for p in adj.keys():
        d = degree(adj, p)
        if d != 2:
            starts.append(p)

    polylines: List[Polyline] = []

    def walk_from(start: Point, nxt: Point | None) -> List[Point]:
        path: List[Point] = [start]
        prev: Point | None = None
        cur: Point = start
        if nxt is not None:
            path.append(nxt)
            take_edge(cur, nxt)
            prev, cur = cur, nxt
        while True:
            nbrs = [q for q in adj.get(cur, []) if q != prev]
            # choose an unvisited edge if possible
            nxts = [q for q in nbrs if not edge_seen(cur, q)]
            if not nxts:
                break
            # Prefer continuing straight-ish: pick neighbor closest to prev→cur direction
            if prev is not None and len(nxts) > 1:
                vx, vy = cur[0] - prev[0], cur[1] - prev[1]
                def score(q: Point) -> float:
                    qx, qy = q[0] - cur[0], q[1] - cur[1]
                    dot = vx*qx + vy*qy
                    norm = math.hypot(vx, vy) * math.hypot(qx, qy) + 1e-9
                    return -(dot / norm)
                nxts.sort(key=score)
            n = nxts[0]
            take_edge(cur, n)
            path.append(n)
            prev, cur = cur, n
            if degree(adj, cur) != 2:
                break
        return path

    # 1) Start from endpoints/junctions so we get open strokes first
    for s in starts:
        for n in adj.get(s, []):
            if not edge_seen(s, n):
                pts = walk_from(s, n)
                if len(pts) >= 2:
                    poly = [(float(x), float(y)) for (x, y) in pts]
                    poly = rdp(poly, simplify_eps_px)
                    if polyline_length_px(poly) >= min_len_px:
                        polylines.append(poly)

    # 2) Pick up any remaining edges in cycles (all degree==2)
    for s in adj.keys():
        for n in adj.get(s, []):
            if not edge_seen(s, n):
                pts = walk_from(s, n)
                if len(pts) >= 2:
                    poly = [(float(x), float(y)) for (x, y) in pts]
                    poly = rdp(poly, simplify_eps_px)
                    if polyline_length_px(poly) >= min_len_px:
                        polylines.append(poly)

    return polylines


# ---------------------------- Geometry helpers -------------------------------

def polyline_length_px(poly: Sequence[FloatPoint]) -> float:
    if len(poly) < 2:
        return 0.0
    return float(sum(math.hypot(poly[i+1][0]-poly[i][0], poly[i+1][1]-poly[i][1])
                     for i in range(len(poly)-1)))


def nearest_order(polys: List[Polyline]) -> List[Polyline]:
    """Greedy reorder polylines to reduce pen-up moves. Reverses paths if helpful."""
    if not polys:
        return polys

    def endpoints(p: Polyline) -> Tuple[FloatPoint, FloatPoint]:
        return p[0], p[-1]

    remaining = polys.copy()
    # start from the longest path for robustness
    remaining.sort(key=polyline_length_px, reverse=True)
    ordered: List[Polyline] = [remaining.pop(0)]

    while remaining:
        _, cur_end = endpoints(ordered[-1])
        best_i, best_flip, best_dist = 0, False, float('inf')
        for i, cand in enumerate(remaining):
            s, e = endpoints(cand)
            ds = math.hypot(cur_end[0]-s[0], cur_end[1]-s[1])
            de = math.hypot(cur_end[0]-e[0], cur_end[1]-e[1])
            if ds < best_dist:
                best_i, best_flip, best_dist = i, False, ds
            if de < best_dist:
                best_i, best_flip, best_dist = i, True, de
        nxt = remaining.pop(best_i)
        if best_flip:
            nxt = list(reversed(nxt))
        ordered.append(nxt)

    return ordered


# ------------------------------ SVG writing ----------------------------------

def save_svg(polylines: List[Polyline],
             svg_path: str,
             img_w: int,
             img_h: int,
             width_mm: float,
             height_mm: float | None,
             margin_mm: float,
             stroke_mm: float,
             optimize_order: bool) -> None:
    if not polylines:
        raise RuntimeError("No vector paths generated. Try adjusting thresholding or mode.")

    if optimize_order:
        polylines = nearest_order(polylines)

    # Maintain aspect ratio and fit within (width_mm - 2*margin) x (height_mm - 2*margin)
    if height_mm is None:
        # Preserve aspect using width only
        height_mm = (width_mm * img_h) / img_w

    avail_w = max(1e-6, width_mm - 2*margin_mm)
    avail_h = max(1e-6, height_mm - 2*margin_mm)

    sx = avail_w / img_w
    sy = avail_h / img_h
    s = min(sx, sy)  # contain

    actual_w = img_w * s
    actual_h = img_h * s
    offx = (width_mm - actual_w) / 2.0
    offy = (height_mm - actual_h) / 2.0

    dwg = svgwrite.Drawing(filename=svg_path,
                           size=(f"{width_mm}mm", f"{height_mm}mm"),
                           viewBox=f"0 0 {width_mm} {height_mm}")

    # Background transparent; paths in mm units
    for poly in polylines:
        if len(poly) < 2:
            continue
        pts_mm = [
            (offx + p[0] * s, offy + p[1] * s)
            for p in poly
        ]
        dwg.add(dwg.polyline(points=pts_mm,
                             stroke="#000",
                             fill="none",
                             stroke_width=f"{stroke_mm}mm",
                             stroke_linecap="round",
                             stroke_linejoin="round"))

    dwg.save()


# --------------------------------- Main --------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Raster to vector SVG for pen plotters")
    ap.add_argument("input", help="Input raster image (jpg/png/etc)")
    ap.add_argument("output", nargs="?", help="Output SVG path (default: input name with .svg)")

    ap.add_argument("--mode", choices=["centerline", "outline"], default="centerline",
                    help="Vectorization mode (default: centerline)")

    ap.add_argument("--width-mm", type=float, default=800.0, help="Output width in mm")
    ap.add_argument("--height-mm", type=float, default=None, help="Output height in mm (default: keep aspect)")
    ap.add_argument("--margin-mm", type=float, default=10.0, help="Margin on all sides in mm")
    ap.add_argument("--stroke-mm", type=float, default=0.35, help="SVG stroke width in mm")

    ap.add_argument("--invert", action="store_true", help="Invert thresholded image")
    ap.add_argument("--blur-ksize", type=int, default=3, help="Gaussian blur ksize (odd, 0/1 to disable)")
    ap.add_argument("--threshold", choices=["adaptive", "otsu"], default="adaptive",
                    help="Thresholding method before vectorization")
    ap.add_argument("--block-size", type=int, default=35, help="Adaptive threshold block size (odd)")
    ap.add_argument("--C", type=int, default=10, help="Adaptive threshold constant C")

    ap.add_argument("--simplify-epsilon-mm", type=float, default=0.5,
                    help="RDP simplification tolerance in mm")
    ap.add_argument("--min-length-mm", type=float, default=2.0,
                    help="Drop paths shorter than this length (mm)")

    ap.add_argument("--optimize-order", action="store_true",
                    help="Greedy reorder to reduce pen-up travel")

    ap.add_argument("--max-side", type=int, default=2000,
                    help="Resize longest image side to this many pixels (0 to disable)")

    args = ap.parse_args()

    out_path = args.output
    if not out_path:
        base, _ = os.path.splitext(args.input)
        out_path = base + ".svg"

    gray = imread_grayscale(args.input, None if args.max_side in (None, 0) else args.max_side)
    h, w = gray.shape

    binary = threshold_bw(
        gray,
        method=args.threshold,
        block_size=args.block_size,
        C=args.C,
        invert=args.invert,
        blur_ksize=args.blur_ksize,
        use_otsu=(args.threshold == "otsu")
    )

    # Convert physical mm tolerances to pixel units using the scale that will be applied later.
    # We don't know the final exact scale yet (contain vs aspect), so approximate using width.
    # This is sufficient for simplification/min filtering stability.
    # px_per_mm ≈ w / (width_mm - 2*margin)
    avail_w = max(1e-6, args.width_mm - 2*args.margin_mm)
    px_per_mm = w / avail_w
    simplify_eps_px = max(0.0, args.simplify_epsilon_mm * px_per_mm)
    min_len_px = max(0.0, args.min_length_mm * px_per_mm)

    if args.mode == "outline":
        polylines = find_outline_polylines(binary, simplify_eps_px, min_len_px)
    else:
        if skeletonize is None:
            raise RuntimeError(
                "scikit-image is required for centerline mode. Install with: pip install scikit-image"
            )
        # skeletonize expects boolean with foreground=True
        skel_bool = (binary > 0)
        skel = skeletonize(skel_bool).astype(np.uint8) * 255
        polylines = trace_skeleton_to_polylines(skel, simplify_eps_px, min_len_px)

    save_svg(polylines, out_path, w, h, args.width_mm, args.height_mm, args.margin_mm,
             args.stroke_mm, args.optimize_order)

    print(f"Saved SVG with {len(polylines)} paths → {out_path}")


if __name__ == "__main__":
    main()
