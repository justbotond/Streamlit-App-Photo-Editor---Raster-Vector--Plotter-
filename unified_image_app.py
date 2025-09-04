#!/usr/bin/env python3
"""
Streamlit Web App — Raster → Vector (B/W) for pen plotters / wall painting robots

• Python backend (OpenCV + scikit-image) with a reactive web UI (Streamlit)
• Upload an image and tune parameters; previews update automatically
• Shows Original, Grayscale, and After (vector preview) side-by-side
• Download clean SVG polylines (mm units) for downstream G-code tools
• Progress bar reflects each stage of computation

Run locally
-----------
1) Install deps (Python 3.9+ recommended):
   pip install streamlit opencv-python scikit-image svgwrite numpy pillow

2) Start the app:
   streamlit run streamlit_raster_to_vector_app.py

3) Open the browser page that Streamlit prints (usually http://localhost:8501)
"""

from __future__ import annotations
import math
from typing import List, Sequence, Tuple, Dict, Set

import numpy as np
import cv2
import streamlit as st
import svgwrite
from PIL import Image

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None

# ------------------------------ Types ----------------------------------------
Point = Tuple[int, int]
FloatPoint = Tuple[float, float]
Polyline = List[FloatPoint]

# -------------------------- Image utilities ----------------------------------

def load_image_to_bgr(file) -> np.ndarray:
    """Read uploaded file-like object into BGR numpy image."""
    data = file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return img

def ensure_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def threshold_bw(gray: np.ndarray,
                 method: str = "adaptive",
                 block_size: int = 35,
                 C: int = 10,
                 invert: bool = False,
                 blur_ksize: int = 3,
                 use_otsu: bool = False) -> np.ndarray:
    g = gray
    if blur_ksize and blur_ksize > 1 and blur_ksize % 2 == 1:
        g = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    if use_otsu or method.lower() == "otsu":
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        bs = block_size if block_size % 2 == 1 else block_size + 1
        th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, bs, C)
    if invert:
        th = cv2.bitwise_not(th)
    return th

# ------------------------------ Geometry -------------------------------------

def polyline_length_px(poly: Sequence[FloatPoint]) -> float:
    if len(poly) < 2:
        return 0.0
    return float(sum(
        math.hypot(poly[i+1][0]-poly[i][0], poly[i+1][1]-poly[i][1])
        for i in range(len(poly)-1)
    ))

# Outline mode

def find_outline_polylines(binary: np.ndarray,
                           simplify_eps_px: float,
                           min_len_px: float) -> List[Polyline]:
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polylines: List[Polyline] = []
    for cnt in contours:
        if len(cnt) < 2:
            continue
        approx = cv2.approxPolyDP(cnt, simplify_eps_px, True)
        pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(pts) >= 2 and (pts[0][0] != pts[-1][0] or pts[0][1] != pts[-1][1]):
            pts.append(pts[0])
        if polyline_length_px(pts) >= min_len_px:
            polylines.append(pts)
    return polylines

# Centerline mode helpers

def neighbors8(p: Point) -> List[Point]:
    x, y = p
    return [
        (x-1, y-1), (x, y-1), (x+1, y-1),
        (x-1, y),             (x+1, y),
        (x-1, y+1), (x, y+1), (x+1, y+1),
    ]

def build_adjacency_from_skeleton(skel: np.ndarray) -> Dict[Point, List[Point]]:
    h, w = skel.shape
    on = np.argwhere(skel > 0)
    on_set: Set[Point] = set((int(x), int(y)) for y, x in on)  # (x,y)
    adj: Dict[Point, List[Point]] = {}
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
    adj = build_adjacency_from_skeleton(skel)
    if not adj:
        return []
    visited: Set[frozenset] = set()
    def take_edge(u: Point, v: Point):
        visited.add(frozenset((u, v)))
    def edge_seen(u: Point, v: Point) -> bool:
        return frozenset((u, v)) in visited
    starts: List[Point] = []
    for p in adj.keys():
        if degree(adj, p) != 2:
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
            nxts = [q for q in nbrs if not edge_seen(cur, q)]
            if not nxts:
                break
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
    for s in starts:
        for n in adj.get(s, []):
            if not edge_seen(s, n):
                pts = walk_from(s, n)
                if len(pts) >= 2:
                    poly = [(float(x), float(y)) for (x, y) in pts]
                    poly = rdp(poly, simplify_eps_px)
                    if polyline_length_px(poly) >= min_len_px:
                        polylines.append(poly)
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

# Ordering & SVG

def nearest_order(polys: List[Polyline]) -> List[Polyline]:
    if not polys:
        return polys
    def endpoints(p: Polyline):
        return p[0], p[-1]
    remaining = polys.copy()
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

def svg_bytes(polylines: List[Polyline],
              img_w: int,
              img_h: int,
              width_mm: float,
              height_mm: float | None,
              margin_mm: float,
              stroke_mm: float,
              optimize_order_flag: bool) -> bytes:
    if not polylines:
        raise RuntimeError("No vector paths generated. Adjust parameters.")
    polys = nearest_order(polylines) if optimize_order_flag else polylines
    if height_mm is None:
        height_mm = (width_mm * img_h) / img_w
    avail_w = max(1e-6, width_mm - 2*margin_mm)
    avail_h = max(1e-6, height_mm - 2*margin_mm)
    sx = avail_w / img_w
    sy = avail_h / img_h
    s = min(sx, sy)
    actual_w = img_w * s
    actual_h = img_h * s
    offx = (width_mm - actual_w) / 2.0
    offy = (height_mm - actual_h) / 2.0
    dwg = svgwrite.Drawing(size=(f"{width_mm}mm", f"{height_mm}mm"),
                           viewBox=f"0 0 {width_mm} {height_mm}")
    for poly in polys:
        if len(poly) < 2:
            continue
        pts_mm = [(offx + p[0]*s, offy + p[1]*s) for p in poly]
        dwg.add(dwg.polyline(points=pts_mm,
                             stroke="#000",
                             fill="none",
                             stroke_width=f"{stroke_mm}mm",
                             stroke_linecap="round",
                             stroke_linejoin="round"))
    return dwg.tostring().encode("utf-8")

# ---------------------------- Vectorization stepper ---------------------------

def vectorize_with_progress(gray: np.ndarray,
                            progress_cb,
                            *,
                            mode: str,
                            thr_method: str,
                            invert: bool,
                            blur_ksize: int,
                            block_size: int,
                            C: int,
                            width_mm: float,
                            height_mm_opt: float,
                            margin_mm: float,
                            simplify_eps_mm: float,
                            min_len_mm: float,
                            stroke_mm: float,
                            optimize: bool):
    """Return (binary, polylines, preview_img_bgr, svg_bytes)."""

    # Step 1: threshold
    progress_cb(10, "Thresholding…")
    binary = threshold_bw(gray, method=thr_method, block_size=block_size, C=C,
                          invert=invert, blur_ksize=blur_ksize, use_otsu=(thr_method == "otsu"))

    # Step 2: compute px-per-mm for geometric params
    progress_cb(25, "Preparing geometry…")
    h, w = gray.shape
    avail_w = max(1e-6, width_mm - 2*margin_mm)
    px_per_mm = w / avail_w
    simplify_eps_px = max(0.0, simplify_eps_mm * px_per_mm)
    min_len_px = max(0.0, min_len_mm * px_per_mm)

    # Step 3: vectorization
    if mode == "outline":
        progress_cb(55, "Tracing contours…")
        polylines = find_outline_polylines(binary, simplify_eps_px, min_len_px)
    else:
        if skeletonize is None:
            raise RuntimeError("Install scikit-image for centerline mode: pip install scikit-image")
        progress_cb(45, "Skeletonizing…")
        skel = skeletonize(binary > 0).astype(np.uint8) * 255
        progress_cb(65, "Tracing centerlines…")
        polylines = trace_skeleton_to_polylines(skel, simplify_eps_px, min_len_px)

    # Step 4: preview rendering
    progress_cb(80, "Rendering preview…")
    preview = np.zeros((h, w, 3), dtype=np.uint8)
    px_thick = max(1, int(round(stroke_mm * 2)))
    for poly in polylines:
        if len(poly) >= 2:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(preview, [pts], isClosed=False, color=(255, 255, 255), thickness=px_thick)

    # Step 5: SVG
    progress_cb(95, "Building SVG…")
    height_mm = None if height_mm_opt <= 0 else height_mm_opt
    svg = svg_bytes(polylines, w, h, width_mm, height_mm, margin_mm, stroke_mm, optimize)

    progress_cb(100, "Done")
    return binary, polylines, preview, svg

# --------------------------------- UI ----------------------------------------

st.set_page_config(page_title="Raster → Vector (Plotter)", layout="wide")
st.title("Raster → Vector (B/W) — Plotter-friendly Web App")

with st.sidebar:
    st.header("Input & Quality")
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","tif","tiff","webp"])
    max_side = st.slider("Resize longest side (px)", 400, 4000, 1400, 100,
                         help="Larger = more detail but slower.")

    st.header("Vectorization")
    mode = st.radio("Mode", ["centerline","outline"], index=0, help="Centerline=skeleton; Outline=contours")
    thr_method = st.radio("Threshold", ["adaptive","otsu"], index=0)
    invert = st.checkbox("Invert", value=False)
    blur_ksize = st.slider("Blur ksize", 0, 15, 3, 1)
    block_size = st.slider("Adaptive block size (odd)", 3, 101, 35, 2)
    C = st.slider("Adaptive C", -30, 30, 10, 1)

    st.header("Geometry / Export (mm)")
    width_mm = st.slider("Width", 100, 2000, 800, 10)
    height_mm_opt = st.slider("Height (0=auto)", 0, 2000, 0, 10)
    margin_mm = st.slider("Margin", 0, 100, 10, 1)
    stroke_mm = st.slider("Stroke preview", 0.0, 2.0, 0.35, 0.05)

    st.header("Simplify / Filter")
    simplify_eps_mm = st.slider("Simplify epsilon", 0.0, 5.0, 0.5, 0.05, help="Higher = fewer nodes")
    min_len_mm = st.slider("Min path length", 0.0, 20.0, 2.0, 0.5)
    optimize = st.checkbox("Optimize pen-up travel", value=True)

# Main preview area
col1, col2, col3 = st.columns(3)

if uploaded is None:
    col1.info("Upload an image to begin")
else:
    pbar = st.progress(0, text="Starting…")

    def step(pct, msg):
        pbar.progress(pct, text=msg)

    img_bgr = load_image_to_bgr(uploaded)
    img_bgr = ensure_max_side(img_bgr, max_side)
    gray = to_grayscale(img_bgr)

    # Show Original & Gray immediately
    col1.subheader("Original")
    col1.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    col2.subheader("Grayscale")
    col2.image(gray, clamp=True, use_container_width=True)

    try:
        binary, polylines, preview, svg = vectorize_with_progress(
            gray, step,
            mode=mode,
            thr_method=thr_method,
            invert=invert,
            blur_ksize=blur_ksize,
            block_size=block_size if block_size % 2 == 1 else block_size + 1,
            C=C,
            width_mm=width_mm,
            height_mm_opt=height_mm_opt,
            margin_mm=margin_mm,
            simplify_eps_mm=simplify_eps_mm,
            min_len_mm=min_len_mm,
            stroke_mm=stroke_mm,
            optimize=optimize,
        )

        col3.subheader("After (Vector preview)")
        col3.image(preview, channels="BGR", use_container_width=True)

        st.divider()
        st.download_button("⬇️ Download SVG", data=svg, file_name="vectorized.svg",
                           mime="image/svg+xml")
        st.caption(f"Paths: {len(polylines)}  ·  Image: {gray.shape[1]}×{gray.shape[0]} px")
    except Exception as e:
        st.error(str(e))
    finally:
        pbar.progress(100, text="Ready")
