#!/usr/bin/env python3
"""
Unified Streamlit App — Photo Editor + Raster→Vector (Plotter)

Features
- Photo Editor (Lightroom-style) with:
  • Upload or Image URL
  • ✨ Auto Grade button (automatic color grading)
  • ↩️ Reset button (shows original image)
- Raster → Vector for pen plotters, SVG download

Notes
- Use opencv-python-headless on Streamlit Cloud.
"""

from __future__ import annotations
import math
from typing import List, Sequence, Tuple, Dict, Set

import numpy as np
import streamlit as st  # import Streamlit first so we can render errors

# Try OpenCV; show a helpful message if the headless wheel isn't installed
try:
    import cv2
except Exception as e:
    st.set_page_config(page_title="Unified Image App", layout="wide")
    st.title("Unified Image App — Photo Editor & Raster→Vector")
    st.error(
        "OpenCV (cv2) is not available.\n\n"
        "On Streamlit Cloud, depend on **opencv-python-headless** (not opencv-python).\n"
        "Update your `requirements.txt` to include:\n\n"
        "    opencv-python-headless>=4.8\n\n"
        "Remove `opencv-python` if present, then redeploy.\n\n"
        f"Original import error: {type(e).__name__}: {e}"
    )
    st.stop()

import svgwrite
from PIL import Image
import requests
import streamlit.components.v1 as components

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None

# ------------------------------ Types ----------------------------------------
Point = Tuple[int, int]
FloatPoint = Tuple[float, float]
Polyline = List[FloatPoint]

# ============================== AdSense Helper ===============================

def render_adsense(slot_key: str, *, height: int = 120):
    """Render a responsive AdSense slot if secrets are provided."""
    ads = st.secrets.get("adsense", {})
    client = ads.get("client")
    slot = ads.get(slot_key)
    if not client or not slot:
        return
    components.html(
        f"""
        <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={client}" crossorigin="anonymous"></script>
        <ins class="adsbygoogle"
            style="display:block"
            data-ad-client="{client}"
            data-ad-slot="{slot}"
            data-ad-format="auto"
            data-full-width-responsive="true"></ins>
        <script>(adsbygoogle = window.adsbygoogle || []).push({{}});</script>
        """,
        height=height,
    )

# ============================== Utility Helpers ==============================

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0, 255)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img

def load_image_to_bgr_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data.")
    return img

def load_image_to_bgr(file) -> np.ndarray:
    return load_image_to_bgr_from_bytes(file.read())

def load_image_from_url(url: str, timeout: float = 15.0) -> np.ndarray:
    if not url or not isinstance(url, str):
        raise ValueError("Empty URL.")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL must start with http:// or https://")
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return load_image_to_bgr_from_bytes(resp.content)

def ensure_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

# =============================== Photo Editor ================================

def apply_basic(img_bgr: np.ndarray,
                exposure: float, contrast: float,
                highlights: float, shadows: float,
                whites: float, blacks: float) -> np.ndarray:
    img = img_bgr.astype(np.float32) / 255.0
    img = np.clip(img * (2.0 ** exposure), 0, 1)
    c = contrast
    if abs(c) > 1e-6:
        gamma = 1.0 / (1.0 + c) if c > 0 else 1.0 - c*0.5
        img = np.clip((img ** gamma), 0, 1)
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[...,0], lab[...,1], lab[...,2]
    Ln = L / 255.0
    if abs(highlights) > 1e-6:
        mask_h = np.clip((Ln - 0.5)*2, 0, 1)
        L = L + highlights*50.0*mask_h
    if abs(shadows) > 1e-6:
        mask_s = np.clip((0.5 - Ln)*2, 0, 1)
        L = L + shadows*50.0*mask_s
    if abs(whites) > 1e-6:
        L = np.clip(L + whites*40.0*np.power(np.clip(L/255.0,0,1), 2.0), 0, 255)
    if abs(blacks) > 1e-6:
        L = np.clip(L + blacks*40.0*np.power(1.0 - np.clip(L/255.0,0,1), 2.0), 0, 255)
    lab = np.stack([np.clip(L,0,255), A, B], axis=-1)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out

def apply_white_balance(img_bgr: np.ndarray, temp: float, tint: float) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    r_gain = 1.0 + 0.01*temp
    b_gain = 1.0 - 0.01*temp
    g_gain = 1.0 + 0.01*tint
    gains = np.array([b_gain, g_gain, r_gain], dtype=np.float32)
    out = img * gains
    return _ensure_uint8(out)

def apply_vibrance_saturation(img_bgr: np.ndarray, vibrance: float, saturation: float) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H,S,V = hsv[...,0], hsv[...,1], hsv[...,2]
    if abs(vibrance) > 1e-6:
        factor = 1.0 + (vibrance/100.0) * (1.0 - (S/255.0))
        S = np.clip(S * factor, 0, 255)
    if abs(saturation) > 1e-6:
        S = np.clip(S * (1.0 + saturation/100.0), 0, 255)
    hsv = np.stack([H,S,V], axis=-1)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_hsl(img_bgr: np.ndarray, h_adj: dict, s_adj: dict, l_adj: dict) -> np.ndarray:
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
    H,L,S = hls[...,0], hls[...,1], hls[...,2]
    Hn = (H * 2.0)
    ranges = {
        'reds': [(345,360),(0,15)],
        'oranges': [(15,45)],
        'yellows': [(45,75)],
        'greens': [(75,165)],
        'aquas': [(165,195)],
        'blues': [(195,255)],
        'purples': [(255,285)],
        'magentas': [(285,345)],
    }
    H2 = Hn
    for name, spans in ranges.items():
        m = np.zeros_like(H, dtype=np.uint8)
        for a,b in spans:
            if a <= b:
                m |= ((H2 >= a) & (H2 < b)).astype(np.uint8)
            else:
                m |= ((H2 >= a) | (H2 < b)).astype(np.uint8)
        if name in h_adj:
            H2 = (H2 + m*h_adj[name]) % 360
        if name in s_adj:
            S = np.clip(S * (1.0 + (m*s_adj[name]/100.0)), 0, 255)
        if name in l_adj:
            L = np.clip(L + (m*l_adj[name]), 0, 255)
    H_out = (H2/2.0).astype(np.float32)
    out = cv2.cvtColor(np.stack([H_out,L,S], axis=-1).astype(np.uint8), cv2.COLOR_HLS2BGR)
    return out

def apply_detail(img_bgr: np.ndarray, sharpening: float, noise_red: float, texture: float, clarity: float, dehaze: float) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    if noise_red > 0:
        h = 5 + int(noise_red*3)
        img = cv2.fastNlMeansDenoisingColored(_ensure_uint8(img), None, h, h, 7, 21).astype(np.float32)
    if abs(clarity) > 1e-6:
        l = cv2.cvtColor(_ensure_uint8(img), cv2.COLOR_BGR2LAB).astype(np.float32)
        L = l[...,0]
        blur = cv2.GaussianBlur(L, (0,0), 3)
        high = L - blur
        L = np.clip(L + clarity*high, 0, 255)
        l[...,0] = L
        img = cv2.cvtColor(l.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    if abs(texture) > 1e-6:
        blur = cv2.GaussianBlur(_ensure_uint8(img), (0,0), 1.0)
        high = _ensure_uint8(img) - blur
        img = np.clip(img + texture*high, 0, 255)
    if sharpening > 0:
        sigma = 1.0 + 2.0*sharpening
        blur = cv2.GaussianBlur(_ensure_uint8(img), (0,0), sigma)
        img = np.clip(1.5*_ensure_uint8(img) - 0.5*blur, 0, 255)
    if abs(dehaze) > 1e-6:
        hsv = cv2.cvtColor(_ensure_uint8(img), cv2.COLOR_BGR2HSV).astype(np.float32)
        V = hsv[...,2]
        alpha = 1.0 + 0.02*dehaze
        beta = -10.0 * (dehaze/100.0)
        V = np.clip(V*alpha + beta, 0, 255)
        hsv[...,2] = V
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return _ensure_uint8(img)

def apply_crop_rotate(img_bgr: np.ndarray, angle_deg: float,
                      crop_l: float, crop_r: float, crop_t: float, crop_b: float,
                      target_aspect: float|None) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if abs(angle_deg) > 1e-3:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x0 = int(w*crop_l/100.0)
    x1 = int(w*(1.0 - crop_r/100.0))
    y0 = int(h*crop_t/100.0)
    y1 = int(h*(1.0 - crop_b/100.0))
    x0,x1 = max(0,x0), max(x0+1,min(w,x1))
    y0,y1 = max(0,y0), max(y0+1,min(h,y1))
    img_bgr = img_bgr[y0:y1, x0:x1]
    if target_aspect and img_bgr.size:
        hh, ww = img_bgr.shape[:2]
        cur = ww/float(hh)
        if abs(cur - target_aspect) > 1e-3:
            if cur > target_aspect:
                new_w = int(target_aspect*hh)
                off = (ww - new_w)//2
                img_bgr = img_bgr[:, off:off+new_w]
            else:
                new_h = int(ww/target_aspect)
                off = (hh - new_h)//2
                img_bgr = img_bgr[off:off+new_h, :]
    return img_bgr

def apply_lens_geometry(img_bgr: np.ndarray, vignetting: float, ca_shift: float, distortion: float,
                        persp_x: float, persp_y: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    if abs(vignetting) > 1e-6:
        Y, X = np.ogrid[:h, :w]
        cx, cy = w/2, h/2
        r = np.sqrt((X-cx)**2 + (Y-cy)**2)
        r /= r.max()
        mask = 1.0 + vignetting/100.0 * (1 - r)
        out = np.clip(out.astype(np.float32) * mask[...,None], 0, 255).astype(np.uint8)
    if abs(ca_shift) > 1e-6:
        shift = int(round(ca_shift))
        def shift_channel(ch, dx, dy):
            M = np.float32([[1,0,dx],[0,1,dy]])
            return cv2.warpAffine(ch, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        b,g,r = cv2.split(out)
        out = cv2.merge([shift_channel(b,-shift,0), g, shift_channel(r,shift,0)])
    if abs(distortion) > 1e-6:
        k1 = distortion/10000.0
        xx, yy = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
        rr = xx*xx + yy*yy
        x_dist = xx*(1 + k1*rr)
        y_dist = yy*(1 + k1*rr)
        map_x = ((x_dist + 1)*0.5*(w-1)).astype(np.float32)
        map_y = ((y_dist + 1)*0.5*(h-1)).astype(np.float32)
        out = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if abs(persp_x) > 1e-6 or abs(persp_y) > 1e-6:
        dx = w * persp_x / 100.0
        dy = h * persp_y / 100.0
        src = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst = np.float32([[0+dx,0+dy],[w-dx,0+dy],[w-dx,h-dy],[0+dx,h-dy]])
        M = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out

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

def polyline_length_px(poly: Sequence[FloatPoint]) -> float:
    if len(poly) < 2:
        return 0.0
    return float(sum(math.hypot(poly[i+1][0]-poly[i][0], poly[i+1][1]-poly[i][1])
                     for i in range(len(poly)-1)))

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
    on_set: Set[Point] = set((int(x), int(y)) for y, x in on)
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
              img_w: int, img_h: int,
              width_mm: float, height_mm: float | None,
              margin_mm: float, stroke_mm: float,
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
    dwg = svgwrite.Drawing(size=(f"{width_mm}mm", f"{height_mm}mm"), viewBox=f"0 0 {width_mm} {height_mm}")
    for poly in polys:
        if len(poly) < 2:
            continue
        pts_mm = [(offx + p[0]*s, offy + p[1]*s) for p in poly]
        dwg.add(dwg.polyline(points=pts_mm,
                             stroke="#000", fill="none",
                             stroke_width=f"{stroke_mm}mm",
                             stroke_linecap="round", stroke_linejoin="round"))
    return dwg.tostring().encode("utf-8")

# ============================ Auto Color Grading ==============================

def auto_color_grade(img_bgr: np.ndarray) -> np.ndarray:
    """Simple auto color grading:
    1) Gray-world white balance
    2) CLAHE on LAB L channel
    3) Gentle saturation boost
    """
    img = img_bgr.astype(np.float32)
    # 1) Gray-world WB
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6
    g_mean = float(means.mean())
    gains = g_mean / means
    img = np.clip(img * gains[None, None, :], 0, 255).astype(np.uint8)

    # 2) CLAHE on L
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    lab2 = cv2.merge([Lc, A, B])
    img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # 3) gentle saturation boost
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * 1.08, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

# ================================ UI LAYOUT ==================================

st.set_page_config(page_title="Unified Image App", layout="wide")
st.title("Unified Image App — Photo Editor & Raster→Vector")

# Top AdSense slot
render_adsense("slot_top", height=120)

with st.sidebar:
    st.header("Choose Feature")
    app_mode = st.radio("Application", ["Photo Editor", "Raster → Vector"], index=0)
    st.divider()

# Small helper to choose input source uniformly across modes
def get_input_image_controls(accept_types: list[str], default_max_side: int):
    st.subheader("Input")
    input_source = st.radio("Image source", ["Upload", "Image URL"], horizontal=True, key=f"src_{app_mode}")
    uploaded = None
    url = ""
    if input_source == "Upload":
        uploaded = st.file_uploader("Upload an image", type=accept_types, key=f"uploader_{app_mode}")
    else:
        url = st.text_input("Paste direct image URL (http/https)", placeholder="https://example.com/photo.jpg", key=f"url_{app_mode}")
    max_side = st.slider("Resize longest side (px)", 400, 4000, default_max_side, 100)
    return input_source, uploaded, url, max_side

# ------------------------------- PHOTO EDITOR --------------------------------
if app_mode == "Photo Editor":
    with st.sidebar:
        input_source, uploaded, url, max_side = get_input_image_controls(
            ["jpg","jpeg","png","bmp","tif","tiff","webp"], 1600
        )

        # Auto/Reset buttons
        c1, c2 = st.columns(2)
        auto_btn = c1.button("✨ Auto Grade", use_container_width=True)
        reset_btn = c2.button("↩️ Reset", use_container_width=True)

        # Remember user's intent in session_state
        if reset_btn:
            st.session_state["pe_force_original"] = True
        if auto_btn:
            st.session_state["pe_force_original"] = False
            st.session_state["pe_do_auto"] = True

        st.subheader("Basic Adjustments")
        exposure = st.slider("Exposure (EV)", -2.0, 2.0, 0.0, 0.05)
        contrast = st.slider("Contrast", -0.9, 0.9, 0.0, 0.01)
        highlights = st.slider("Highlights", -1.0, 1.0, 0.0, 0.01)
        shadows = st.slider("Shadows", -1.0, 1.0, 0.0, 0.01)
        whites = st.slider("Whites", -1.0, 1.0, 0.0, 0.01)
        blacks = st.slider("Blacks", -1.0, 1.0, 0.0, 0.01)

        st.subheader("Color Adjustments")
        temp = st.slider("Temp", -100, 100, 0, 1)
        tint = st.slider("Tint", -100, 100, 0, 1)
        vibrance = st.slider("Vibrance", -100, 100, 0, 1)
        saturation = st.slider("Saturation", -100, 100, 0, 1)
        with st.expander("HSL / Color Mixer"):
            h_adj = {}
            s_adj = {}
            l_adj = {}
            for name in ["reds","oranges","yellows","greens","aquas","blues","purples","magentas"]:
                st.caption(name.title())
                h_adj[name] = st.slider(f"{name} hue", -45, 45, 0, 1, key=f"h_{name}")
                s_adj[name] = st.slider(f"{name} sat", -100, 100, 0, 1, key=f"s_{name}")
                l_adj[name] = st.slider(f"{name} lum", -50, 50, 0, 1, key=f"l_{name}")

        st.subheader("Detail & Sharpness")
        sharpening = st.slider("Sharpening", 0.0, 3.0, 0.0, 0.05)
        noise_red = st.slider("Noise Reduction", 0.0, 3.0, 0.0, 0.05)
        texture = st.slider("Texture", -1.0, 1.0, 0.0, 0.05)
        clarity = st.slider("Clarity", -1.0, 1.0, 0.0, 0.05)
        dehaze = st.slider("Dehaze", -50.0, 50.0, 0.0, 1.0)

        st.subheader("Crop & Straighten")
        angle = st.slider("Angle/Level", -15.0, 15.0, 0.0, 0.1)
        crop_l = st.slider("Crop Left %", 0.0, 40.0, 0.0, 0.5)
        crop_r = st.slider("Crop Right %", 0.0, 40.0, 0.0, 0.5)
        crop_t = st.slider("Crop Top %", 0.0, 40.0, 0.0, 0.5)
        crop_b = st.slider("Crop Bottom %", 0.0, 40.0, 0.0, 0.5)
        aspect = st.selectbox("Aspect Ratio", ["Original","1:1","4:5","3:2","16:9"])
        aspect_map = {"Original": None, "1:1": 1.0, "4:5": 4/5, "3:2": 3/2, "16:9": 16/9}
        target_aspect = aspect_map[aspect]

        st.subheader("Lens & Geometry")
        vignetting = st.slider("Vignetting correction", -100.0, 100.0, 0.0, 1.0)
        ca_shift = st.slider("Chromatic Aberration shift (px)", -5.0, 5.0, 0.0, 0.1)
        distortion = st.slider("Barrel/Pincushion (k1×1e4)", -50.0, 50.0, 0.0, 1.0)
        persp_x = st.slider("Perspective X %", -20.0, 20.0, 0.0, 0.1)
        persp_y = st.slider("Perspective Y %", -20.0, 20.0, 0.0, 0.1)

        st.subheader("Local Adjustments")
        with st.expander("Radial Filter"):
            cx = st.slider("Center X", 0.0, 1.0, 0.5, 0.01)
            cy = st.slider("Center Y", 0.0, 1.0, 0.5, 0.01)
            rx = st.slider("Radius X", 0.05, 1.0, 0.35, 0.01)
            ry = st.slider("Radius Y", 0.05, 1.0, 0.25, 0.01)
            radial_strength = st.slider(
                "Exposure Strength (Radial)", -100.0, 100.0, 0.0, 1.0, key="pe_radial_strength"
            )
        with st.expander("Graduated Filter"):
            grad_angle = st.slider("Angle", -180.0, 180.0, 0.0, 1.0)
            grad_pos = st.slider("Position (0=top)", 0.0, 1.0, 0.5, 0.01)
            grad_feather = st.slider("Feather %", 1.0, 100.0, 50.0, 1.0)
            grad_strength = st.slider(
                "Exposure Strength (Graduated)", -100.0, 100.0, 0.0, 1.0, key="pe_grad_strength"
            )

    col1, col2 = st.columns(2)
    pbar = st.progress(0, text="Waiting for image…")

    img0, load_err = None, None
    if input_source == "Upload" and uploaded is not None:
        try:
            pbar.progress(5, text="Loading uploaded image…")
            img0 = ensure_max_side(load_image_to_bgr(uploaded), max_side)
        except Exception as e:
            load_err = e
    elif input_source == "Image URL" and url:
        try:
            pbar.progress(5, text="Downloading image from URL…")
            img0 = ensure_max_side(load_image_from_url(url), max_side)
        except Exception as e:
            load_err = e

    if img0 is None:
        if load_err:
            col1.error(f"Failed to load image: {load_err}")
        col1.info("Upload an image or enter a URL to begin")
    else:
        col1.subheader("Original")
        col1.image(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Decide pipeline path: Reset → show original; Auto → auto grade; else sliders pipeline
        if st.session_state.get("pe_force_original", False):
            pbar.progress(100, text="Showing original (Reset)")
            col2.subheader("Edited (Reset)")
            col2.image(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB), use_container_width=True)
        elif st.session_state.get("pe_do_auto", False):
            pbar.progress(25, text="Auto color grading…")
            auto_img = auto_color_grade(img0)
            pbar.progress(100, text="Done (Auto)")
            col2.subheader("Edited (Auto Grade)")
            col2.image(cv2.cvtColor(auto_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            # Auto is one-shot; clear flag so subsequent slider changes apply normally
            st.session_state["pe_do_auto"] = False
        else:
            def step(p, msg):
                pbar.progress(p, text=msg)

            step(15, "Basic adjustments…")
            img = apply_basic(img0, exposure, contrast, highlights, shadows, whites, blacks)

            step(30, "White balance & color…")
            img = apply_white_balance(img, temp, tint)
            img = apply_vibrance_saturation(img, vibrance, saturation)
            img = apply_hsl(img, h_adj, s_adj, l_adj)

            step(55, "Detail & sharpness…")
            img = apply_detail(img, sharpening, noise_red, texture, clarity, dehaze)

            step(70, "Lens & geometry…")
            img = apply_lens_geometry(img, vignetting, ca_shift, distortion, persp_x, persp_y)

            step(90, "Crop & straighten…")
            img = apply_crop_rotate(img, angle, crop_l, crop_r, crop_t, crop_b, target_aspect)

            step(100, "Done")
            col2.subheader("Edited")
            col2.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

# ------------------------------ RASTER→VECTOR --------------------------------
else:
    with st.sidebar:
        input_source, uploaded, url, max_side = get_input_image_controls(
            ["jpg","jpeg","png","bmp","tif","tiff","webp"], 1400
        )

        st.subheader("Vectorization")
        mode = st.radio("Mode", ["centerline","outline"], index=0, help="Centerline=skeleton; Outline=contours")
        thr_method = st.radio("Threshold", ["adaptive","otsu"], index=0)
        invert = st.checkbox("Invert", value=False)
        blur_ksize = st.slider("Blur ksize", 0, 15, 3, 1)
        block_size = st.slider("Adaptive block size (odd)", 3, 101, 35, 2)
        C = st.slider("Adaptive C", -30, 30, 10, 1)

        st.subheader("Geometry / Export (mm)")
        width_mm = st.slider("Width", 100, 2000, 800, 10)
        height_mm_opt = st.slider("Height (0=auto)", 0, 2000, 0, 10)
        margin_mm = st.slider("Margin", 0, 100, 10, 1)
        stroke_mm = st.slider("Stroke preview", 0.0, 2.0, 0.35, 0.05)

        st.subheader("Simplify / Filter")
        simplify_eps_mm = st.slider("Simplify epsilon", 0.0, 5.0, 0.5, 0.05, help="Higher = fewer nodes")
        min_len_mm = st.slider("Min path length", 0.0, 20.0, 2.0, 0.5)
        optimize = st.checkbox("Optimize pen-up travel", value=True)

    col1, col2, col3 = st.columns(3)
    pbar = st.progress(0, text="Waiting for image…")

    img_bgr, load_err = None, None
    if input_source == "Upload" and uploaded is not None:
        try:
            pbar.progress(5, text="Loading uploaded image…")
            img_bgr = ensure_max_side(load_image_to_bgr(uploaded), max_side)
        except Exception as e:
            load_err = e
    elif input_source == "Image URL" and url:
        try:
            pbar.progress(5, text="Downloading image from URL…")
            img_bgr = ensure_max_side(load_image_from_url(url), max_side)
        except Exception as e:
            load_err = e

    if img_bgr is None:
        if load_err:
            col1.error(f"Failed to load image: {load_err}")
        col1.info("Upload an image or enter a URL to begin")
    else:
        def step(pct, msg):
            pbar.progress(pct, text=msg)

        gray = to_grayscale(img_bgr)

        col1.subheader("Original")
        col1.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        col2.subheader("Grayscale")
        col2.image(gray, clamp=True, use_container_width=True)

        try:
            # Vectorize (function defined above Photo Editor in original code)
            def vectorize_with_progress(gray: np.ndarray, progress_cb, *,
                                        mode: str, thr_method: str, invert: bool, blur_ksize: int,
                                        block_size: int, C: int,
                                        width_mm: float, height_mm_opt: float, margin_mm: float,
                                        simplify_eps_mm: float, min_len_mm: float,
                                        stroke_mm: float, optimize: bool):
                progress_cb(10, "Thresholding…")
                binary = threshold_bw(gray, method=thr_method, block_size=block_size, C=C,
                                      invert=invert, blur_ksize=blur_ksize, use_otsu=(thr_method=="otsu"))
                progress_cb(25, "Preparing geometry…")
                h, w = gray.shape
                avail_w = max(1e-6, width_mm - 2*margin_mm)
                px_per_mm = w / avail_w
                simplify_eps_px = max(0.0, simplify_eps_mm * px_per_mm)
                min_len_px = max(0.0, min_len_mm * px_per_mm)
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
                progress_cb(80, "Rendering preview…")
                preview = np.zeros((h, w, 3), dtype=np.uint8)
                px_thick = max(1, int(round(stroke_mm * 2)))
                for poly in polylines:
                    if len(poly) >= 2:
                        pts = np.array(poly, dtype=np.int32)
                        cv2.polylines(preview, [pts], isClosed=False, color=(255,255,255), thickness=px_thick)
                progress_cb(95, "Building SVG…")
                height_mm = None if height_mm_opt <= 0 else height_mm_opt
                svg = svg_bytes(polylines, w, h, width_mm, height_mm, margin_mm, stroke_mm, optimize)
                progress_cb(100, "Done")
                return binary, polylines, preview, svg

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
            st.download_button("⬇️ Download SVG", data=svg, file_name="vectorized.svg", mime="image/svg+xml")
            st.caption(f"Paths: {len(polylines)}  ·  Image: {gray.shape[1]}×{gray.shape[0]} px")
        except Exception as e:
            st.error(str(e))
        finally:
            pbar.progress(100, text="Ready")

# Bottom AdSense slot
st.divider()
render_adsense("slot_bottom", height=120)
