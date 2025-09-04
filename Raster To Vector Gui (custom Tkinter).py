#!/usr/bin/env python3
"""
Raster → Vector GUI for pen plotters / wall painting robots.

Now with a responsive **progress bar** for long operations and background
threading so the UI stays fluid while vectorization/export runs.

Features
- Browse an image, adjust parameters (sliders/dropdowns), live BEFORE/AFTER preview
- Choose vectorization mode: centerline (skeleton) or outline (contours)
- Save clean SVG polylines (mm units) for downstream G‑code
- Progress bar + status text during heavy work (e.g., vectorize/save)

Install deps:
  pip install opencv-python scikit-image svgwrite numpy pillow

Run:
  python raster_to_vector_gui.py
"""

import math
import os
from typing import List, Sequence, Tuple, Dict, Set
import threading

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import svgwrite

try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None

# ------------------------------- Types ---------------------------------------
Point = Tuple[int, int]
FloatPoint = Tuple[float, float]
Polyline = List[FloatPoint]

# ------------------------- Image & Thresholding ------------------------------

def threshold_bw(gray: np.ndarray,
                 method: str = "adaptive",
                 block_size: int = 35,
                 C: int = 10,
                 invert: bool = False,
                 blur_ksize: int = 3,
                 use_otsu: bool = False) -> np.ndarray:
    """Return binary (uint8 0/255). Foreground=white."""
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

# ------------------------------ Outline mode ---------------------------------

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

# ------------------------------- SVG writing ---------------------------------

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
        raise RuntimeError("No vector paths generated. Adjust parameters and try again.")
    if optimize_order:
        polylines = nearest_order(polylines)
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
    dwg = svgwrite.Drawing(filename=svg_path,
                           size=(f"{width_mm}mm", f"{height_mm}mm"),
                           viewBox=f"0 0 {width_mm} {height_mm}")
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

# ------------------------------- GUI -----------------------------------------

class RasterToVectorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Raster → Vector Plotter GUI")
        self.master.minsize(1000, 650)

        # State
        self.image_path: str | None = None
        self.gray: np.ndarray | None = None
        self.binary: np.ndarray | None = None
        self.preview_photo_before = None
        self.preview_photo_after = None
        self.polylines: List[Polyline] = []

        # Top bar
        frame_top = ttk.Frame(master)
        frame_top.pack(fill="x")
        ttk.Button(frame_top, text="Browse Image…", command=self.load_image).pack(side="left", padx=6, pady=6)
        ttk.Button(frame_top, text="Save SVG…", command=self.on_save_clicked).pack(side="left", padx=6, pady=6)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frame_top, textvariable=self.status_var).pack(side="right", padx=6)

        # Progress bar
        self.progress = ttk.Progressbar(master, mode="indeterminate")
        self.progress.pack(fill="x")
        self.progress.stop()

        # Controls
        controls = ttk.LabelFrame(master, text="Parameters")
        controls.pack(side="left", fill="y", padx=8, pady=8)

        self.params = {
            "mode": tk.StringVar(value="centerline"),
            "threshold": tk.StringVar(value="adaptive"),
            "invert": tk.BooleanVar(value=False),
            "blur_ksize": tk.IntVar(value=3),
            "block_size": tk.IntVar(value=35),
            "C": tk.IntVar(value=10),
            # export geometry (mm)
            "width_mm": tk.DoubleVar(value=800.0),
            "height_mm": tk.DoubleVar(value=0.0),  # 0 = auto keep aspect
            "margin_mm": tk.DoubleVar(value=10.0),
            "stroke_mm": tk.DoubleVar(value=0.35),
            # simplify / filter (approximate in px based on width)
            "simplify_epsilon_mm": tk.DoubleVar(value=0.5),
            "min_length_mm": tk.DoubleVar(value=2.0),
            "optimize_order": tk.BooleanVar(value=True),
        }

        # Mode & threshold options
        self._dropdown(controls, "Mode", self.params["mode"], ["centerline", "outline"]) 
        self._dropdown(controls, "Threshold", self.params["threshold"], ["adaptive", "otsu"]) 
        ttk.Checkbutton(controls, text="Invert", variable=self.params["invert"], command=self.update_preview).pack(anchor="w", padx=6)

        # Sliders
        self._slider(controls, "Blur ksize", self.params["blur_ksize"], 0, 15, 1)
        self._slider(controls, "Adaptive block size", self.params["block_size"], 3, 101, 2)
        self._slider(controls, "Adaptive C", self.params["C"], -30, 30, 1)

        # Geometry
        geom = ttk.LabelFrame(controls, text="Export Geometry (mm)")
        geom.pack(fill="x", padx=6, pady=6)
        self._slider(geom, "Width (mm)", self.params["width_mm"], 100, 2000, 10)
        self._slider(geom, "Height (mm, 0=auto)", self.params["height_mm"], 0, 2000, 10)
        self._slider(geom, "Margin (mm)", self.params["margin_mm"], 0, 100, 1)
        self._slider(geom, "Stroke preview (mm)", self.params["stroke_mm"], 0, 2, 0.05)

        # Vectorization
        vec = ttk.LabelFrame(controls, text="Vectorization")
        vec.pack(fill="x", padx=6, pady=6)
        self._slider(vec, "Simplify epsilon (mm)", self.params["simplify_epsilon_mm"], 0, 5, 0.05)
        self._slider(vec, "Min path length (mm)", self.params["min_length_mm"], 0, 20, 0.5)
        ttk.Checkbutton(vec, text="Optimize pen-up travel", variable=self.params["optimize_order"], command=self.update_preview).pack(anchor="w", padx=6)

        # Preview area
        frame_preview = ttk.Frame(master)
        frame_preview.pack(side="right", expand=True, fill="both", padx=8, pady=8)
        self.lbl_before = ttk.Label(frame_preview, text="Before (grayscale)")
        self.lbl_before.pack(anchor="w")
        self.canvas_before = ttk.Label(frame_preview)
        self.canvas_before.pack(expand=True, fill="both")
        self.lbl_after = ttk.Label(frame_preview, text="After (thresholded)")
        self.lbl_after.pack(anchor="w", pady=(10,0))
        self.canvas_after = ttk.Label(frame_preview)
        self.canvas_after.pack(expand=True, fill="both")

    # ---------------------------- UI helpers ---------------------------------
    def _dropdown(self, parent, label, var, options):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Label(row, text=label, width=18).pack(side="left")
        om = ttk.OptionMenu(row, var, var.get(), *options, command=lambda *a: self.update_preview())
        om.pack(side="left", fill="x", expand=True)

    def _slider(self, parent, label, var, mn, mx, step):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=6, pady=4)
        txt = ttk.Label(row, text=f"{label}: {var.get()}")
        txt.pack(anchor="w")
        def on_move(v):
            try:
                val = float(v)
            except Exception:
                val = var.get()
            var.set(val)
            if step < 1:
                txt.configure(text=f"{label}: {val:.2f}")
            else:
                txt.configure(text=f"{label}: {int(val)}")
            self.update_preview()
        s = ttk.Scale(row, from_=mn, to=mx, orient="horizontal", command=on_move)
        s.set(var.get())
        s.pack(fill="x")

    # ------------------------------ I/O --------------------------------------
    def load_image(self):
        path = filedialog.askopenfilename(title="Choose an image",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", f"Failed to read image: {path}")
            return
        self.image_path = path
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.gray = gray
        self.show_image(gray, self.canvas_before)
        self.update_preview()

    def show_image(self, img: np.ndarray, label: ttk.Label, max_side: int = 800):
        if img.ndim == 2:
            disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            disp = img
        h, w = disp.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            disp = cv2.resize(disp, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(disp)
        ph = ImageTk.PhotoImage(pil)
        label.configure(image=ph)
        label.image = ph

    # ----------------------------- Preview -----------------------------------
    def update_preview(self):
        if self.gray is None:
            return
        try:
            method = self.params["threshold"].get()
            invert = self.params["invert"].get()
            blur_ksize = int(round(self.params["blur_ksize"].get()))
            block_size = int(round(self.params["block_size"].get()))
            if block_size % 2 == 0:
                block_size += 1
            C = int(round(self.params["C"].get()))
            binary = threshold_bw(self.gray, method=method, block_size=block_size,
                                  C=C, invert=invert, blur_ksize=blur_ksize,
                                  use_otsu=(method=="otsu"))
            self.binary = binary
            self.show_image(binary, self.canvas_after)
            self.status_var.set("Preview updated")
        except Exception as e:
            self.status_var.set(str(e))

    # ---------------------- Progress / background work -----------------------
    def start_progress(self, msg: str):
        self.status_var.set(msg)
        self.progress.start(8)  # ms per move
        self.master.configure(cursor="watch")
        for child in self.master.winfo_children():
            try:
                child.configure(state="disabled")
            except Exception:
                pass
        self.progress.configure(mode="indeterminate")

    def stop_progress(self, msg: str = "Done"):
        self.progress.stop()
        self.status_var.set(msg)
        self.master.configure(cursor="")
        for child in self.master.winfo_children():
            try:
                child.configure(state="normal")
            except Exception:
                pass

    # ------------------------------ Save SVG ---------------------------------
    def on_save_clicked(self):
        if self.gray is None or self.binary is None:
            messagebox.showinfo("Open an image", "Please open an image first.")
            return
        out_path = filedialog.asksaveasfilename(defaultextension=".svg",
                                                filetypes=[("SVG", "*.svg")],
                                                initialfile=(os.path.splitext(os.path.basename(self.image_path))[0] + ".svg") if self.image_path else "output.svg")
        if not out_path:
            return

        # Run vectorization + save in a background thread
        t = threading.Thread(target=self._vectorize_and_save, args=(out_path,), daemon=True)
        self.start_progress("Vectorizing & saving…")
        t.start()

    def _vectorize_and_save(self, out_path: str):
        try:
            # Gather params (read once to avoid cross-thread races)
            mode = self.params["mode"].get()
            width_mm = float(self.params["width_mm"].get())
            height_mm = float(self.params["height_mm"].get())
            height_mm = None if height_mm <= 0 else height_mm
            margin_mm = float(self.params["margin_mm"].get())
            stroke_mm = float(self.params["stroke_mm"].get())
            simplify_eps_mm = float(self.params["simplify_epsilon_mm"].get())
            min_len_mm = float(self.params["min_length_mm"].get())
            optimize = bool(self.params["optimize_order"].get())

            h, w = self.binary.shape
            avail_w = max(1e-6, width_mm - 2*margin_mm)
            px_per_mm = w / avail_w
            simplify_eps_px = max(0.0, simplify_eps_mm * px_per_mm)
            min_len_px = max(0.0, min_len_mm * px_per_mm)

            # Vectorize (potentially heavy)
            if mode == "outline":
                polylines = find_outline_polylines(self.binary, simplify_eps_px, min_len_px)
            else:
                if skeletonize is None:
                    raise RuntimeError("Install scikit-image for centerline mode: pip install scikit-image")
                skel = skeletonize(self.binary > 0).astype(np.uint8) * 255
                polylines = trace_skeleton_to_polylines(skel, simplify_eps_px, min_len_px)

            # Save SVG
            save_svg(polylines, out_path, w, h, width_mm, height_mm, margin_mm, stroke_mm, optimize)

            # Notify success back on UI thread
            self.master.after(0, lambda: (self.stop_progress(f"Saved → {out_path}"),
                                          messagebox.showinfo("Saved", f"SVG written to:\n{out_path}")))
        except Exception as e:
            self.master.after(0, lambda: (self.stop_progress("Error"),
                                          messagebox.showerror("Error", str(e))))


if __name__ == "__main__":
    root = tk.Tk()
    app = RasterToVectorGUI(root)
    root.mainloop()
