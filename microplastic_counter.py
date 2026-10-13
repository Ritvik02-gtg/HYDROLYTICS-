import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

SIZE_BINS_UM = [0, 20, 50, 100, 300, 1000, 5000]  
COLORS = {"fiber": (0, 165, 255), "bead": (255, 200, 0),
          "fragment": (0, 255, 0), "film": (255, 0, 255)}
CHANNELS = {"b": 0, "g": 1, "r": 2}
MAX_SIDE = 2400                                    


def to_gray(img, channel):
    if channel == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(img[:, :, CHANNELS[channel]])   


def detect_mode(gray):
    """Bright background -> dark particles on a light filter; dark background -> fluorescence."""
    return "dark" if float(np.median(gray)) >= 100 else "fluorescent"


def noise_floor(gray, min_contrast):
    """Raise the contrast threshold on grainy images so texture is not counted as particles."""
    resid = gray.astype(np.float32) - cv2.medianBlur(gray, 5).astype(np.float32)
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    return max(float(min_contrast), 6.0 * sigma)


def flatten_background(gray, mode, sigma=None):
    """Remove uneven illumination so one threshold works across the field.
    Kernel scales with image size (fixed 51 px was too small for large images and
    too large for thumbnails). Median is robust to big particles biasing the estimate."""
    if sigma is None:
        sigma = max(15, min(gray.shape) // 15)
    f = 4 if sigma > 60 else 1                     
    small = cv2.resize(gray, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA) if f > 1 else gray
    bg = cv2.medianBlur(small, min(sigma // f, 255) | 1)
    if f > 1:
        bg = cv2.resize(bg, gray.shape[::-1], interpolation=cv2.INTER_LINEAR)
    bg = bg.astype(np.float32)
    g = gray.astype(np.float32)
    diff = g - bg if mode == "fluorescent" else bg - g
    return np.clip(diff, 0, 255).astype(np.uint8)


def dish_mask(gray, shrink=0.93):
    """Find a petri dish / round filter and return a mask of its interior (or None).
    Prevents the dish rim and its shadow being counted as a giant 'fiber'."""
    h, w = gray.shape
    r_min, r_max = int(0.3 * min(h, w)), int(0.55 * max(h, w))
    circles = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=2,
                               minDist=min(h, w), param1=100, param2=60,
                               minRadius=r_min, maxRadius=r_max)
    if circles is None:
        return None
    x, y, r = circles[0, 0]
    m = np.zeros_like(gray)
    cv2.circle(m, (int(x), int(y)), int(r * shrink), 255, -1)
    return m


def segment(signal, split_touching=True, floor=25, roi=None):
    """floor = minimum contrast (0-255 grey levels) a particle must have over the
    background. Stops the automatic threshold from latching onto noise."""
    blur = cv2.GaussianBlur(signal, (3, 3), 0)
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(blur, max(thr, floor), 255, cv2.THRESH_BINARY)
    if roi is not None:
        mask &= roi
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    if not split_touching:
        return cv2.connectedComponents(mask, connectivity=8)[1]

    # watershed on distance transform to separate touching compact particles
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    local = ndi.maximum_filter(dist, size=9) == dist
    markers, _ = ndi.label(local & (dist > 2.5))
    markers = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers.astype(np.int32) + 1)
    labels = np.where((markers > 1) & (mask > 0), markers - 1, 0).astype(np.int32)
    # particles with no marker (e.g. thin fibers) -> keep as their own components
    missed = ((mask > 0) & (labels == 0)).astype(np.uint8)
    _, extra = cv2.connectedComponents(missed, connectivity=8)
    labels[extra > 0] = extra[extra > 0] + labels.max()
    return labels



def classify(aspect, circularity, solidity, thinness):
   
    if aspect >= 3.5 or (thinness > 60 and solidity < 0.5):
        return "fiber"
    if circularity >= 0.87 and aspect < 1.2:
        return "bead"
    if solidity < 0.70:
        return "film"
    return "fragment"


def plastic_likelihood(shape, sat, val, contrast, mode):
    """Rough visual triage, NOT identification. Saturated unnatural colours (blue, red,
    green, purple) are a common hint for synthetic fibres; brown/tan is usually organic.
    Everything still needs hot-needle / FTIR / Raman confirmation."""
    if mode == "fluorescent":                
        return "likely" if contrast >= 60 else "possible"
    if sat >= 90 and val >= 60:
        return "likely"
    if shape in ("bead", "fiber", "film"):
        return "possible"
    return "uncertain"


def measure(labels, um_per_px, min_um, max_um, signal, min_contrast, img_bgr, mode, min_px=3):
    """Per-particle measurements. Works on each object's bounding-box crop
    (ndi.find_objects) instead of building a full-image mask per label, which was
    O(n_particles x image_size) and very slow on busy images."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    rows = []
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        m = (labels[sl] == lab)
        area_px = int(m.sum())
        if area_px < min_px:
            continue
        contrast = float(signal[sl][m].mean())
        if contrast < min_contrast:
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        c = c + np.array([[sl[1].start, sl[0].start]])         
        area = max(cv2.contourArea(c), float(area_px))
        per = cv2.arcLength(c, True)
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        long_px, short_px = max(w, h), max(min(w, h), 1.0)
        thinness = per ** 2 / area if area else 0
    
        if thinness > 60:
            long_px = max(long_px, per / 2)
            short_px = max(area / long_px, 1.0)
        feret_um = long_px * um_per_px
        if not (min_um <= feret_um <= max_um):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c)) or area
        circ = min(4 * np.pi * area / per ** 2, 1.0) if per > 0 else 0
        aspect = long_px / short_px
        shape = classify(aspect, circ, area / hull_area, thinness)
        hue, sat, val = (float(v) for v in hsv[sl][m].mean(axis=0))
        b, g, r = (int(v) for v in img_bgr[sl][m].mean(axis=0))
        rows.append(dict(
            id=lab, x_px=round(cx, 1), y_px=round(cy, 1),
            length_um=round(feret_um, 2), width_um=round(short_px * um_per_px, 2),
            area_um2=round(area * um_per_px ** 2, 2), contrast=round(contrast, 1),
            aspect_ratio=round(aspect, 2), circularity=round(circ, 3),
            mean_rgb=f"{r},{g},{b}", saturation=round(sat, 1),
            shape=shape, mp_likelihood=plastic_likelihood(shape, sat, val, contrast, mode),
            _contour=c))
    return rows


def assess_image(gray, labels, mode, n_particles, roi=None):
    """Return (warnings, looks_valid). Heuristic check that the image resembles a
    sparse particle sample rather than an ordinary photo."""
    warnings, bad = [], False
    med = float(np.median(gray))
    area = float((roi > 0).sum()) if roi is not None else gray.size
    fg = float((labels > 0).sum()) / max(area, 1)
    bg_std = float(cv2.GaussianBlur(gray, (101, 101), 0).std())
    if min(gray.shape) < 500:
        warnings.append(f"Low resolution ({gray.shape[1]}x{gray.shape[0]} px). Small particles "
                        "cannot be resolved; use the original camera file, not a messaging-app copy.")
    if mode == "fluorescent" and med > 110:
        warnings.append(f"Background is bright (median {med:.0f}/255) but mode is 'fluorescent'. Try 'auto' or 'dark'.")
    if mode == "dark" and med < 100:
        warnings.append(f"Background is dark (median {med:.0f}/255) but mode is 'dark'. Try 'auto' or 'fluorescent'.")
    if fg > 0.20:
        warnings.append(f"{fg:.0%} of the field was flagged as particles. Real samples are sparse; "
                        "this looks like an ordinary photo, bubbles, or a crowded/contaminated field.")
        bad = True
    if bg_std > 35 and roi is None:
        warnings.append("Large uneven areas of light and dark (scene-like content, not an even filter). "
                        "If this is a petri dish, enable dish masking.")
        bad = True
    if n_particles > 1500:
        warnings.append(f"{n_particles} detections is unusually many; likely debris, texture, bubbles or noise.")
    if n_particles == 0:
        warnings.append("No particles met the contrast threshold.")
    return warnings, not bad


@dataclass
class Result:
    annotated: np.ndarray
    df: pd.DataFrame
    area_mm2: float
    mode: str
    warnings: list = field(default_factory=list)
    valid: bool = True


def analyze(img_bgr, um_per_px=1.0, mode="auto", channel="g", min_um=0.0, max_um=5000.0,
            split_touching=True, min_contrast=25, dish=False):
    """Single pipeline shared by the CLI and the web app (previously duplicated)."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img_bgr = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        um_per_px /= s
    gray = to_gray(img_bgr, channel)
    if mode == "auto":
        mode = detect_mode(gray)
    roi = dish_mask(gray) if dish else None
    signal = flatten_background(gray, mode)
    floor = noise_floor(gray, min_contrast)
    labels = segment(signal, split_touching, floor, roi)
    rows = measure(labels, um_per_px, min_um, max_um, signal, floor, img_bgr, mode)
    warnings, valid = assess_image(gray, labels, mode, len(rows), roi)
    if dish and roi is None:
        warnings.append("Dish masking requested but no circular dish was found.")

    out = img_bgr.copy()
    if roi is not None:
        cs, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (128, 128, 128), 1)
    for r in rows:
        cv2.drawContours(out, [r["_contour"]], -1, COLORS[r["shape"]], 1 if max(h, w) < 1500 else 2)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_contour"} for r in rows])
    px_area = float((roi > 0).sum()) if roi is not None else gray.size
    return Result(out, df, px_area * um_per_px ** 2 / 1e6, mode, warnings, valid)



def draw_legend(img):
    for i, (name, col) in enumerate(COLORS.items()):
        cv2.putText(img, name, (10, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img


def summarize(df, area_mm2=None, volume_L=None):
    lines = [f"Total particles: {len(df)}"]
    if df.empty:
        return "\n".join(lines + ["No particles detected."])
    lines += ["", "By shape:", df["shape"].value_counts().to_string(),
              "", "Visual microplastic triage (needs FTIR/Raman/hot-needle to confirm):",
              df["mp_likelihood"].value_counts().to_string(),
              "", "By size class (longest dimension, um):",
              pd.cut(df.length_um, SIZE_BINS_UM, right=False).value_counts().sort_index().to_string(),
              "", f"Median length: {df.length_um.median():.1f} um   "
                  f"Mean: {df.length_um.mean():.1f} um   Max: {df.length_um.max():.1f} um"]
    if area_mm2:
        lines.append(f"Surface density: {len(df) / area_mm2:.2f} particles/mm2")
    if volume_L:
        lines.append(f"Concentration: {len(df) / volume_L:.1f} particles/L (filtered volume {volume_L} L)")
    return "\n".join(lines)


def process(path, volume_L=None, **kw):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    res = analyze(img, **kw)
    print(f"Image type used: {res.mode}")
    for w in res.warnings:
        print("WARNING:", w)
    if not res.valid:
        print("WARNING: this image does not look like a particle sample; treat the numbers below as unreliable.\n")
    stem = Path(path).with_suffix("")
    cv2.imwrite(f"{stem}_annotated.png", draw_legend(res.annotated))
    res.df.to_csv(f"{stem}_particles.csv", index=False)
    text = summarize(res.df, res.area_mm2, volume_L)
    Path(f"{stem}_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return res.df



def make_demo(path, seed=1):
    """Synthetic fluorescent field: fragments, fibers, beads, uneven lighting, noise."""
    rng = np.random.default_rng(seed)
    H, W = 900, 1200
    img = np.zeros((H, W), np.uint8)
    truth = {"fragment": 60, "fiber": 18, "bead": 25}
    for _ in range(truth["fragment"]):
        c = (int(rng.integers(30, W - 30)), int(rng.integers(30, H - 30)))
        ax = (int(rng.integers(5, 16)), int(rng.integers(4, 11)))
        cv2.ellipse(img, c, ax, float(rng.uniform(0, 180)), 0, 360, int(rng.integers(150, 255)), -1)
    for _ in range(truth["fiber"]):
        p = rng.integers(40, [W - 40, H - 40])
        ang, L = rng.uniform(0, 2 * np.pi), rng.integers(50, 160)
        q = (p + L * np.array([np.cos(ang), np.sin(ang)])).astype(int)
        cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), 220, 2)
    for _ in range(truth["bead"]):
        c = (int(rng.integers(20, W - 20)), int(rng.integers(20, H - 20)))
        cv2.circle(img, c, int(rng.integers(4, 9)), 255, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32)
    xx, yy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    img += 25 + 30 * (xx * 0.7 + yy * 0.3)                  
    img += rng.normal(0, 6, img.shape)                     
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.merge([img // 4, img, img // 3]))   
    return truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?")
    ap.add_argument("--um-per-px", type=float, default=1.0, help="calibration (um per pixel)")
    ap.add_argument("--mode", choices=["auto", "fluorescent", "dark"], default="auto",
                    help="fluorescent = bright on dark; dark = dark particles on light filter")
    ap.add_argument("--channel", choices=["gray", "r", "g", "b"], default="g")
    ap.add_argument("--min-um", type=float, default=0.0)
    ap.add_argument("--max-um", type=float, default=5000.0)
    ap.add_argument("--volume-L", type=float, default=None, help="water volume filtered, for particles/L")
    ap.add_argument("--min-contrast", type=float, default=25,
                    help="minimum particle contrast over background (0-255 grey levels)")
    ap.add_argument("--no-split", action="store_true", help="disable watershed splitting")
    ap.add_argument("--dish", action="store_true", help="only analyse inside a detected petri dish/round filter")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo:
        a.image = "demo_sample.png"
        truth = make_demo(a.image)
        print("Ground truth in synthetic image:", truth, "(total", sum(truth.values()), ")\n")
        a.um_per_px = 2.0
    if not a.image:
        ap.error("provide an image path or --demo")
    process(a.image, a.volume_L, um_per_px=a.um_per_px, mode=a.mode, channel=a.channel,
            min_um=a.min_um, max_um=a.max_um, split_touching=not a.no_split,
            min_contrast=a.min_contrast, dish=a.dish)
