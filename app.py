"""Streamlit web app for the microplastic counter (single file: UI + analysis).
Run locally:  streamlit run app.py
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ======================= ANALYSIS ENGINE =======================
# (the contents of microplastic_counter.py, merged in so this app is a single file)
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

SIZE_BINS_UM = [0, 20, 50, 100, 300, 1000, 5000]   # common MP size classes (um)
COLORS = {"fiber": (0, 165, 255), "bead": (255, 200, 0),
          "fragment": (0, 255, 0), "film": (255, 0, 255)}
CHANNELS = {"b": 0, "g": 1, "r": 2}
MAX_SIDE = 2400                                     # downscale huge images (keeps calibration)


# ---------------------------- preprocessing ----------------------------
def to_gray(img, channel):
    if channel == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(img[:, :, CHANNELS[channel]])   # e.g. green for Nile Red


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
    f = 4 if sigma > 60 else 1                     # median on a downscaled copy: much faster
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


# ---------------------------- measurement ------------------------------
def classify(aspect, circularity, solidity, thinness):
    # Thresholds are heuristics; tune against particles you have identified by hand.
    # thinness = perimeter^2 / area: catches curly fibers whose bounding box is square.
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
    if mode == "fluorescent":                 # Nile Red stains plastics; bright = candidate
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
        c = c + np.array([[sl[1].start, sl[0].start]])          # back to full-image coords
        area = max(cv2.contourArea(c), float(area_px))
        per = cv2.arcLength(c, True)
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        long_px, short_px = max(w, h), max(min(w, h), 1.0)
        thinness = per ** 2 / area if area else 0
        # curly fibres: minAreaRect underestimates length; use ~half the perimeter instead
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


# ---------------------------- sanity checks ----------------------------
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


# ---------------------------- core pipeline ----------------------------
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


# ---------------------------- reporting --------------------------------
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


# ---------------------------- demo image -------------------------------
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
    img += 25 + 30 * (xx * 0.7 + yy * 0.3)                  # uneven illumination
    img += rng.normal(0, 6, img.shape)                      # sensor noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.merge([img // 4, img, img // 3]))   # greenish fluorescence
    return truth


# ---------------------------- CLI --------------------------------------
# ===============================================================

# ===================== COLOURS & FONT: edit these =====================
BEIGE      = "#F4EEDF"   # page background (warm cream)
BEIGE_DARK = "#E8DDC6"   # sidebar and upload box (sand)
BEIGE_LITE = "#FAF6EB"   # input boxes
OLIVE      = "#6F7B45"   # buttons, slider handle, charts (soft olive)
OLIVE_DARK = "#56603A"   # button hover, small accents (deep olive)
TEXT       = "#3A3D24"   # all text
FONT       = '"Times New Roman", Times, "Liberation Serif", serif'
# ======================================================================

THEME_CSS = """
<style>
.stApp, [data-testid="stHeader"] { background-color: %BEIGE%; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background-color: %BEIGE_DARK%; }

html, body, .stApp,
.stApp p, .stApp li, .stApp label, .stApp a,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp button, .stApp input, .stApp textarea, .stApp select,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricValue"],
.stApp [data-testid="stAlertContainer"],
.stApp [data-testid="stFileUploaderDropzone"],
.stApp [data-testid="stSliderThumbValue"],
.stApp [data-baseweb="select"] div, .stApp [data-baseweb="radio"] div {
    font-family: %FONT% !important;
}

.stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label,
.stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricValue"],
.stApp [data-testid="stCaptionContainer"] { color: %TEXT%; }

.stApp [data-testid="stFileUploaderDropzone"] { background-color: %BEIGE_DARK%; }
.stApp [data-baseweb="input"] > div, .stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="base-input"] { background-color: %BEIGE_LITE%; }

.stApp [data-testid^="stBaseButton"] {
    background-color: %OLIVE%; border: 1px solid %OLIVE_DARK%; color: #FFFFFF;
}
.stApp [data-testid^="stBaseButton"] p { color: #FFFFFF; }
.stApp [data-testid^="stBaseButton"]:hover {
    background-color: %OLIVE_DARK%; border-color: %OLIVE_DARK%; color: #FFFFFF;
}

/* Force dark text on the beige surfaces, even when the browser/app is in dark mode */
.stApp [data-testid="stMarkdownContainer"] *, .stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stRadio"] *, .stApp [data-testid="stCheckbox"] *,
.stApp [data-testid="stSelectbox"] *, .stApp [data-testid="stNumberInput"] *,
.stApp [data-testid="stSlider"] *, .stApp [data-testid="stFileUploader"] *,
.stApp [data-testid="stAlertContainer"] *, .stApp [data-testid="stMetric"] *,
.stApp [data-testid="stSidebar"] *, .stApp [data-testid="stCaptionContainer"] *,
.stApp [data-baseweb="input"] input, .stApp [data-baseweb="select"] * { color: %TEXT%; }

/* Dropdown lists and tooltips render outside the page, so style them separately */
[data-baseweb="popover"] > div, [data-baseweb="menu"], [data-baseweb="popover"] ul {
    background-color: %BEIGE_LITE% !important;
}
[data-baseweb="popover"] *, [data-baseweb="menu"] * { color: %TEXT% !important; }

.stApp [data-testid="stSlider"] [role="slider"] { background-color: %OLIVE% !important; }
.stApp [data-testid="stSliderThumbValue"], .stApp [data-testid="stSliderThumbValue"] * { color: %OLIVE_DARK%; }

/* ---- Remove the dark input boxes: number inputs, selectbox, radio dots ---- */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
.stApp [data-baseweb="input"] > div, .stApp [data-baseweb="select"] > div,
.stApp [data-testid="stNumberInput"] > div, .stApp [data-testid="stNumberInputContainer"],
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: %BEIGE_LITE% !important;
    border-color: %BEIGE_DARK% !important;
}
.stApp [data-baseweb="input"] input, .stApp [data-baseweb="base-input"] input,
.stApp [data-baseweb="select"] input {
    background-color: transparent !important;
    color: %TEXT% !important; -webkit-text-fill-color: %TEXT% !important;
}
.stApp [data-testid="stNumberInput"] button,
.stApp [data-testid="stNumberInputStepUp"], .stApp [data-testid="stNumberInputStepDown"] {
    background-color: %BEIGE_LITE% !important; color: %OLIVE_DARK% !important;
}
.stApp [data-testid="stNumberInput"] button * { color: %OLIVE_DARK% !important; fill: %OLIVE_DARK% !important; }
.stApp [data-testid="stSelectbox"] [data-baseweb="select"],
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] div,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
    background-color: %BEIGE_LITE% !important; color: %TEXT% !important;
}
.stApp [data-baseweb="select"] svg { fill: %OLIVE_DARK% !important; }
.stApp [data-baseweb="radio"] > div:first-child { background-color: %BEIGE_LITE% !important; border-color: %OLIVE% !important; }
.stApp [data-baseweb="radio"][aria-checked="true"] > div:first-child { background-color: %OLIVE% !important; border-color: %OLIVE% !important; }
.stApp [data-testid="stTooltipIcon"] svg { fill: %OLIVE_DARK% !important; }

/* Buttons keep white text (listed last so they win over the rules above) */
.stApp [data-testid^="stBaseButton"], .stApp [data-testid^="stBaseButton"] *,
.stApp [data-testid^="stBaseButton"]:hover * { color: #FFFFFF; }

/* ---------------- Sidebar: hover menu ---------------- */
/* The collapse arrow is a quiet icon, not an olive button */
.stApp [data-testid="stSidebarHeader"] [data-testid^="stBaseButton"],
.stApp [data-testid="stSidebarCollapseButton"] [data-testid^="stBaseButton"] {
    background: transparent; border: none; color: %OLIVE_DARK%;
}
.stApp [data-testid="stSidebarHeader"] [data-testid^="stBaseButton"] *,
.stApp [data-testid="stSidebarCollapseButton"] [data-testid^="stBaseButton"] * {
    color: %OLIVE_DARK%; fill: %OLIVE_DARK%;
}

/* Each section is a plain row separated by a hairline */
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] {
    position: relative; overflow: visible !important;
    background: transparent !important; border: none !important; border-radius: 0 !important;
    border-bottom: 1px solid rgba(58, 61, 36, 0.22) !important; box-shadow: none !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] details { border: none !important; }
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] summary {
    padding: 0.85rem 0.25rem; border-left: 2px solid transparent;
    transition: padding-left 0.15s ease, border-color 0.15s ease;
}
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
    font-size: 1.05rem; letter-spacing: 0.01em;
}
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary {
    padding-left: 0.65rem; border-left-color: %OLIVE%;
}
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover *,
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary * {
    color: %OLIVE_DARK%;
}

/* The open section floats over the rows below it, so nothing jumps while you hover */
.stApp [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
    position: absolute; left: 0; right: 0; top: 100%; z-index: 60;
    padding: 1rem 1rem 0.5rem; background: %BEIGE%;
    border: 1px solid rgba(58, 61, 36, 0.28); border-top: 2px solid %OLIVE%;
    box-shadow: 0 8px 20px rgba(58, 61, 36, 0.16);
}
.stApp [data-testid="stSidebar"] [data-testid="stExpanderDetails"] label p { font-size: 0.92rem; }

/* Hidden helper frame that enables open-on-hover */
.stApp [data-testid="stElementContainer"]:has(> iframe[height="0"]) { display: none; }

/* Link button styled as one more menu row */
.stApp [data-testid="stSidebar"] [data-testid^="stBaseLinkButton"] {
    justify-content: flex-start; background: transparent; border: none; border-radius: 0;
    border-bottom: 1px solid rgba(58, 61, 36, 0.22); padding: 0.85rem 0.25rem;
    border-left: 2px solid transparent; color: %TEXT%;
    transition: padding-left 0.15s ease, border-color 0.15s ease;
}
.stApp [data-testid="stSidebar"] [data-testid^="stBaseLinkButton"] * { color: %TEXT%; font-size: 1.05rem; }
.stApp [data-testid="stSidebar"] [data-testid^="stBaseLinkButton"]:hover,
.stApp [data-testid="stSidebar"] [data-testid^="stBaseLinkButton"]:focus-visible {
    padding-left: 0.65rem; border-left-color: %OLIVE%; background: transparent;
}
.stApp [data-testid="stSidebar"] [data-testid^="stBaseLinkButton"]:hover * { color: %OLIVE_DARK%; }

@media (prefers-reduced-motion: reduce) {
    .stApp [data-testid="stSidebar"] [data-testid="stExpander"] summary { transition: none; }
}
</style>
"""
for _k, _v in {"%BEIGE_DARK%": BEIGE_DARK, "%BEIGE_LITE%": BEIGE_LITE, "%BEIGE%": BEIGE,
               "%OLIVE_DARK%": OLIVE_DARK, "%OLIVE%": OLIVE, "%TEXT%": TEXT, "%FONT%": FONT}.items():
    THEME_CSS = THEME_CSS.replace(_k, _v)


APP_URL = "https://microplastic-countergit-d699me5vgzbug76zxaejfr.streamlit.app/"

HOVER_JS = """
<script>
(function () {
  const doc = window.parent.document;
  if (!window.parent.matchMedia("(hover: hover)").matches) return;   // touch: tap to open
  function bind() {
    doc.querySelectorAll('[data-testid="stSidebar"] [data-testid="stExpander"] details').forEach(function (d) {
      if (d.dataset.hoverMenu) return;
      d.dataset.hoverMenu = "1";
      let timer;
      const toggle = function (open) { if (d.open !== open) d.querySelector("summary").click(); };
      d.addEventListener("mouseenter", function () { clearTimeout(timer); toggle(true); });
      d.addEventListener("mouseleave", function () {
        if (d.contains(doc.activeElement) && doc.activeElement.tagName === "INPUT") return;  // keep open while typing
        timer = setTimeout(function () { toggle(false); }, 220);
      });
    });
  }
  bind();
  new MutationObserver(bind).observe(doc.body, { childList: true, subtree: true });
})();
</script>
"""


@st.cache_data(show_spinner=False, max_entries=8)
def decode(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


@st.cache_data(show_spinner=False, max_entries=16)
def run(data: bytes, **kw):
    """Cached on (image bytes, settings): moving an unrelated widget no longer re-runs the analysis."""
    return analyze(decode(data), **kw)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    st.set_page_config(page_title="Microplastic Counter", layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.title("Microplastic Counter")
    st.caption("Upload a microscope image to detect, size and classify particles.")

    components.html(HOVER_JS, height=0)

    with st.sidebar:
        with st.expander("Image"):
            um_per_px = st.number_input("Calibration (µm per pixel)", 0.01, 1000.0, 2.5, 0.1,
                                        help="Image a stage micrometer at the same magnification.")
            mode = st.radio("Image type", ["auto", "fluorescent", "dark"],
                            help="auto picks from the background brightness. fluorescent: bright particles on dark "
                                 "(Nile Red). dark: dark particles on a light filter.")
            channel = st.radio("Channel", ["g", "gray", "r", "b"], horizontal=True,
                               help="Which colour layer of the photo is analysed. g = green (best for Nile Red "
                                    "stained microplastics), r = red, b = blue, gray = all colours averaged.")
        with st.expander("Particle size"):
            min_um = st.number_input("Ignore particles smaller than (µm)", 0.0, 5000.0, 10.0)
            max_um = st.number_input("Ignore particles larger than (µm)", 1.0, 20000.0, 5000.0)
        with st.expander("Detection"):
            min_contrast = st.slider("Minimum contrast (0-255)", 5, 100, 25,
                                     help="Raise to ignore faint specks, texture and noise.")
            dish = st.checkbox("Petri dish / round filter: analyse inside only",
                               help="Stops the dish rim and its shadow being counted.")
            no_split = st.checkbox("Disable splitting of touching particles")
        with st.expander("Sample"):
            volume = st.number_input("Filtered water volume (L, optional)", 0.0, 1000.0, 0.0)
        st.link_button("Use our other tools", APP_URL, use_container_width=True)

    up = st.file_uploader("Microscope image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if up is None:
        st.info("Upload an image to begin.")
        return
    data = up.getvalue()
    img = decode(data)
    if img is None:
        st.error("Could not read that image.")
        return
    if min_um >= max_um:
        st.error("The minimum size must be smaller than the maximum size.")
        return

    with st.spinner("Analysing..."):
        res = run(data, um_per_px=um_per_px, mode=mode, channel=channel, min_um=min_um,
                  max_um=max_um, split_touching=not no_split, min_contrast=min_contrast, dish=dish)
    df = res.df

    st.caption("Image type used: " + ("dark particles on a light background" if res.mode == "dark"
               else "bright particles on a dark background"))
    for w in res.warnings:
        st.warning(w)
    if not res.valid:
        st.error("This image does not look like a microscope image of a particle sample, so the "
                 "counts would not be meaningful.")
        st.image(rgb(img), caption="Uploaded image", use_container_width=True)
        if not st.checkbox("Show the results anyway"):
            return

    c1, c2 = st.columns(2)
    c1.image(rgb(img), caption="Original", use_container_width=True)
    c2.image(rgb(draw_legend(res.annotated.copy())), caption="Detected", use_container_width=True)

    if df.empty:
        st.warning("No particles detected. Try the other image type or a different channel.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Particles", len(df))
    m2.metric("Likely MP (visual)", int((df.mp_likelihood == "likely").sum()))
    m3.metric("Median length (µm)", f"{df.length_um.median():.1f}")
    m4.metric("Density (per mm²)", f"{len(df) / res.area_mm2:.2f}")
    if volume > 0:
        st.metric("Concentration (particles/L)", f"{len(df) / volume:.1f}")

    a, b, c = st.columns(3)
    a.subheader("By shape")
    a.bar_chart(df["shape"].value_counts(), color=OLIVE)
    b.subheader("Visual MP triage")
    b.bar_chart(df["mp_likelihood"].value_counts(), color=OLIVE)
    c.subheader("By size class (µm)")
    cls = pd.cut(df.length_um, SIZE_BINS_UM, right=False).value_counts().sort_index()
    c.bar_chart(pd.Series(cls.values, index=[str(i) for i in cls.index]), color=OLIVE)

    ok, png = cv2.imencode(".png", draw_legend(res.annotated.copy()))
    d1, d2 = st.columns(2)
    d1.download_button("Download particle table (CSV)", df.to_csv(index=False), "particles.csv", "text/csv")
    if ok:
        d2.download_button("Download annotated image (PNG)", png.tobytes(), "annotated.png", "image/png")
    st.dataframe(df, use_container_width=True)
    st.caption("Shape and 'likelihood' labels are visual heuristics; the tool cannot confirm polymer type. "
               "Confirm candidates with a hot-needle test, FTIR or Raman.")


if __name__ == "__main__":
    main()
