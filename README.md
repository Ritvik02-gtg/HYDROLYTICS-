# HYDROLYTICS ( microplastic counter )
# {Turning microscopic water evidence into faster, smarter sustainability action}

A free, browser-based tool that counts, measures and classifies microplastic particles from a single photograph or microscope image of a filter or Petri dish. It needs no training data, no GPU and no coding: upload an image, adjust a few settings and read the results.

**Main app:** https://teamhydrolytics.streamlit.app/

> This is a visual screening tool. It counts and measures particles but does not identify the polymer. Candidate particles should be confirmed with a method that identifies the material, such as a hot-needle test, FTIR or Raman spectroscopy.

---

## Why this project

Counting microplastics by eye under a microscope is slow and varies from person to person. Recent deep learning approaches (for example YOLOv7 and Mask R-CNN) are accurate but usually need large labelled datasets, GPU resources and programming skill, and some are not openly available. This project takes a different route: classical image processing wrapped in a simple web interface, so schools, field groups and small laboratories can screen samples quickly and consistently.

## Integrated tools

This project brings together three sub-tools and a chatbot in one platform, the [main app](https://teamhydrolytics.streamlit.app/). They are integrated using the repositories published at [github.com/saithoshithb](https://github.com/saithoshithb).

## Sustainable Development Goals

This project supports four of the United Nations Sustainable Development Goals:

| SDG | Goal | How the project contributes |
|---|---|---|
| **SDG 6** | Clean Water and Sanitation | Gives a fast, low-cost way to screen filtered water samples for microplastic particles, supporting water-quality monitoring |
| **SDG 9** | Industry, Innovation and Infrastructure | Offers an open, accessible image-analysis tool that needs no GPU, training data or coding, bringing a research method to schools and small laboratories |
| **SDG 12** | Responsible Consumption and Production | Produces counts, shapes and sizes of particles that help show how much plastic is entering the environment and support efforts to reduce it |
| **SDG 14** | Life Below Water | Helps monitor microplastic pollution that harms aquatic ecosystems and marine organisms |

The tool is a screening aid, so it supports these goals by making monitoring faster and more widely available rather than by replacing confirmatory laboratory analysis.

## Features

- Works on **fluorescent** images (bright particles on a dark background, e.g. Nile Red staining) and **dark-particle** images (dark particles on a light filter), with automatic mode detection
- Choice of colour channel (green, red, blue or gray)
- Automatic background flattening for uneven lighting
- Contrast threshold that rises automatically on grainy images
- Optional **Petri dish mode** that analyses only the inside of a round dish or filter, so the rim and its shadow are not counted
- Separation of touching particles (can be turned off)
- Calibrated sizes in micrometres from a single µm-per-pixel value
- Shape classes: fibre, bead, film and fragment
- Visual likelihood label for each particle: likely, possible or uncertain
- Summary figures and charts, and downloads of the results

## How it works

1. **Read the image.** The photo is decoded and, if wider than 2400 px, shrunk (the calibration is adjusted to match).
2. **Pick a channel and mode.** A single colour channel is extracted. In auto mode, the median brightness decides between fluorescent and dark mode.
3. **Flatten the background.** A heavy median blur estimates the lighting, and subtracting it leaves a signal image in which particles are bright.
4. **Set the noise floor.** Image grain is measured, and the contrast threshold is the larger of the slider value and about six times the noise.
5. **Find particles.** Otsu thresholding, followed by morphological opening and closing, creates a particle mask. A distance transform and watershed separate touching particles.
6. **Measure each particle.** Contours give area, perimeter, longest and shortest side, circularity, aspect ratio and solidity. Size limits are applied in micrometres.
7. **Classify.** Rule-based shape classes and a visual likelihood label are assigned.
8. **Report.** Results are shown in the interface and can be downloaded.

## Using the app

The repository also includes a `demo pics` folder with sample images, so you can try the tool straight away without your own photos.

1. Upload a photo (PNG or JPG).
2. Open the sections in the left menu and set your options.
3. Read the annotated image, summary figures, charts and particle table.
4. Download the CSV and the annotated PNG.

### Left menu settings

| Section | Setting | Purpose |
|---|---|---|
| Image | Calibration (µm per pixel) | Converts pixels to real sizes. Image a stage micrometer or ruler at the same magnification to find it. |
| Image | Image type | `fluorescent`, `dark` or `auto` |
| Image | Channel | Colour layer analysed. Green works best for Nile Red. |
| Particle size | Ignore smaller / larger than (µm) | Sets the size window of particles to count |
| Detection | Minimum contrast (0-255) | Raise to ignore faint specks and noise |
| Detection | Petri dish / round filter | Analyse inside the dish only |
| Detection | Disable splitting | Count touching particles as one |
| Sample | Filtered water volume (L) | Adds a particles-per-litre figure |

### Results

- **Summary figures:** particle count, likely microplastics (visual), median length, density per mm², and concentration per litre if a volume is entered
- **Charts:** particles by shape, by visual likelihood and by size class (0-20, 20-50, 50-100, 100-300, 300-1000 and 1000-5000 µm)
- **Particle table columns:** length, width, area, contrast, aspect ratio, circularity, mean colour, saturation, shape and likelihood
- **Downloads:** `particles.csv` and the annotated image

### Shape definitions

| Shape | Rule (summary) |
|---|---|
| Fibre | Long and thin (length-to-width ratio of 3.5 or more) or a very thin curly outline |
| Bead | Nearly round (circularity of 0.87 or more, aspect ratio under 1.2) |
| Film | Flat, ragged or hollow piece (solidity below 0.70) |
| Fragment | Any other solid, irregular piece |

These are visual estimates from the outline, not material identification. The thresholds can be tuned in the code.

## Run locally

Requires Python 3.9 or newer.

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Command-line use

`microplastic_counter.py` is the standalone version of the analysis and needs no web interface.

```bash
python microplastic_counter.py image.png --um-per-px 2.5
python microplastic_counter.py image.png --um-per-px 2.5 --mode dark --min-um 20
python microplastic_counter.py dish.jpg --mode dark --dish
python microplastic_counter.py --demo
```

Options include `--mode`, `--channel`, `--min-um`, `--max-um`, `--min-contrast`, `--volume-L`, `--no-split`, `--dish` and `--demo`. Outputs are `<name>_annotated.png`, `<name>_particles.csv` and `<name>_summary.txt`.

## Repository structure

```
.
├── demo pics/                 Sample images for trying the tool
├── app.py                     Streamlit web app (interface and analysis)
├── microplastic_counter.py    Standalone command-line version of the analysis
├── config.toml                Streamlit theme (cream, sand and olive colours)
├── requirements.txt           Python dependencies
└── README.md
```

To apply the custom theme on Streamlit, place `config.toml` inside a `.streamlit/` folder at the project root.

## Built with

| Library | Used for |
|---|---|
| Streamlit | Web interface, widgets, charts, tables and downloads |
| OpenCV (headless) | Image reading, thresholding, dish detection, contours, drawing |
| NumPy | Image arrays and numerical operations |
| SciPy (`ndimage`) | Labelling particles, finding peaks, fast per-particle measurement |
| pandas | Particle table, summaries, size classes and CSV export |

## Limitations

- Accuracy has not yet been validated against manual counts or spectroscopic confirmation.
- The tool cannot tell plastic from other material, such as natural fibres or organic matter. Nile Red can also stain fats and some organic matter.
- It is designed for sparse particle samples. Very cluttered images may need setting adjustments, and the app warns when a photo does not look like a sparse sample.
- Results depend on the calibration value you enter. A wrong µm-per-pixel value shifts every size and density figure.
- The per-litre figure assumes the photo covers the whole filter.

## Tips for good results

- Calibrate with a stage micrometer or ruler at the same magnification as your sample.
- Use even lighting and a clean, in-focus image.
- For Nile Red samples, use fluorescent mode and the green channel.
- If the count looks too high, raise the minimum contrast or turn on Petri dish mode. If it looks too low, lower the contrast.
- Compare the app's count with a manual count on a few images before relying on it.
