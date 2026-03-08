from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from sklearn.cluster import KMeans
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas as pdf_canvas
from PIL import Image, ImageDraw, ImageFont
import base64
import tempfile
import os
import json


app = FastAPI(title="HandmadeAI - Paint by Numbers")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Image Processing ────────────────────────────────────────────────────────

def process_image_full(input_image: np.ndarray, num_colors: int, saturation_factor: float = 1.5):
    """
    Full pipeline: returns (processed_image, numbered_stencil, palette_rgb, labels_map).
    - processed_image : vibrant colour-quantized BGR image
    - numbered_stencil: white background, black edges, region numbers (BGR)
    - palette_rgb     : list of [R, G, B] for each colour index 1..N
    - labels_map      : 2-D array where each pixel = colour index (0-based)
    """

    # ── 1. Colour quantization ────────────────────────────────────────────────
    input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    h, w = input_rgb.shape[:2]
    pixels = input_rgb.reshape((-1, 3)).astype(np.float64)

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)   # RGB
    labels = kmeans.labels_.reshape((h, w))

    # ── 2. Sort colours by luminance so numbering feels natural ───────────────
    luminance = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
    sort_order = np.argsort(luminance)          # darkest → lightest
    remap = np.zeros(num_colors, dtype=int)
    for new_idx, old_idx in enumerate(sort_order):
        remap[old_idx] = new_idx
    labels = remap[labels]
    centers = centers[sort_order]

    # ── 3. Processed (vibrant) image ──────────────────────────────────────────
    segmented_rgb = centers[labels]
    segmented_bgr = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR)
    stylized = cv2.bilateralFilter(segmented_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    hsv = cv2.cvtColor(stylized, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── 4 & 5. Smooth SVG-like contour stencil ──────────────────────────────
    # Instead of Canny edges (pixelated), we trace contours per colour region
    # and draw them with anti-aliased polylines for clean, sleek boundaries.

    # Work at 2× resolution for super-smooth anti-aliased lines, then downscale
    scale = 2
    canvas_h, canvas_w = h * scale, w * scale
    stencil_hr = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    line_color = (140, 140, 140)       # medium grey for edges
    line_thickness = max(1, int(round(min(h, w) / 800)) * scale)  # thin, sleek lines

    for color_idx in range(num_colors):
        mask = (labels == color_idx).astype(np.uint8) * 255

        # Upscale the mask to 2× using nearest-neighbour (keeps region shapes)
        mask_hr = cv2.resize(mask, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)

        # Light morphological close to remove tiny holes / noise
        morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * scale, 3 * scale))
        mask_hr = cv2.morphologyEx(mask_hr, cv2.MORPH_CLOSE, morph_k)

        # Find contours
        contours, _ = cv2.findContours(mask_hr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 40 * scale * scale:
                continue   # skip dust-sized regions

            # Approximate the contour to reduce jaggedness while keeping shape
            epsilon = 0.8 * scale   # tighter ≈ smoother curves
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)

            # Draw with anti-aliased (LINE_AA) rendering
            cv2.drawContours(stencil_hr, [approx], -1, line_color,
                             thickness=line_thickness, lineType=cv2.LINE_AA)

    # Downscale back to original size — area interpolation acts as anti-alias
    stencil_np = cv2.resize(stencil_hr, (w, h), interpolation=cv2.INTER_AREA)

    # Convert to PIL for text drawing
    stencil_pil = Image.fromarray(cv2.cvtColor(stencil_np, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(stencil_pil)

    # Choose font size based on image resolution
    base_font_size = max(10, min(h, w) // 50)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", base_font_size)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(8, base_font_size - 3))
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    # For each colour, find connected regions and label the largest ones
    for color_idx in range(num_colors):
        mask = (labels == color_idx).astype(np.uint8)
        # Find connected components
        num_labels_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for cc_idx in range(1, num_labels_cc):  # skip background (0)
            area = stats[cc_idx, cv2.CC_STAT_AREA]
            # Only label regions large enough to fit a number
            min_area = max(100, (base_font_size * base_font_size) // 2)
            if area < min_area:
                continue

            cx, cy = int(centroids[cc_idx][0]), int(centroids[cc_idx][1])

            # Make sure centroid is actually inside the region
            # If not, find the nearest pixel that is
            if not (0 <= cy < h and 0 <= cx < w and labels[cy, cx] == color_idx):
                # Use distance transform to find interior point
                region_mask = (cc_labels == cc_idx).astype(np.uint8)
                dist = cv2.distanceTransform(region_mask, cv2.DIST_L2, 5)
                max_loc = np.unravel_index(np.argmax(dist), dist.shape)
                cy, cx = int(max_loc[0]), int(max_loc[1])

            # Draw the number (1-indexed)
            label_text = str(color_idx + 1)
            bbox = draw.textbbox((0, 0), label_text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # Use smaller font for small regions
            chosen_font = font
            if area < min_area * 4:
                chosen_font = font_small
                bbox = draw.textbbox((0, 0), label_text, font=font_small)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            tx = cx - tw // 2
            ty = cy - th // 2

            # Draw with slight white halo for readability
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    draw.text((tx + dx, ty + dy), label_text, fill=(255, 255, 255), font=chosen_font)
            draw.text((tx, ty), label_text, fill=(80, 80, 80), font=chosen_font)

    stencil_final = np.array(stencil_pil)
    numbered_stencil = cv2.cvtColor(stencil_final, cv2.COLOR_RGB2BGR)

    # palette as list of [R,G,B]
    palette_rgb = centers.tolist()

    return processed, numbered_stencil, palette_rgb, labels


def image_to_base64(image: np.ndarray, fmt: str = ".png") -> str:
    _, buffer = cv2.imencode(fmt, image)
    return base64.b64encode(buffer).decode("utf-8")


# ─── PDF Builders ─────────────────────────────────────────────────────────────

def _fit_image_on_page(c, tmp_path, page_w, page_h, margin, img_w, img_h, y_offset=0):
    """Draw an image centred on page, return (draw_w, draw_h, y_bottom)."""
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin - y_offset
    aspect = img_w / img_h

    if usable_w / usable_h > aspect:
        draw_h = usable_h
        draw_w = draw_h * aspect
    else:
        draw_w = usable_w
        draw_h = draw_w / aspect

    x = (page_w - draw_w) / 2
    y = (page_h - draw_h) / 2 - y_offset / 2

    c.drawImage(tmp_path, x, y, width=draw_w, height=draw_h)
    return draw_w, draw_h, y


def _draw_branding(c, page_w, page_h, margin):
    """Tiny branding text in the top-right corner of every page."""
    c.saveState()
    c.setFont("Helvetica", 6)
    c.setFillColor(Color(0.65, 0.65, 0.65))
    c.drawRightString(page_w - margin, page_h - margin + 2, "HandmadeAI")
    c.restoreState()


def build_full_pdf(processed_bgr: np.ndarray, stencil_bgr: np.ndarray,
                   palette_rgb: list, page_size=A4) -> bytes:
    """
    Build a single 3-page PDF:
      Page 1 – Rendered (colour) image
      Page 2 – Numbered stencil
      Page 3 – Colour key with numbered swatches
    """
    page_w, page_h = page_size
    margin = 0.4 * inch
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=page_size)

    # Helper to write one full-bleed image page (no title)
    def _add_image_page(image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if len(image_bgr.shape) == 3 else image_bgr
        pil_img = Image.fromarray(rgb)
        img_h, img_w = image_bgr.shape[:2]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_img.save(tmp, format="PNG")
            tmp_path = tmp.name

        _draw_branding(c, page_w, page_h, margin)
        _fit_image_on_page(c, tmp_path, page_w, page_h, margin, img_w, img_h)
        os.unlink(tmp_path)
        c.showPage()

    # ── Page 1: Rendered Image ────────────────────────────────────────────────
    _add_image_page(processed_bgr)

    # ── Page 2: Numbered Stencil ──────────────────────────────────────────────
    _add_image_page(stencil_bgr)

    # ── Page 3: Colour Key ────────────────────────────────────────────────────
    _draw_branding(c, page_w, page_h, margin)

    num = len(palette_rgb)

    # Layout swatches in a grid
    cols = 3 if num > 6 else 2
    rows = (num + cols - 1) // cols

    swatch_w = 22 * mm
    swatch_h = 14 * mm
    gap_x = 12 * mm
    gap_y = 8 * mm
    text_w = 42 * mm

    cell_w = swatch_w + text_w + gap_x
    cell_h = swatch_h + gap_y

    grid_w = cols * cell_w - gap_x
    grid_h = rows * cell_h - gap_y

    start_x = (page_w - grid_w) / 2

    # Center the grid vertically on the page
    available_h = page_h - margin * 2
    start_y = page_h - margin - (available_h - grid_h) / 2

    for idx, rgb_val in enumerate(palette_rgb):
        col_i = idx % cols
        row_i = idx // cols

        x = start_x + col_i * cell_w
        y = start_y - row_i * cell_h

        r, g, b = rgb_val[0] / 255, rgb_val[1] / 255, rgb_val[2] / 255

        c.setFillColor(Color(r, g, b))
        c.setStrokeColor(Color(0.82, 0.82, 0.82))
        c.setLineWidth(0.6)
        c.roundRect(x, y - swatch_h, swatch_w, swatch_h, 3 * mm, fill=1, stroke=1)

        lum = 0.299 * rgb_val[0] + 0.587 * rgb_val[1] + 0.114 * rgb_val[2]
        if lum < 128:
            c.setFillColor(Color(1, 1, 1))
        else:
            c.setFillColor(Color(0.15, 0.15, 0.15))

        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(x + swatch_w / 2, y - swatch_h / 2 - 5, str(idx + 1))

        c.setFillColor(Color(0.15, 0.15, 0.15))
        c.setFont("Helvetica-Bold", 11)
        text_x = x + swatch_w + 4 * mm
        c.drawString(text_x, y - 5, f"Colour {idx + 1}")

        c.setFont("Helvetica", 9)
        c.setFillColor(Color(0.45, 0.45, 0.45))
        hex_str = f"#{rgb_val[0]:02X}{rgb_val[1]:02X}{rgb_val[2]:02X}"
        c.drawString(text_x, y - 17, hex_str)
        c.drawString(text_x, y - 28, f"RGB({rgb_val[0]}, {rgb_val[1]}, {rgb_val[2]})")

    c.save()
    return buf.getvalue()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/process")
async def process_image_route(
    file: UploadFile = File(...),
    num_colors: int = Form(8),
    saturation_factor: float = Form(1.5),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image"}

    processed, numbered_stencil, palette_rgb, _ = process_image_full(
        img, num_colors, saturation_factor
    )

    return {
        "processed": image_to_base64(processed),
        "line_drawing": image_to_base64(numbered_stencil),
        "palette": palette_rgb,   # [[R,G,B], ...]  1-indexed logically
    }


@app.post("/download_pdf")
async def download_pdf(
    file: UploadFile = File(...),
    num_colors: int = Form(8),
    saturation_factor: float = Form(1.5),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image"}

    processed, numbered_stencil, palette_rgb, _ = process_image_full(
        img, num_colors, saturation_factor
    )

    pdf_bytes = build_full_pdf(processed, numbered_stencil, palette_rgb)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="handmadeai_paint_by_numbers.pdf"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)