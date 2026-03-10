import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import os

# ─── Config ──────────────────────────────────────────────────────────────────

BACKEND_URL = "https://handmade-ai-ten.vercel.app"  # Change to "http://localhost:8000" for local testing

st.set_page_config(
    page_title="HandmadeAI · Paint by Numbers",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700&display=swap');

    :root {
        --bg:           #FEFEFE;
        --surface:      #FFFFFF;
        --surface-alt:  #FFF5F3;
        --border:       #F0E0DC;
        --pink:         #F472B6;
        --pink-deep:    #E11D7B;
        --orange:       #FB923C;
        --gradient:     linear-gradient(135deg, #F472B6 0%, #FB923C 100%);
        --gradient-soft:linear-gradient(135deg, #FDE8EF 0%, #FEF0E4 100%);
        --text:         #1A1A2E;
        --text-sec:     #6B7280;
        --text-muted:   #9CA3AF;
        --shadow-sm:    0 1px 3px rgba(244,114,182,0.08);
        --shadow-md:    0 4px 16px rgba(244,114,182,0.10);
        --radius:       14px;
    }

    .stApp { background: var(--bg) !important; }
    header[data-testid="stHeader"] { background: transparent !important; }
    footer { display: none !important; }
    #MainMenu { display: none !important; }

    h1, h2, h3 {
        font-family: 'Sora', sans-serif !important;
        color: var(--text) !important;
        font-weight: 700 !important;
    }
    p, span, label, div, li, .stMarkdown, .stText {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text) !important;
    }

    /* ── Sidebar ──────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        width: 340px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
    }
    .sidebar-logo {
        display: flex; align-items: center; gap: 0.7rem;
        padding: 0.6rem 0 1rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    .sidebar-logo img { width: 40px; height: 40px; object-fit: contain; filter: drop-shadow(0 2px 6px rgba(244,114,182,0.2)); }
    .sidebar-logo .name {
        font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.25rem;
        background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sb-label {
        font-family: 'Sora', sans-serif !important; font-size: 0.72rem !important;
        font-weight: 600 !important; letter-spacing: 1.5px; text-transform: uppercase;
        color: var(--text-muted) !important; margin: 1rem 0 0.4rem;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        border: 2px dashed var(--border) !important; border-radius: 12px !important;
        padding: 0.7rem !important; background: var(--surface-alt) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover { border-color: var(--pink) !important; }
    section[data-testid="stSidebar"] [data-testid="stImage"] img {
        max-height: 160px !important; width: auto !important; max-width: 100% !important;
        border-radius: 10px; object-fit: contain; box-shadow: var(--shadow-sm);
    }

    /* Sliders */
    .stSlider > div > div > div > div { background: var(--gradient) !important; }
    .stSlider [data-testid="stThumbValue"] { color: var(--pink-deep) !important; font-weight: 600 !important; }

    /* Buttons */
    .stButton > button {
        font-family: 'Sora', sans-serif !important; font-weight: 600 !important;
        font-size: 0.88rem !important; border-radius: 12px !important;
        padding: 0.6rem 1.4rem !important; border: none !important;
        background: var(--gradient) !important; color: #fff !important;
        box-shadow: 0 4px 14px rgba(244,114,182,0.25) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(244,114,182,0.35) !important;
    }
    .stDownloadButton > button {
        font-family: 'Sora', sans-serif !important; font-weight: 600 !important;
        font-size: 0.8rem !important; border-radius: 10px !important;
        padding: 0.45rem 1rem !important; background: var(--surface-alt) !important;
        border: 1.5px solid var(--border) !important; color: var(--text) !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--pink) !important; color: var(--pink-deep) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* ── Main ─────────────────────────────────────── */
    .block-container { padding-top: 1rem !important; }

    .main-hero { text-align: center; padding: 1.5rem 1rem 0.3rem; }
    .main-hero img { width: 60px; height: 60px; object-fit: contain; filter: drop-shadow(0 4px 14px rgba(244,114,182,0.3)); margin-bottom: 0.4rem; }
    .main-hero .title {
        font-family: 'Sora', sans-serif; font-weight: 800; font-size: 2.2rem;
        background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        line-height: 1.15; margin-bottom: 0.2rem;
    }
    .main-hero .sub { font-family: 'Outfit', sans-serif; font-size: 0.95rem; color: var(--text-muted); font-weight: 400; }

    .empty-state { text-align: center; padding: 4rem 2rem; }
    .empty-state .big-icon { font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.5; }
    .empty-state p { color: var(--text-muted) !important; font-size: 1rem; }
    .empty-state .hint { font-size: 0.82rem !important; margin-top: 0.3rem; }

    /* Result cards */
    .result-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius); overflow: hidden;
        box-shadow: var(--shadow-sm); transition: box-shadow 0.3s ease, transform 0.3s ease;
    }
    .result-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
    .result-card .card-header {
        padding: 0.6rem 1rem; background: var(--surface-alt);
        border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 0.5rem;
    }
    .result-card .card-header .badge {
        font-family: 'Sora', sans-serif; font-weight: 600; font-size: 0.82rem; color: var(--text);
    }
    .badge-icon { width: 24px; height: 24px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; }
    .badge-pink { background: #FDE8EF; }
    .badge-orange { background: #FEF0E4; }

    .soft-divider { height: 1px; background: var(--gradient-soft); border: none; margin: 1.2rem 0; }

    /* ── Colour Key ───────────────────────────────── */
    .colour-key-wrap {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem 1.5rem;
        box-shadow: var(--shadow-sm);
    }
    .colour-key-title {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: var(--text);
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .colour-key-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
        gap: 0.55rem;
    }
    .ck-item {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.5rem 0.65rem;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--bg);
        transition: all 0.2s ease;
    }
    .ck-item:hover {
        border-color: var(--pink);
        box-shadow: var(--shadow-sm);
        transform: translateY(-1px);
    }
    .ck-swatch {
        width: 36px; height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
        box-shadow: inset 0 -1px 2px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.06);
    }
    .ck-info {
        display: flex;
        flex-direction: column;
        gap: 1px;
        min-width: 0;
    }
    .ck-info .ck-name {
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        font-size: 0.78rem;
        color: var(--text);
        white-space: nowrap;
    }
    .ck-info .ck-hex {
        font-family: 'Outfit', sans-serif;
        font-size: 0.7rem;
        color: var(--text-muted);
        letter-spacing: 0.5px;
    }

    .app-footer {
        text-align: center; padding: 2rem 0 1rem;
        font-family: 'Outfit', sans-serif; font-size: 0.72rem;
        letter-spacing: 1.5px; text-transform: uppercase; color: var(--text-muted);
    }
    .app-footer span {
        background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600;
    }

    .stSpinner > div > div { border-top-color: var(--pink) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load logo ───────────────────────────────────────────────────────────────

logo_b64 = ""
for p in ["logo.png", os.path.join(os.path.dirname(__file__), "logo.png")]:
    if os.path.exists(p):
        with open(p, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        break
logo_src = f"data:image/png;base64,{logo_b64}" if logo_b64 else ""


# ─── Helper: build colour key HTML ──────────────────────────────────────────

def build_colour_key_html(palette):
    """Return HTML for a gorgeous colour-key grid."""
    items = ""
    for i, rgb in enumerate(palette):
        r, g, b = rgb
        hex_c = f"#{r:02X}{g:02X}{b:02X}"
        # Text colour: white on dark swatches, dark on light
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        txt = "#fff" if lum < 140 else "#1A1A2E"
        items += f"""
        <div class="ck-item">
            <div class="ck-swatch" style="background:{hex_c}; color:{txt};">{i+1}</div>
            <div class="ck-info">
                <span class="ck-name">Colour {i+1}</span>
                <span class="ck-hex">{hex_c} · ({r}, {g}, {b})</span>
            </div>
        </div>"""

    return f"""
    <div class="colour-key-wrap">
        <div class="colour-key-title">🎨 Colour Key — {len(palette)} colours</div>
        <div class="colour-key-grid">{items}</div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    if logo_src:
        st.markdown(f'<div class="sidebar-logo"><img src="{logo_src}" /><div class="name">HandmadeAI</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-logo"><div style="font-size:1.6rem">🎨</div><div class="name">HandmadeAI</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-label">📷 Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag & drop or browse", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
    if uploaded_file:
        st.image(uploaded_file, caption="Original", use_container_width=False)

    st.markdown('<div class="sb-label">🎛️ Settings</div>', unsafe_allow_html=True)
    num_colors = st.slider("Colours", 2, 24, 8, 1, help="Fewer = bolder · More = finer")
    saturation = st.slider("Saturation", 0.5, 3.0, 1.5, 0.1, help="Boost colour vividness")

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    generate = st.button("✨  Generate", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

hero_img = f'<img src="{logo_src}" />' if logo_src else '<div style="font-size:3rem">🎨</div>'
st.markdown(f"""
<div class="main-hero">
    {hero_img}
    <div class="title">HandmadeAI</div>
    <div class="sub">Upload a photo, adjust settings, and generate your paint-by-numbers artwork</div>
</div>
<div class="soft-divider"></div>
""", unsafe_allow_html=True)


# ─── Process ─────────────────────────────────────────────────────────────────

if generate and uploaded_file:
    with st.spinner("Creating your masterpiece …"):
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"num_colors": str(num_colors), "saturation_factor": str(saturation)}

        try:
            resp = requests.post(f"{BACKEND_URL}/process", files=files, data=data, timeout=180)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("⚠️  Backend unreachable — run `uvicorn backend:app --reload` on port 8000")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.session_state["result"] = result
    st.session_state["last_file"] = uploaded_file.getvalue()
    st.session_state["last_name"] = uploaded_file.name
    st.session_state["last_type"] = uploaded_file.type
    st.session_state["last_colors"] = num_colors
    st.session_state["last_sat"] = saturation

elif generate and not uploaded_file:
    st.warning("Upload an image in the sidebar first!")


# ─── Display results ─────────────────────────────────────────────────────────

if "result" in st.session_state:
    result = st.session_state["result"]
    processed_b64 = result["processed"]
    linedraw_b64 = result["line_drawing"]
    palette = result.get("palette", [])

    # ── Row 1: Rendered + Stencil side by side ────────────────────────────────
    col_rendered, col_stencil = st.columns(2, gap="medium")

    with col_rendered:
        st.markdown("""
        <div class="result-card"><div class="card-header">
            <span class="badge-icon badge-pink">🖼</span>
            <span class="badge">Rendered Image</span>
        </div></div>""", unsafe_allow_html=True)
        st.image(Image.open(BytesIO(base64.b64decode(processed_b64))), use_container_width=True)

    with col_stencil:
        st.markdown("""
        <div class="result-card"><div class="card-header">
            <span class="badge-icon badge-orange">✏️</span>
            <span class="badge">Numbered Stencil</span>
        </div></div>""", unsafe_allow_html=True)
        st.image(Image.open(BytesIO(base64.b64decode(linedraw_b64))), use_container_width=True)

    # ── Row 2: Colour Key ─────────────────────────────────────────────────────
    if palette:
        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
        st.markdown(build_colour_key_html(palette), unsafe_allow_html=True)

    # ── Row 3: Download PDFs ──────────────────────────────────────────────────
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    dl1, dl_spacer, dl2 = st.columns([1, 0.2, 1])
    fb = st.session_state["last_file"]
    fn = st.session_state["last_name"]
    ft = st.session_state["last_type"]
    nc = st.session_state["last_colors"]
    sat = st.session_state["last_sat"]

    with dl1:
        st.write("")  # spacer
    with dl_spacer:
        st.write("")
    with dl2:
        st.write("")

    try:
        pdf_r = requests.post(
            f"{BACKEND_URL}/download_pdf",
            files={"file": (fn, fb, ft)},
            data={"num_colors": str(nc), "saturation_factor": str(sat)},
            timeout=180,
        )
        pdf_r.raise_for_status()
        st.download_button(
            "⬇  Download Complete PDF  (Rendered + Stencil + Colour Key)",
            pdf_r.content,
            "handmadeai_paint_by_numbers.pdf",
            "application/pdf",
            use_container_width=True,
        )
    except Exception:
        st.warning("Could not generate PDF.")

else:
    st.markdown("""
    <div class="empty-state">
        <div class="big-icon">🖌️</div>
        <p>Your artwork will appear here</p>
        <p class="hint">Upload an image in the sidebar and hit Generate</p>
    </div>
    """, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-footer">
    <span>HandmadeAI</span> &nbsp;·&nbsp; Pixels into paintings &nbsp;·&nbsp; Made with ❤️
</div>
""", unsafe_allow_html=True)