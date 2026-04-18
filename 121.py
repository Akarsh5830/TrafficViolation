import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Traffic Violation Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS  – dark navy dashboard theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:   #0A0E1A;
    --bg-card:      #111827;
    --bg-card2:     #1a2235;
    --accent:       #3B82F6;
    --accent2:      #6366F1;
    --accent3:      #10B981;
    --danger:       #EF4444;
    --warning:      #F59E0B;
    --text-primary: #F1F5F9;
    --text-muted:   #94A3B8;
    --border:       rgba(59,130,246,0.18);
}

/* ── App shell ── */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D1321 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

/* ── Sidebar logo block ── */
.sidebar-logo {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d1f3c 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    margin-bottom: 24px;
}
.sidebar-logo .logo-icon {
    font-size: 2.8rem;
    display: block;
    margin-bottom: 8px;
}
.sidebar-logo .logo-title {
    color: #F1F5F9;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.sidebar-logo .logo-sub {
    color: #64748B;
    font-size: 0.70rem;
    margin-top: 2px;
}

/* ── Sidebar nav items ── */
.nav-section {
    color: #475569;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 20px 4px 8px 4px;
}
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 10px;
    color: #94A3B8;
    font-size: 0.87rem;
    font-weight: 500;
    cursor: pointer;
    margin-bottom: 4px;
    transition: all 0.2s;
}
.nav-item.active {
    background: rgba(59,130,246,0.15);
    color: #60A5FA;
    border-left: 3px solid #3B82F6;
}

/* ── Sidebar stat box ── */
.sidebar-stat {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
}
.sidebar-stat .stat-label {
    color: #64748B;
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.sidebar-stat .stat-value {
    color: #F1F5F9;
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 2px;
}
.sidebar-stat .stat-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
    margin-top: 4px;
}
.badge-green  { background: rgba(16,185,129,0.15); color: #10B981; }
.badge-blue   { background: rgba(59,130,246,0.15); color: #60A5FA; }
.badge-yellow { background: rgba(245,158,11,0.15); color: #F59E0B; }
.badge-red    { background: rgba(239,68,68,0.15);  color: #EF4444; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #0f2044 0%, #0a1628 50%, #0d1f3c 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -20%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse, rgba(59,130,246,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.page-header .header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    color: #60A5FA;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.page-header h1 {
    color: #F1F5F9 !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.02em;
}
.page-header p {
    color: #64748B;
    font-size: 0.90rem;
    margin: 0;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(59,130,246,0.35);
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--card-accent, #3B82F6);
}
.metric-card .mc-icon {
    width: 40px; height: 40px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    margin-bottom: 14px;
    background: var(--icon-bg, rgba(59,130,246,0.12));
}
.metric-card .mc-value {
    color: #F1F5F9;
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-card .mc-label {
    color: #64748B;
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* ── Upload zone ── */
.upload-zone {
    background: var(--bg-card);
    border: 2px dashed rgba(59,130,246,0.30);
    border-radius: 20px;
    padding: 48px 24px;
    text-align: center;
    transition: border-color 0.3s, background 0.3s;
}
.upload-zone:hover {
    border-color: rgba(59,130,246,0.60);
    background: rgba(59,130,246,0.04);
}
.upload-zone .uz-icon { font-size: 3rem; margin-bottom: 14px; display: block; }
.upload-zone .uz-title { color: #F1F5F9; font-size: 1.05rem; font-weight: 600; }
.upload-zone .uz-sub   { color: #64748B; font-size: 0.82rem; margin-top: 4px; }

/* ── Section heading ── */
.section-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
}
.section-heading .sh-dot {
    width: 4px; height: 22px;
    background: linear-gradient(180deg, #3B82F6, #6366F1);
    border-radius: 2px;
}
.section-heading .sh-title {
    color: #F1F5F9;
    font-size: 1.0rem;
    font-weight: 700;
}

/* ── Panel card ── */
.panel-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px;
    height: 100%;
}

/* ── Violation tag ── */
.vtag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(239,68,68,0.10);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 10px;
    padding: 10px 16px;
    margin: 5px 0;
    width: 100%;
}
.vtag .vtag-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #EF4444;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(239,68,68,0.6);
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
}
.vtag .vtag-name {
    color: #FCA5A5;
    font-size: 0.84rem;
    font-weight: 600;
}
.vtag-safe {
    background: rgba(16,185,129,0.10) !important;
    border-color: rgba(16,185,129,0.25) !important;
}
.vtag-safe .vtag-dot { background: #10B981 !important; box-shadow: 0 0 8px rgba(16,185,129,0.6) !important; animation: none !important; }
.vtag-safe .vtag-name { color: #6EE7B7 !important; }

/* ── Progress bar ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.conf-label { color: #94A3B8; font-size: 0.78rem; width: 170px; flex-shrink: 0; }
.conf-bar-bg {
    flex: 1;
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #3B82F6, #6366F1);
}
.conf-pct { color: #F1F5F9; font-size: 0.78rem; font-weight: 600; width: 38px; text-align: right; }

/* ── Streamlit widget overrides ── */
.stSlider > div { color: #94A3B8 !important; }
[data-testid="stSlider"] .rc-slider-track { background: #3B82F6 !important; }
[data-testid="stSlider"] .rc-slider-handle { border-color: #3B82F6 !important; background: #3B82F6 !important; }
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #4F46E5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.30) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed rgba(59,130,246,0.28) !important;
    border-radius: 16px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"] label { color: #94A3B8 !important; font-size: 0.85rem !important; }
.stTabs [data-baseweb="tab-list"] {
    background: #111827 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.87rem !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1D4ED8, #4338CA) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.30) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 20px !important;
}
/* progress bar */
.stProgress > div > div > div { background: linear-gradient(90deg, #3B82F6, #6366F1) !important; }
/* download button */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #059669, #047857) !important;
    box-shadow: 0 4px 15px rgba(5,150,105,0.30) !important;
}
/* selectbox / slider labels */
.stSelectbox label, .stSlider label { color: #94A3B8 !important; font-size: 0.82rem !important; font-weight: 500 !important; }
/* divider */
hr { border-color: var(--border) !important; }
/* images */
[data-testid="stImage"] { border-radius: 14px !important; overflow: hidden !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  MODEL & CONSTANTS
# ══════════════════════════════════════════════════════════════════
CLASS_META = {
    'Number_plate':                         {'icon': '🔢', 'color': '#3B82F6',  'short': 'Number Plate'},
    'mobile_usage':                         {'icon': '📱', 'color': '#F59E0B',  'short': 'Mobile Usage'},
    'pillion_rider_not_wearing_helmet':     {'icon': '🪖', 'color': '#8B5CF6',  'short': 'Pillion No Helmet'},
    'rider_and_pillion_not_wearing_helmet': {'icon': '⛑',  'color': '#EC4899',  'short': 'Both No Helmet'},
    'rider_not_wearing_helmet':             {'icon': '🚫', 'color': '#EF4444',  'short': 'Rider No Helmet'},
    'triple_riding':                        {'icon': '👥', 'color': '#10B981',  'short': 'Triple Riding'},
    'vehicle_with_offence':                 {'icon': '⚠️', 'color': '#F97316',  'short': 'Vehicle Offence'},
}
CLASS_NAMES = list(CLASS_META.keys())

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return YOLO("best.pt")
    except Exception:
        return None

model = load_model()

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
def fmt(name: str) -> str:
    return CLASS_META.get(name, {}).get('short', name.replace('_', ' ').title())

def get_color(name: str):
    hex_c = CLASS_META.get(name, {}).get('color', '#3B82F6')
    h = hex_c.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))   # BGR not needed; we use BGR below

def hex_to_bgr(hex_c: str):
    h = hex_c.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

def draw_boxes(image: np.ndarray, results, conf_thresh: float = 0.25) -> np.ndarray:
    img = image.copy()
    h_img, w_img = img.shape[:2]
    if results[0].boxes is None:
        return img

    boxes  = results[0].boxes.xyxy.cpu().numpy().astype(int)
    clss   = results[0].boxes.cls.cpu().numpy().astype(int)
    confs  = results[0].boxes.conf.cpu().numpy()
    used_areas = []

    for box, cls_id, conf in zip(boxes, clss, confs):
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = box
        name  = CLASS_NAMES[cls_id]
        label = f"{fmt(name)} {conf:.2f}"
        bgr   = hex_to_bgr(CLASS_META.get(name, {}).get('color', '#3B82F6'))

        # box
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
        # corner accents
        L = min(20, (x2-x1)//4, (y2-y1)//4)
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(img, (cx,cy), (cx+dx*L, cy), bgr, 3)
            cv2.line(img, (cx,cy), (cx, cy+dy*L), bgr, 3)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        positions = [
            (x1, y1-6), (x1, y2+th+8),
            (x1+10, y1-26), (x1+10, y2+26)
        ]
        fx, fy = x1, y1
        for px, py in positions:
            px = max(0, min(px, w_img-tw))
            py = max(th, min(py, h_img))
            overlap = any(
                not (px+tw < ux1 or px > ux2 or py < uy1 or py-th > uy2)
                for ux1,uy1,ux2,uy2 in used_areas
            )
            if not overlap:
                fx, fy = px, py
                break

        used_areas.append((fx, fy-th, fx+tw, fy))
        # pill background
        cv2.rectangle(img, (fx-2, fy-th-4), (fx+tw+4, fy+2), bgr, -1)
        cv2.putText(img, label, (fx, fy-3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)

    return img

def violation_summary(results):
    """Return list of (name, conf) for detected violations."""
    if results[0].boxes is None:
        return []
    out = {}
    for cls_id, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
        name = CLASS_NAMES[int(cls_id)]
        if name not in out or conf > out[name]:
            out[name] = float(conf)
    return sorted(out.items(), key=lambda x: -x[1])

# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
if 'total_processed' not in st.session_state:
    st.session_state.total_processed = 0
if 'total_violations' not in st.session_state:
    st.session_state.total_violations = 0
if 'violation_counts' not in st.session_state:
    st.session_state.violation_counts = {k: 0 for k in CLASS_NAMES}

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🚦</span>
        <div class="logo-title">TrafficVision AI</div>
        <div class="logo-sub">YOLOv8 · Violation Detection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Detection Settings</div>', unsafe_allow_html=True)

    confidence = st.slider("Confidence Threshold", 0.10, 1.0, 0.40, 0.05,
                           help="Minimum confidence score for a detection to be shown")
    iou_thresh = st.slider("IoU Threshold (NMS)", 0.10, 1.0, 0.45, 0.05,
                           help="Overlap threshold for Non-Maximum Suppression")

    st.markdown('<div class="nav-section">Model Info</div>', unsafe_allow_html=True)
    model_status = "🟢 Loaded" if model else "🔴 Not Found"
    st.markdown(f"""
    <div class="sidebar-stat">
        <div class="stat-label">Model Status</div>
        <div class="stat-value" style="font-size:1.0rem;">{model_status}</div>
        <span class="stat-badge badge-blue">YOLOv8</span>
    </div>
    <div class="sidebar-stat">
        <div class="stat-label">Detection Classes</div>
        <div class="stat-value">7</div>
        <span class="stat-badge badge-green">Active</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Session Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-stat">
        <div class="stat-label">Files Processed</div>
        <div class="stat-value">{st.session_state.total_processed}</div>
    </div>
    <div class="sidebar-stat">
        <div class="stat-label">Violations Found</div>
        <div class="stat-value">{st.session_state.total_violations}</div>
        <span class="stat-badge badge-red">{'⚠ Alert' if st.session_state.total_violations > 0 else 'Clear'}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Violation Classes</div>', unsafe_allow_html=True)
    for name, meta in CLASS_META.items():
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:5px 4px;
                    border-radius:8px;margin-bottom:2px;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:{meta['color']};flex-shrink:0;"></div>
            <span style="color:#94A3B8;font-size:0.78rem;">{meta['icon']} {meta['short']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#334155;font-size:0.70rem;text-align:center;padding:8px;">
        BBD University · Final Year Project<br>
        CSE-AI · 2026
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════

# ── Page header ──
st.markdown("""
<div class="page-header">
    <div class="header-badge">🚦 AI-Powered • Real-Time • Multi-Class</div>
    <h1>Traffic Violation Detection</h1>
    <p>Automated surveillance analysis using YOLOv8 deep learning — detect helmet violations,
       triple riding, mobile usage, and more from images or video.</p>
</div>
""", unsafe_allow_html=True)

# ── Metric cards ──
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card" style="--card-accent:#3B82F6;">
        <div class="mc-icon" style="--icon-bg:rgba(59,130,246,0.12);">🎯</div>
        <div class="mc-value">92%</div>
        <div class="mc-label">Model Accuracy</div>
    </div>
    <div class="metric-card" style="--card-accent:#10B981;">
        <div class="mc-icon" style="--icon-bg:rgba(16,185,129,0.12);">⚡</div>
        <div class="mc-value">32 FPS</div>
        <div class="mc-label">Inference Speed</div>
    </div>
    <div class="metric-card" style="--card-accent:#6366F1;">
        <div class="mc-icon" style="--icon-bg:rgba(99,102,241,0.12);">🔍</div>
        <div class="mc-value">7</div>
        <div class="mc-label">Violation Classes</div>
    </div>
    <div class="metric-card" style="--card-accent:#F59E0B;">
        <div class="mc-icon" style="--icon-bg:rgba(245,158,11,0.12);">📊</div>
        <div class="mc-value">{st.session_state.total_violations}</div>
        <div class="mc-label">Session Violations</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["📸  Image Detection", "🎥  Video Detection", "📋  About & Classes"])

# ════════════════════════════════════════════════════════════
#  TAB 1 — IMAGE
# ════════════════════════════════════════════════════════════
with tab1:
    col_up, col_res = st.columns([1, 1.4], gap="large")

    with col_up:
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">Upload Image</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="img_uploader"
        )
        st.markdown("""
        <div style="color:#475569;font-size:0.76rem;text-align:center;margin-top:6px;">
            Supports JPG · PNG · JPEG
        </div>
        """, unsafe_allow_html=True)

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Image", use_container_width=True)

            run_btn = st.button("🔍  Run Detection", use_container_width=True, key="img_run")
        else:
            st.markdown("""
            <div class="upload-zone">
                <span class="uz-icon">📂</span>
                <div class="uz-title">Upload a surveillance image</div>
                <div class="uz-sub">JPG · PNG · JPEG supported</div>
            </div>
            """, unsafe_allow_html=True)
            run_btn = False

    with col_res:
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">Detection Results</div>
        </div>
        """, unsafe_allow_html=True)

        if uploaded_file and run_btn:
            if model is None:
                st.error("⚠️ Model file `best.pt` not found. Place it in the app directory.")
            else:
                with st.spinner("Analyzing image…"):
                    img_np  = np.array(image)
                    results = model.predict(img_np, conf=confidence, iou=iou_thresh, verbose=False)
                    annotated = draw_boxes(img_np, results, confidence)
                    violations = violation_summary(results)

                st.image(annotated, caption="Detection Output", use_container_width=True)

                # update session
                st.session_state.total_processed += 1
                st.session_state.total_violations += len(violations)
                for name, _ in violations:
                    st.session_state.violation_counts[name] += 1

                st.markdown("<br>", unsafe_allow_html=True)

                if violations:
                    st.markdown(f"""
                    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.20);
                                border-radius:14px;padding:16px 20px;margin-bottom:16px;">
                        <div style="color:#FCA5A5;font-size:0.78rem;font-weight:600;
                                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px;">
                            ⚠ {len(violations)} Violation{'s' if len(violations)>1 else ''} Detected
                        </div>
                    """, unsafe_allow_html=True)

                    for name, conf_val in violations:
                        meta = CLASS_META.get(name, {})
                        col_a = meta.get('color', '#EF4444')
                        st.markdown(f"""
                        <div class="vtag" style="background:rgba{tuple(int(col_a.lstrip('#')[i:i+2],16) for i in (0,2,4))}20;
                                                  border-color:{col_a}40;">
                            <div class="vtag-dot" style="background:{col_a};box-shadow:0 0 8px {col_a}80;"></div>
                            <div class="vtag-name" style="color:{col_a};">{meta.get('icon','')} {fmt(name)}</div>
                            <div style="margin-left:auto;color:#94A3B8;font-size:0.75rem;font-weight:600;">
                                {conf_val:.0%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    # confidence bars
                    st.markdown("""
                    <div class="panel-card" style="margin-top:0;">
                        <div style="color:#64748B;font-size:0.75rem;font-weight:600;
                                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px;">
                            Confidence Breakdown
                        </div>
                    """, unsafe_allow_html=True)
                    for name, conf_val in violations:
                        pct = int(conf_val * 100)
                        col_a = CLASS_META.get(name, {}).get('color', '#3B82F6')
                        st.markdown(f"""
                        <div class="conf-row">
                            <div class="conf-label">{fmt(name)}</div>
                            <div class="conf-bar-bg">
                                <div class="conf-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{col_a},{col_a}99);"></div>
                            </div>
                            <div class="conf-pct">{pct}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.markdown("""
                    <div class="vtag vtag-safe">
                        <div class="vtag-dot"></div>
                        <div class="vtag-name">✅ No violations detected</div>
                    </div>
                    """, unsafe_allow_html=True)

        elif not uploaded_file:
            st.markdown("""
            <div style="background:var(--bg-card, #111827);border:1px solid rgba(59,130,246,0.15);
                        border-radius:18px;padding:60px 24px;text-align:center;height:100%;">
                <div style="font-size:3rem;margin-bottom:14px;">🖼️</div>
                <div style="color:#475569;font-size:0.88rem;">
                    Upload an image to see detection results here
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TAB 2 — VIDEO
# ════════════════════════════════════════════════════════════
with tab2:
    col_v1, col_v2 = st.columns([1, 1.4], gap="large")

    with col_v1:
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">Upload Video</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_video = st.file_uploader(
            "Upload video file",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed",
            key="vid_uploader"
        )
        st.markdown("""
        <div style="color:#475569;font-size:0.76rem;text-align:center;margin-top:6px;">
            Supports MP4 · AVI · MOV
        </div>
        """, unsafe_allow_html=True)

        if uploaded_video:
            st.video(uploaded_video)
            run_video = st.button("▶  Start Detection", use_container_width=True, key="vid_run")
        else:
            st.markdown("""
            <div class="upload-zone">
                <span class="uz-icon">🎬</span>
                <div class="uz-title">Upload a surveillance video</div>
                <div class="uz-sub">MP4 · AVI · MOV supported</div>
            </div>
            """, unsafe_allow_html=True)
            run_video = False

    with col_v2:
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">Live Detection Feed</div>
        </div>
        """, unsafe_allow_html=True)

        if uploaded_video and run_video:
            if model is None:
                st.error("⚠️ Model file `best.pt` not found.")
            else:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                tfile.flush()

                cap     = cv2.VideoCapture(tfile.name)
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps     = int(cap.get(cv2.CAP_PROP_FPS)) or 25
                w_v     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h_v     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                output_path = "output_processed.mp4"
                out = cv2.VideoWriter(output_path,
                                      cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (w_v, h_v))

                stframe  = st.empty()
                prog_bar = st.progress(0, text="Processing video…")

                # Stats placeholders
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                frames_ph   = stat_col1.empty()
                viol_ph     = stat_col2.empty()
                fps_ph      = stat_col3.empty()

                all_violations: dict = {}
                frame_idx = 0
                t0 = time.time()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results   = model.predict(frame, conf=confidence, iou=iou_thresh, verbose=False)
                    annotated = draw_boxes(frame, results, confidence)

                    for cls_id, conf_val in zip(results[0].boxes.cls, results[0].boxes.conf):
                        n = CLASS_NAMES[int(cls_id)]
                        if n not in all_violations or conf_val > all_violations[n]:
                            all_violations[n] = float(conf_val)

                    out.write(annotated)
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    stframe.image(rgb_frame, use_container_width=True)

                    frame_idx += 1
                    elapsed = time.time() - t0
                    live_fps = frame_idx / elapsed if elapsed > 0 else 0
                    pct = min(frame_idx / max(total_f, 1), 1.0)
                    prog_bar.progress(pct, text=f"Processing… {int(pct*100)}%")

                    frames_ph.metric("Frames", frame_idx)
                    viol_ph.metric("Violations", len(all_violations))
                    fps_ph.metric("Live FPS", f"{live_fps:.1f}")

                cap.release()
                out.release()
                prog_bar.progress(1.0, text="✅ Processing complete!")

                st.session_state.total_processed += 1
                st.session_state.total_violations += len(all_violations)

                # Results
                st.markdown("<br>", unsafe_allow_html=True)
                if all_violations:
                    st.markdown(f"""
                    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.20);
                                border-radius:14px;padding:16px 20px;">
                        <div style="color:#FCA5A5;font-size:0.78rem;font-weight:600;
                                    text-transform:uppercase;margin-bottom:12px;">
                            ⚠ {len(all_violations)} Violation Type{'s' if len(all_violations)>1 else ''} Detected
                        </div>
                    """, unsafe_allow_html=True)
                    for name, conf_val in sorted(all_violations.items(), key=lambda x: -x[1]):
                        meta  = CLASS_META.get(name, {})
                        col_a = meta.get('color', '#EF4444')
                        st.markdown(f"""
                        <div class="vtag">
                            <div class="vtag-dot" style="background:{col_a};box-shadow:0 0 8px {col_a}80;"></div>
                            <div class="vtag-name" style="color:{col_a};">
                                {meta.get('icon','')} {fmt(name)}
                            </div>
                            <div style="margin-left:auto;color:#94A3B8;font-size:0.75rem;font-weight:600;">
                                {conf_val:.0%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="vtag vtag-safe">
                        <div class="vtag-dot"></div>
                        <div class="vtag-name">✅ No violations detected in video</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Download
                st.markdown("<br>", unsafe_allow_html=True)
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "⬇️  Download Processed Video",
                            f,
                            file_name="traffic_violation_output.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
        else:
            st.markdown("""
            <div style="background:#111827;border:1px solid rgba(59,130,246,0.15);
                        border-radius:18px;padding:60px 24px;text-align:center;">
                <div style="font-size:3rem;margin-bottom:14px;">📹</div>
                <div style="color:#475569;font-size:0.88rem;">
                    Upload a video to stream detection results here
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns([1.2, 1], gap="large")

    with c1:
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">About This Project</div>
        </div>
        <div class="panel-card">
            <div style="color:#F1F5F9;font-size:0.95rem;font-weight:700;margin-bottom:10px;">
                🚦 ML-Powered Traffic Violation Detection
            </div>
            <div style="color:#94A3B8;font-size:0.85rem;line-height:1.7;">
                This system uses <strong style="color:#60A5FA;">YOLOv8</strong> — a state-of-the-art
                real-time object detection model — to automatically identify traffic violations from
                surveillance imagery and video feeds.<br><br>
                Trained on the <strong style="color:#60A5FA;">Indian 2 Wheeler Driving Dataset (I2WDD)</strong>,
                the model detects 7 distinct violation categories with high accuracy, enabling
                scalable, automated traffic enforcement without manual oversight.
            </div>
            <hr style="border-color:rgba(59,130,246,0.12);margin:16px 0;">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:4px;">
                <div style="background:#1a2235;border-radius:10px;padding:12px;">
                    <div style="color:#64748B;font-size:0.72rem;font-weight:600;text-transform:uppercase;">Model</div>
                    <div style="color:#F1F5F9;font-size:0.88rem;font-weight:700;margin-top:2px;">YOLOv8 Nano</div>
                </div>
                <div style="background:#1a2235;border-radius:10px;padding:12px;">
                    <div style="color:#64748B;font-size:0.72rem;font-weight:600;text-transform:uppercase;">Accuracy</div>
                    <div style="color:#F1F5F9;font-size:0.88rem;font-weight:700;margin-top:2px;">92% (mAP@0.5: 87.3%)</div>
                </div>
                <div style="background:#1a2235;border-radius:10px;padding:12px;">
                    <div style="color:#64748B;font-size:0.72rem;font-weight:600;text-transform:uppercase;">Dataset</div>
                    <div style="color:#F1F5F9;font-size:0.88rem;font-weight:700;margin-top:2px;">I2WDD (892 images)</div>
                </div>
                <div style="background:#1a2235;border-radius:10px;padding:12px;">
                    <div style="color:#64748B;font-size:0.72rem;font-weight:600;text-transform:uppercase;">Training</div>
                    <div style="color:#F1F5F9;font-size:0.88rem;font-weight:700;margin-top:2px;">100 Epochs · GPU</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">Team</div>
        </div>
        <div class="panel-card">
        """, unsafe_allow_html=True)
        team = [
            ("Akarsh Yadav",    "1220439023", "Project Lead & ML Engineer"),
            ("Prashant Upreti", "1220439134", "Model Training & Dataset"),
            ("Piyush Bhatt",    "1220439129", "Backend & Integration"),
            ("Sahil Ansari",    "1220439151", "UI/UX & Testing"),
        ]
        for name, roll, role in team:
            initials = ''.join(w[0] for w in name.split()[:2])
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:14px;
                        padding:10px 0;border-bottom:1px solid rgba(59,130,246,0.08);">
                <div style="width:38px;height:38px;border-radius:10px;
                            background:linear-gradient(135deg,#2563EB,#4F46E5);
                            display:flex;align-items:center;justify-content:center;
                            color:white;font-size:0.82rem;font-weight:700;flex-shrink:0;">
                    {initials}
                </div>
                <div>
                    <div style="color:#F1F5F9;font-size:0.87rem;font-weight:600;">{name}</div>
                    <div style="color:#64748B;font-size:0.75rem;">{role} · {roll}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="section-heading">
            <div class="sh-dot"></div>
            <div class="sh-title">Violation Classes</div>
        </div>
        """, unsafe_allow_html=True)

        for name, meta in CLASS_META.items():
            col_a = meta['color']
            st.markdown(f"""
            <div style="background:#111827;border:1px solid {col_a}25;border-left:3px solid {col_a};
                        border-radius:12px;padding:14px 16px;margin-bottom:10px;
                        display:flex;align-items:center;gap:14px;">
                <div style="font-size:1.5rem;flex-shrink:0;">{meta['icon']}</div>
                <div>
                    <div style="color:#F1F5F9;font-size:0.87rem;font-weight:600;">{meta['short']}</div>
                    <div style="color:#475569;font-size:0.73rem;margin-top:2px;">{name}</div>
                </div>
                <div style="margin-left:auto;">
                    <div style="width:10px;height:10px;border-radius:50%;background:{col_a};
                                box-shadow:0 0 6px {col_a}70;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="panel-card">
            <div style="color:#64748B;font-size:0.75rem;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px;">
                Supervisor
            </div>
            <div style="color:#F1F5F9;font-size:0.92rem;font-weight:700;">Dr. Sharda Tiwari</div>
            <div style="color:#64748B;font-size:0.80rem;margin-top:2px;">Assistant Professor</div>
            <div style="color:#64748B;font-size:0.80rem;">Dept. of CSE · BBD University, Lucknow</div>
            <hr style="border-color:rgba(59,130,246,0.12);margin:14px 0;">
            <div style="color:#64748B;font-size:0.78rem;line-height:1.6;">
                B.Tech Final Year Project · 2026<br>
                Computer Science & Engineering<br>
                Specialization: Artificial Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div style="text-align:center;padding:28px 0 8px 0;
            border-top:1px solid rgba(59,130,246,0.10);margin-top:32px;">
    <div style="color:#1e293b;font-size:0.78rem;">
        🚦 Traffic Violation Detection System &nbsp;·&nbsp;
        YOLOv8 &nbsp;·&nbsp; BBD University &nbsp;·&nbsp; 2026
    </div>
</div>
""", unsafe_allow_html=True)