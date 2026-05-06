# SMART STORE AI NAVIGATION SYSTEM — v2


import json
import faiss
import heapq
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="SmartNav — AI Store Navigation",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# GLOBAL CSS


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp { background: #0A0C10; font-family: 'DM Sans', sans-serif; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Top bar ── */
.topbar {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 15px 32px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.02);
}
.topbar-logo {
    font-family: 'Space Mono', monospace;
    font-size: 14px; font-weight: 700;
    color: #E8FF47; letter-spacing: .1em;
}
.topbar-logo span { color: rgba(255,255,255,.28); margin: 0 6px; }
.topbar-status {
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; color: rgba(255,255,255,.35);
    font-family: 'Space Mono', monospace;
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #4ADE80; box-shadow: 0 0 7px #4ADE80;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 11px 16px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #E8FF47 !important;
    box-shadow: 0 0 0 3px rgba(232,255,71,.1) !important;
}
.stTextInput > div > div > input::placeholder { color: rgba(255,255,255,.2) !important; }
.stTextInput label { display: none !important; }

/* ── Section label ── */
.slbl {
    font-family: 'Space Mono', monospace;
    font-size: 9px; color: rgba(255,255,255,.3);
    letter-spacing: .15em; text-transform: uppercase;
    margin-bottom: 7px;
}

/* ── Metrics row ── */
.metrics-row { display: grid; grid-template-columns:1fr 1fr 1fr; gap:8px; }
.metric-box {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 10px; padding: 12px 10px; text-align: center;
}
.mval {
    font-family: 'Space Mono', monospace;
    font-size: 19px; font-weight: 700;
    color: #E8FF47; line-height: 1; margin-bottom: 5px;
}
.mlbl { font-size: 10px; color: rgba(255,255,255,.28); text-transform:uppercase; letter-spacing:.07em; }

/* ── Located-in card ── */
.shelf-card {
    background: linear-gradient(135deg,rgba(59,130,246,.09),rgba(255,255,255,.02));
    border: 1px solid rgba(59,130,246,.25);
    border-radius: 12px; padding: 14px 16px;
}
.shelf-name { font-size: 15px; font-weight: 600; color: #60A5FA; margin-bottom: 3px; }
.shelf-meta { font-size: 10px; color: rgba(255,255,255,.3); font-family:'Space Mono',monospace; }

/* ── Product card ── */
.product-card {
    background: rgba(255,255,255,.02);
    border: 1px solid rgba(232,255,71,.12);
    border-radius: 12px; overflow: hidden;
}
.product-inner { display: flex; gap: 14px; padding: 14px; align-items: flex-start; }
.product-img {
    width: 82px; height: 82px; border-radius: 8px;
    object-fit: contain; background: #fff;
    flex-shrink: 0; border: 1px solid rgba(255,255,255,.08);
}
.product-img-placeholder {
    width: 82px; height: 82px; border-radius: 8px;
    background: rgba(255,255,255,.06);
    flex-shrink: 0; display:flex; align-items:center;
    justify-content:center; color:rgba(255,255,255,.2);
    font-size: 26px;
}
.product-body { flex: 1; min-width: 0; }
.product-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }
.product-badge {
    font-family:'Space Mono',monospace; font-size:9px;
    color:#E8FF47; background:rgba(232,255,71,.1);
    padding:3px 8px; border-radius:4px; letter-spacing:.08em;
}
.product-sim { font-family:'Space Mono',monospace; font-size:10px; color:rgba(255,255,255,.3); }
.product-name { font-size: 12px; color:rgba(255,255,255,.78); line-height:1.55; margin-bottom:8px; }
.product-meta-row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.star-row { color:#FBBF24; font-size:11px; }
.rating-count { font-size:10px; color:rgba(255,255,255,.3); }
.product-price { font-size:13px; font-weight:600; color:#4ADE80; }
.product-footer {
    border-top: 1px solid rgba(255,255,255,.06);
    padding: 10px 14px;
    display:flex; gap:10px;
}
.btn-amazon {
    display:inline-flex; align-items:center; gap:6px;
    background: #E8FF47; color:#0A0C10;
    font-family:'DM Sans',sans-serif; font-size:12px; font-weight:600;
    padding: 7px 14px; border-radius:7px;
    text-decoration:none; letter-spacing:.02em;
    transition: opacity .15s;
}
.btn-amazon:hover { opacity:.85; }
.btn-amazon svg { width:13px; height:13px; }
.btn-copy {
    display:inline-flex; align-items:center; gap:6px;
    background: rgba(255,255,255,.06); color:rgba(255,255,255,.6);
    font-family:'DM Sans',sans-serif; font-size:12px;
    padding: 7px 14px; border-radius:7px;
    text-decoration:none; border:1px solid rgba(255,255,255,.1);
}

/* ── Path stepper ── */
.path-stepper { display:flex; flex-direction:column; }
.step-row { display:flex; gap:12px; align-items:flex-start; }
.step-left { display:flex; flex-direction:column; align-items:center; width:28px; flex-shrink:0; }
.step-icon {
    width:28px; height:28px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-family:'Space Mono',monospace; font-size:10px; font-weight:700;
    flex-shrink:0;
}
.step-icon.start  { background:#4ADE80; color:#052e16; }
.step-icon.end    { background:#E8FF47; color:#1a1a00; }
.step-icon.mid    { background:rgba(232,255,71,.12); color:#E8FF47;
                    border:1px solid rgba(232,255,71,.3); }
.step-line { width:1px; flex:1; min-height:16px;
             background:rgba(232,255,71,.18); margin-top:4px; }
.step-content { padding-bottom:14px; flex:1; }
.step-node-id {
    font-family:'Space Mono',monospace; font-size:10px;
    font-weight:700; color:rgba(255,255,255,.55); margin-bottom:2px;
}
.step-label { font-size:13px; color:rgba(255,255,255,.82); font-weight:500; margin-bottom:2px; }
.step-desc { font-size:11px; color:rgba(255,255,255,.35); font-family:'Space Mono',monospace; }

/* Map toolbar */
.map-toolbar {
    display:flex; align-items:center; justify-content:space-between;
    padding:12px 22px; border-bottom:1px solid rgba(255,255,255,.06);
}
.map-tb-title { font-family:'Space Mono',monospace; font-size:10px;
               color:rgba(255,255,255,.28); letter-spacing:.1em; text-transform:uppercase; }
.legend { display:flex; gap:16px; align-items:center; }
.leg { display:flex; align-items:center; gap:5px; font-size:10px; color:rgba(255,255,255,.28); }
.ldot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.lline { width:16px; height:2px; border-radius:1px; flex-shrink:0; }

/* scrollbar */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,.1); border-radius:2px; }
</style>
""", unsafe_allow_html=True)


# LOAD RESOURCES


@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_index():
    return faiss.read_index("products_faiss.index")

@st.cache_data
def load_products():
    products = []
    with open("embeddings_ready.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            products.append(json.loads(line))
    return products

@st.cache_data
def load_store():
    with open("store_layout_navigation.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_product_lookup():
    """Enriched lookup: product_id → img_link, product_link, rating, price"""
    try:
        with open("product_lookup.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

model        = load_model()
index        = load_index()
products     = load_products()
store_data   = load_store()
product_meta = load_product_lookup()

nodes = store_data["path_nodes"]
edges = store_data["edges"]


# BUILD GRAPH


graph = {n["node_id"]: [] for n in nodes}
for edge in edges:
    graph[edge["from"]].append((edge["to"],   edge["distance"]))
    graph[edge["to"]].append(  (edge["from"], edge["distance"]))

node_lookup = {
    n["node_id"]: {"x": n["x"], "y": n["y"], "label": n["label"]}
    for n in nodes
}

SHELF_TO_NODE = {
    "A1":"N037","A2":"N038","A3":"N039","A4":"N040",
    "B1":"N041","B2":"N042","B3":"N043","B4":"N044",
    "C1":"N045","C2":"N046","C3":"N047","C4":"N048",
    "D1":"N049","D2":"N050","D3":"N051","D4":"N052",
}

SHELF_COLORS = {
    "A": "#3B82F6", "B": "#10B981", "C": "#8B5CF6", "D": "#F59E0B",
}

SHELF_LABELS = {
    "A1":"Electronics Cables","A2":"Chargers & Adapters",
    "A3":"Audio Accessories","A4":"Smart Devices",
    "B1":"Computer Peripherals","B2":"Storage & Memory",
    "B3":"Networking Devices","B4":"Laptop Accessories",
    "C1":"Office Printers","C2":"Stationery & Pens",
    "C3":"Office Electronics","C4":"Calculators & Tools",
    "D1":"Cables & Converters","D2":"Power Banks",
    "D3":"Wearables & IoT","D4":"Gaming Devices",
}

# Human-readable step descriptions
STEP_LABELS = {
    "csh-EN":      ("Entrance", "You are here — start of your journey"),
    "bot-EN":      ("Move forward", "Walk straight ahead from entrance"),
    "bot-L":       ("Turn left", "Head towards the left side of the store"),
    "bot-AB":      ("Continue left", "Walk along the bottom corridor"),
    "bot-BC":      ("Center corridor", "Continue through the middle section"),
    "bot-CD":      ("Right corridor", "Walk towards the right section"),
    "bot-DE":      ("Far right", "Continue to the far right corridor"),
    "r23-L":       ("Turn into aisle", "Enter the aisle on the left"),
    "r23-AB":      ("Aisle junction", "Reach the junction between A and B aisles"),
    "r23-BC":      ("Aisle junction", "Reach the junction between B and C aisles"),
    "r23-CD":      ("Aisle junction", "Reach the junction between C and D aisles"),
    "r23-DE":      ("Aisle junction", "Reach the D aisle area"),
    "r23-EN":      ("Center row", "Walk through the center of the store"),
    "r12-L":       ("Move up", "Continue up the left corridor"),
    "r12-AB":      ("Upper junction A-B", "Approaching your section"),
    "r12-BC":      ("Upper junction B-C", "Approaching your section"),
    "r12-CD":      ("Upper junction C-D", "Approaching your section"),
    "r12-DE":      ("Upper right area", "Heading to the upper right"),
    "r12-EN":      ("Upper center", "Moving through upper center"),
    "r01-L":       ("Almost there", "Top of the left corridor"),
    "r01-AB":      ("Almost there", "Top of the A-B junction"),
    "r01-BC":      ("Almost there", "Top of the B-C junction"),
    "r01-CD":      ("Almost there", "Top of the C-D junction"),
    "r01-DE":      ("Almost there", "Top of the D section"),
    "r01-EN":      ("Upper center", "Top center corridor"),
    "approach-A1": ("Shelf A1", "Electronics Cables — look to your right"),
    "approach-A2": ("Shelf A2", "Chargers & Adapters — look to your right"),
    "approach-A3": ("Shelf A3", "Audio Accessories — look to your right"),
    "approach-A4": ("Shelf A4", "Smart Devices — look to your right"),
    "approach-B1": ("Shelf B1", "Computer Peripherals — look to your right"),
    "approach-B2": ("Shelf B2", "Storage & Memory — look to your right"),
    "approach-B3": ("Shelf B3", "Networking Devices — look to your right"),
    "approach-B4": ("Shelf B4", "Laptop Accessories — look to your right"),
    "approach-C1": ("Shelf C1", "Office Printers — look to your right"),
    "approach-C2": ("Shelf C2", "Stationery & Pens — look to your right"),
    "approach-C3": ("Shelf C3", "Office Electronics — look to your right"),
    "approach-C4": ("Shelf C4", "Calculators & Tools — look to your right"),
    "approach-D1": ("Shelf D1", "Cables & Converters — look to your right"),
    "approach-D2": ("Shelf D2", "Power Banks — look to your right"),
    "approach-D3": ("Shelf D3", "Wearables & IoT — look to your right"),
    "approach-D4": ("Shelf D4", "Gaming Devices — look to your right"),
}

# CORE LOGIC

def search_product(query, top_k=1):
    q_emb = model.encode([query])
    scores, indices = index.search(np.array(q_emb).astype("float32"), top_k)
    p = products[indices[0][0]]
    return {
        "product_id": p.get("id", "N/A"),
        "text":       p.get("text", "N/A"),
        "shelf":      p.get("shelf", "N/A"),
        "confidence": p.get("score", 0),
        "similarity": float(scores[0][0]),
    }

def dijkstra(start, end):
    queue, visited = [(0, start, [])], set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == end:
            return cost, path
        for nb, w in graph[node]:
            if nb not in visited:
                heapq.heappush(queue, (cost + w, nb, path))
    return float("inf"), []

def navigate(query):
    product   = search_product(query)
    shelf     = product["shelf"]
    dest_node = SHELF_TO_NODE.get(shelf, "N037")
    cost, path = dijkstra("N036", dest_node)
    coords    = [{"node_id": nid, **node_lookup[nid]} for nid in path]
    meta      = product_meta.get(product["product_id"], {})
    return {
        "product":   product,
        "meta":      meta,
        "shelf":     shelf,
        "dest_node": dest_node,
        "distance":  cost,
        "path":      path,
        "coords":    coords,
    }

# SVG MAP

def build_svg(result):
    W, H, PAD = 860, 580, 30
    CW, CH    = W - 2*PAD, H - 2*PAD
    def mx(x): return PAD + x / 100 * CW
    def my(y): return PAD + y / 100 * CH

    target   = result["shelf"]
    shelves  = {
        "A1":(10.29,10.87,12.94,10.87),"A2":(10.29,26.09,12.94,10.87),
        "A3":(10.29,41.30,12.94,10.87),"A4":(10.29,56.52,12.94,10.87),
        "B1":(29.41,10.87,12.94,10.87),"B2":(29.41,26.09,12.94,10.87),
        "B3":(29.41,41.30,12.94,10.87),"B4":(29.41,56.52,12.94,10.87),
        "C1":(48.53,10.87,12.94,10.87),"C2":(48.53,26.09,12.94,10.87),
        "C3":(48.53,41.30,12.94,10.87),"C4":(48.53,56.52,12.94,10.87),
        "D1":(67.65,10.87,12.94,10.87),"D2":(67.65,26.09,12.94,10.87),
        "D3":(67.65,41.30,12.94,10.87),"D4":(67.65,56.52,12.94,10.87),
    }

    lines = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    lines.append("""<defs>
  <filter id="glow"><feGaussianBlur in="SourceGraphic" stdDeviation="4" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <filter id="sg"><feGaussianBlur in="SourceGraphic" stdDeviation="8" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <marker id="ah" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
    <path d="M0,0 L0,6 L7,3 Z" fill="#E8FF47" opacity=".9"/>
  </marker>
</defs>""")

    lines.append(f'<rect width="{W}" height="{H}" fill="#0A0C10"/>')
    lines.append(f'<rect x="{PAD}" y="{PAD}" width="{CW}" height="{CH}" rx="10" fill="none" stroke="rgba(255,255,255,.05)" stroke-width="1.5"/>')

    # Grid lines
    for i in range(0, 101, 10):
        lines.append(f'<line x1="{mx(i):.1f}" y1="{PAD}" x2="{mx(i):.1f}" y2="{H-PAD}" stroke="rgba(255,255,255,.025)" stroke-width="1"/>')
        lines.append(f'<line x1="{PAD}" y1="{my(i):.1f}" x2="{W-PAD}" y2="{my(i):.1f}" stroke="rgba(255,255,255,.025)" stroke-width="1"/>')

    # Aisle headers
    for letter, cx_pct in {"A":16.76,"B":35.88,"C":55.0,"D":74.12}.items():
        col = SHELF_COLORS[letter]
        lines.append(f'<text x="{mx(cx_pct):.1f}" y="{PAD-8}" text-anchor="middle" '
                     f'font-family="Space Mono,monospace" font-size="10" font-weight="700" '
                     f'fill="{col}" opacity=".65" letter-spacing="3">AISLE {letter}</text>')

    # Shelves
    for sid, (sx, sy, sw, sh) in shelves.items():
        col     = SHELF_COLORS[sid[0]]
        is_dest = sid == target
        rx_ = mx(sx); ry_ = my(sy)
        rw_ = mx(sx+sw)-mx(sx); rh_ = my(sy+sh)-my(sy)
        cx_ = rx_+rw_/2; cy_ = ry_+rh_/2

        if is_dest:
            lines.append(f'<rect x="{rx_-5:.1f}" y="{ry_-5:.1f}" width="{rw_+10:.1f}" height="{rh_+10:.1f}" '
                         f'rx="12" fill="{col}" opacity=".15" filter="url(#glow)"/>')
            lines.append(f'<rect x="{rx_:.1f}" y="{ry_:.1f}" width="{rw_:.1f}" height="{rh_:.1f}" '
                         f'rx="8" fill="{col}" opacity=".2" stroke="{col}" stroke-width="2"/>')
        else:
            lines.append(f'<rect x="{rx_:.1f}" y="{ry_:.1f}" width="{rw_:.1f}" height="{rh_:.1f}" '
                         f'rx="8" fill="{col}" opacity=".05" stroke="{col}" stroke-width="0.5" stroke-opacity=".25"/>')

        fw = "700" if is_dest else "400"
        op = "1"   if is_dest else "0.5"
        lines.append(f'<text x="{cx_:.1f}" y="{cy_-5:.1f}" text-anchor="middle" '
                     f'font-family="Space Mono,monospace" font-size="10" font-weight="{fw}" fill="{col}" opacity="{op}">{sid}</text>')
        short = SHELF_LABELS.get(sid,"").split("&")[0].strip()[:9]
        lines.append(f'<text x="{cx_:.1f}" y="{cy_+8:.1f}" text-anchor="middle" '
                     f'font-family="DM Sans,sans-serif" font-size="7.5" '
                     f'fill="rgba(255,255,255,{"0.65" if is_dest else "0.22"})">{short}</text>')

    # Route
    coords = result["coords"]
    if len(coords) >= 2:
        pts = " ".join(f"{mx(c['x']):.1f},{my(c['y']):.1f}" for c in coords)
        lines.append(f'<polyline points="{pts}" fill="none" stroke="#E8FF47" stroke-width="9" '
                     f'stroke-linecap="round" stroke-linejoin="round" opacity=".08" filter="url(#sg)"/>')
        lines.append(f'<polyline points="{pts}" fill="none" stroke="rgba(232,255,71,.2)" '
                     f'stroke-width="1" stroke-dasharray="4 4"/>')
        lines.append(f'<polyline points="{pts}" fill="none" stroke="#E8FF47" stroke-width="2.5" '
                     f'stroke-linecap="round" stroke-linejoin="round" opacity=".92" marker-end="url(#ah)"/>')
        for i, c in enumerate(coords):
            nx_, ny_ = mx(c["x"]), my(c["y"])
            if i == 0:
                lines.append(f'<circle cx="{nx_:.1f}" cy="{ny_:.1f}" r="8" fill="#4ADE80" opacity=".9" filter="url(#glow)"/>')
                lines.append(f'<circle cx="{nx_:.1f}" cy="{ny_:.1f}" r="4" fill="#fff" opacity=".9"/>')
            elif i == len(coords)-1:
                lines.append(f'<circle cx="{nx_:.1f}" cy="{ny_:.1f}" r="12" fill="#E8FF47" opacity=".12" filter="url(#glow)"/>')
                lines.append(f'<circle cx="{nx_:.1f}" cy="{ny_:.1f}" r="7" fill="#E8FF47" opacity=".95"/>')
                lines.append(f'<circle cx="{nx_:.1f}" cy="{ny_:.1f}" r="3" fill="#0A0C10"/>')
            else:
                lines.append(f'<circle cx="{nx_:.1f}" cy="{ny_:.1f}" r="2.5" fill="#E8FF47" opacity=".4"/>')

    # Entrance / Cashiers
    ex=mx(40); ew=mx(60)-mx(40); ey=my(94)-2
    lines.append(f'<rect x="{ex:.1f}" y="{ey:.1f}" width="{ew:.1f}" height="18" rx="6" '
                 f'fill="#4ADE80" opacity=".09" stroke="#4ADE80" stroke-width="1" stroke-opacity=".35"/>')
    lines.append(f'<text x="{ex+ew/2:.1f}" y="{ey+12:.1f}" text-anchor="middle" '
                 f'font-family="Space Mono,monospace" font-size="8" fill="#4ADE80" opacity=".65" letter-spacing="2">ENTRANCE</text>')
    for cx_pct, lbl in [(5.88,"CASHIER 1"),(88.97,"CASHIER 2")]:
        cbx=mx(cx_pct-5); cbw=mx(cx_pct+5)-mx(cx_pct-5)
        lines.append(f'<rect x="{cbx:.1f}" y="{ey:.1f}" width="{cbw:.1f}" height="18" rx="5" '
                     f'fill="rgba(245,158,11,.07)" stroke="rgba(245,158,11,.25)" stroke-width="1"/>')
        lines.append(f'<text x="{mx(cx_pct):.1f}" y="{ey+12:.1f}" text-anchor="middle" '
                     f'font-family="Space Mono,monospace" font-size="7" fill="rgba(245,158,11,.55)" letter-spacing="1">{lbl}</text>')

    dest = result["coords"][-1] if result["coords"] else {}
    lines.append(f'<text x="{W-PAD-4}" y="{H-8}" text-anchor="end" font-family="Space Mono,monospace" '
                 f'font-size="8" fill="rgba(255,255,255,.15)">DEST ({dest.get("x","?")}, {dest.get("y","?")})</text>')
    lines.append(f'<text x="{PAD+4}" y="{H-8}" font-family="Space Mono,monospace" '
                 f'font-size="8" fill="rgba(255,255,255,.15)">SRC (50.0, 94.13)</text>')
    lines.append("</svg>")
    return "\n".join(lines)


# RENDER
# Top bar
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">SmartNav <span>/</span> AI Store Navigation</div>
  <div class="topbar-status">
    <div class="status-dot"></div>
    SYSTEM ONLINE · 1351 PRODUCTS INDEXED
  </div>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([38, 62])

with left_col:
    st.markdown('<div style="padding:22px 18px;display:flex;flex-direction:column;gap:16px;">', unsafe_allow_html=True)

    # Search
    st.markdown('<div class="slbl">Product Search</div>', unsafe_allow_html=True)
    query  = st.text_input("search", value="smartwatch fitness tracker",
                           placeholder="Search any product…", label_visibility="collapsed")
    result = navigate(query)

    # ── Metrics 
    shelf_color = SHELF_COLORS.get(result["shelf"][0], "#fff") if result["shelf"] else "#fff"
    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-box">
        <div class="mval" style="color:{shelf_color}">{result['shelf']}</div>
        <div class="mlbl">Shelf</div>
      </div>
      <div class="metric-box">
        <div class="mval">{round(result['distance'],1)}</div>
        <div class="mlbl">Distance</div>
      </div>
      <div class="metric-box">
        <div class="mval">{len(result['path'])}</div>
        <div class="mlbl">Hops</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Located In 
    shelf_name = SHELF_LABELS.get(result["shelf"], result["shelf"])
    dest_info  = node_lookup.get(result["dest_node"], {})
    st.markdown(f"""
    <div class="shelf-card">
      <div class="slbl" style="margin-bottom:5px">Located In</div>
      <div class="shelf-name">{shelf_name}</div>
      <div class="shelf-meta">Node {result['dest_node']} · ({dest_info.get('x','')}, {dest_info.get('y','')})</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Product Card 
    meta     = result["meta"]
    pid      = result["product"]["product_id"]
    img_url  = meta.get("img_link", "")
    amz_url  = meta.get("product_link", "")
    rating   = meta.get("rating", "")
    r_count  = meta.get("rating_count", "")
    price    = meta.get("discounted_price", "")
    sim      = round(result["product"]["similarity"], 3)
    pname    = result["product"]["text"][:180]

    # Star rendering
    try:
        stars_val = float(rating)
        full  = int(stars_val)
        empty = 5 - full
        stars_html = "★" * full + "☆" * empty
    except:
        stars_html = "★★★★☆"

    img_block = (f'<img class="product-img" src="{img_url}" onerror="this.style.display=\'none\'">'
                 if img_url else '<div class="product-img-placeholder">📦</div>')

    amz_btn = (f'<a href="{amz_url}" target="_blank" class="btn-amazon">'
               f'<svg viewBox="0 0 20 20" fill="currentColor"><path d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z"/></svg>'
               f'View on Amazon</a>'
               if amz_url else "")

    st.markdown(f"""
    <div class="product-card" style="margin-top:4px">
      <div class="product-inner">
        {img_block}
        <div class="product-body">
          <div class="product-header">
            <div class="product-badge">BEST MATCH</div>
            <div class="product-sim">sim {sim}</div>
          </div>
          <div class="product-name">{pname}{'…' if len(result['product']['text'])>180 else ''}</div>
          <div class="product-meta-row">
            <span class="star-row">{stars_html}</span>
            <span class="rating-count">{rating} ({r_count} reviews)</span>
            <span class="product-price">{price}</span>
          </div>
        </div>
      </div>
      {f'<div class="product-footer">{amz_btn}</div>' if amz_btn else ''}
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation Path Stepper
    st.markdown('<div style="margin-top:6px"><div class="slbl">Navigation Path</div>', unsafe_allow_html=True)
    st.markdown('<div class="path-stepper">', unsafe_allow_html=True)

    for i, c in enumerate(result["coords"]):
        is_start = i == 0
        is_end   = i == len(result["coords"]) - 1
        icon_cls = "start" if is_start else ("end" if is_end else "mid")
        icon_txt = "S" if is_start else ("★" if is_end else str(i))
        step_num = str(i + 1)

        lbl_info = STEP_LABELS.get(c["label"], (c["label"].replace("-", " ").title(), ""))
        main_lbl = lbl_info[0] if isinstance(lbl_info, tuple) else lbl_info
        sub_lbl  = lbl_info[1] if isinstance(lbl_info, tuple) and len(lbl_info) > 1 else ""

        connector = "" if is_end else '<div class="step-line"></div>'

        st.markdown(f"""
        <div class="step-row">
          <div class="step-left">
            <div class="step-icon {icon_cls}">{icon_txt}</div>
            {connector}
          </div>
          <div class="step-content">
            <div class="step-node-id">{c['node_id']}</div>
            <div class="step-label">{main_lbl}</div>
            <div class="step-desc">{sub_lbl if sub_lbl else f"({c['x']}, {c['y']})"}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    # Map toolbar
    st.markdown(f"""
    <div class="map-toolbar">
      <span class="map-tb-title">Floor Plan A · Real-time Navigation</span>
      <div class="legend">
        <div class="leg"><div class="ldot" style="background:#4ADE80"></div>You are here</div>
        <div class="leg"><div class="ldot" style="background:#E8FF47"></div>Destination</div>
        <div class="leg"><div class="lline" style="background:#E8FF47"></div>Route</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    svg = build_svg(result)
    st.components.v1.html(
        f'<style>body{{margin:0;background:#0A0C10}} svg{{width:100%;height:auto;display:block}}</style>{svg}',
        height=620, scrolling=False,
    )