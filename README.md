# 🛒 Smart Store — AI Indoor Navigation System

> An end-to-end AI-powered retail navigation system that understands natural language product queries, semantically matches them to real Amazon products, and calculates the shortest in-store walking route using graph pathfinding — rendered on a live 2D store map.

<br>

## ✦ Demo

```

User types: "wireless bluetooth earphones"
     ↓
AI matches: "boAt Rockerz 255 Pro+ Bluetooth Wireless..."
     ↓
System assigns: Shelf A3 — Audio Accessories
     ↓
Dijkstra calculates: N036 → N030 → N027 → N026 → N039 (7 hops, 82.9 units)
     ↓
Map renders: Glowing route from Entrance to Shelf A3
```
## 📸 Screenshots


| Scientific Calculator | Smartwatch Tracker |
|:--------------------:|:-----------------:|
| ![](calculator%20search.png) | ![](smartwatch%20tracker.png) |
<br>

## 📁 Repository Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application — full UI with search, product card, path stepper, and live SVG map |
| `SmartStore.ipynb` | Jupyter notebook — data pipeline, embedding generation, FAISS indexing, and navigation prototype |
| `amazon.xlsx` | Raw Amazon products dataset — 1,465 products with names, categories, prices, ratings, reviews |
| `embeddings_ready.jsonl` | Text-prepared product records — one JSON per line, ready for `model.encode()` |
| `product_shelf_mapping.json` | Semantic shelf assignments — each product mapped to the correct store shelf with confidence score |
| `search_text_pipeline.json` | Cleaned and enriched `search_text` field for all 1,351 products |
| `products_faiss.index` | FAISS vector index — pre-built L2 index for sub-millisecond similarity search |
| `store_layout_navigation.json` | Store graph — 55 nodes, 112 edges, shelf coordinates, aisle layout, and pre-validated A* example routes |

<br>

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Query                         │
│              "iphone fast charging cable"               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Semantic Search Layer                       │
│                                                         │
│  sentence-transformers (all-MiniLM-L6-v2)               │
│  Query → 384-dim embedding vector                        │
│                                                         │
│  FAISS IndexFlatL2                                       │
│  Vector → Top-K nearest products                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Product → Shelf Mapping                     │
│                                                         │
│  product_shelf_mapping.json                             │
│  Best match product_id → assigned_shelf_id (e.g. "A1")  │
│  Confidence score: 0.0 (weak) → 1.0 (strong)            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Pathfinding Layer                           │
│                                                         │
│  store_layout_navigation.json                           │
│  55 walkable nodes · 112 bidirectional edges            │
│  Shelf → Destination node (e.g. A1 → N037)              │
│                                                         │
│  Dijkstra's Algorithm                                   │
│  N036 (Entrance) → Destination node                     │
│  Returns: total distance + ordered node path            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Streamlit UI (app.py)                       │
│                                                         │
│  Left panel:  Product card (image, rating, price, link) │
│               Step-by-step navigation path              │
│               Shelf location + metrics                   │
│                                                         │
│  Right panel: Live SVG store map                        │
│               Highlighted destination shelf             │
│               Animated glowing route                    │
└─────────────────────────────────────────────────────────┘
```

<br>

## 🧠 ML Pipeline (SmartStore.ipynb)

The notebook covers the full pipeline from raw data to a running search engine:

**Step 1 — Install dependencies**
```python
pip install sentence-transformers faiss-cpu pandas numpy
```

**Step 2 — Load and prepare product text**

Each product gets a unified `search_text` field combining:
```
product_name + amazon_category (expanded) + semantic_category + section_name + about_product[:120]
```
Cleaning pipeline applied: Unicode normalization → ASCII encoding → URL removal → camelCase splitting → pipe expansion → whitespace normalization.

**Step 3 — Generate embeddings**
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True)
# Output: (1351, 384) float32 matrix
```

**Step 4 — Build FAISS index**
```python
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, 'products_faiss.index')
```

**Step 5 — Semantic search**
```python
query_vec = model.encode(["bluetooth earphones"])
scores, indices = index.search(query_vec.astype("float32"), top_k=5)
```

**Step 6 — Load store graph and run Dijkstra**
```python
# Graph built from store_layout_navigation.json
# 55 nodes, 112 edges, all nodes reachable from entrance
cost, path = dijkstra(graph, "N036", "N039")
```

<br>

## 🗂️ Dataset Details

### `amazon.xlsx`
Raw product data scraped from Amazon India.
Data from https://www.kaggle.com/code/mehakiftikhar/amazon-sales-dataset-eda/input

| Field | Description |
|-------|-------------|
| `product_id` | Unique ASIN identifier |
| `product_name` | Full product title |
| `category` | Pipe-separated category breadcrumb |
| `discounted_price` | Sale price |
| `actual_price` | Original price |
| `discount_percentage` | Discount ratio |
| `rating` | Average customer rating (1–5) |
| `rating_count` | Number of reviews |
| `about_product` | Bullet-point product description |
| `img_link` | Product image URL |
| `product_link` | Full Amazon product page URL |

**Stats:** 1,465 products · 8 top-level categories · avg rating 4.1 · avg discount 47%

---

### `product_shelf_mapping.json`
Semantic shelf assignment for every product.

```json
{
  "product_id": "B07JW9H4J1",
  "product_name": "Wayona Nylon Braided USB to Lightning Cable...",
  "amazon_category": "Computers & Accessories",
  "semantic_category": "USBCables",
  "assigned_shelf_id": "A1",
  "assigned_section_name": "Electronics Cables",
  "shelf_x": 10.29,
  "shelf_y": 10.87,
  "shelf_center_x": 16.76,
  "shelf_center_y": 16.30,
  "confidence_score": 0.93
}
```

Classification used 16 regex rule sets with priority ordering. High confidence (≥ 0.85): 909 products. A post-classification correction pass fixed cable vs. charger disambiguation (46 products), remote/TV accessories mislabeled as computer peripherals (31 products), and power banks misclassified as chargers (4 products).

---

### `store_layout_navigation.json`
The full navigation graph.

```
Sections     : metadata · shelves · aisles · corridors · path_nodes · edges · example_routes
Shelves      : 20 (16 product shelves + entrance + 2 cashiers + customer service)
Path nodes   : 55  (36 corridor grid + 16 shelf approach nodes + 3 special)
Edges        : 112 (30 horizontal + 30 vertical + 44 shelf-access + 8 special)
Connectivity : 100% — all 55 nodes reachable from entrance (N036)
Avg degree   : 4.07 connections per node
```

Coordinate system: `(0, 0)` = top-left, `(100, 100)` = bottom-right. To convert to SVG pixels: `x × 8.6`, `y × 5.8`.

Node types:

| Type | Count | Example |
|------|-------|---------|
| Corridor grid | 36 | `N001` (top-L), `N036` (entrance) |
| Shelf approach | 16 | `N037` (A1), `N052` (D4) |
| Special | 3 | `N053` cashier-1, `N055` customer service |

---

### `embeddings_ready.jsonl`
One JSON object per line — direct input to FAISS indexing.

```json
{"id": "B07JW9H4J1", "text": "Wayona Nylon Braided USB to Lightning...", "shelf": "A1", "score": 0.93}
```

**Stats:** 1,351 records · avg `search_text` length 336 chars · min 179 · max 416

<br>

## 🗺️ Store Layout

```
  AISLE A          AISLE B          AISLE C          AISLE D
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ A1       │    │ B1       │    │ C1       │    │ D1       │
│ Cables   │    │ Periph.  │    │ Printers │    │ Converters│
├──────────┤    ├──────────┤    ├──────────┤    ├──────────┤
│ A2       │    │ B2       │    │ C2       │    │ D2       │
│ Chargers │    │ Storage  │    │ Stationery│   │ Power    │
├──────────┤    ├──────────┤    ├──────────┤    ├──────────┤
│ A3       │    │ B3       │    │ C3       │    │ D3       │
│ Audio    │    │ Network  │    │ Office   │    │ Wearables│
├──────────┤    ├──────────┤    ├──────────┤    ├──────────┤
│ A4       │    │ B4       │    │ C4       │    │ D4       │
│ Smart    │    │ Laptop   │    │ Calculat.│    │ Gaming   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘

[CASHIER 1]         [  ENTRANCE / EXIT  ]         [CASHIER 2]
```

**Closest shelf to entrance:** C4 — 3 hops  
**Furthest shelf from entrance:** A1 — 8 hops

<br>

The system will match the product, identify its shelf, and render the shortest walking route on the store map.

<br>

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Semantic search | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Vector indexing | `FAISS` · IndexFlatL2 |
| Pathfinding | Dijkstra's algorithm (custom implementation) |
| Store graph | JSON navigation graph · 55 nodes · 112 edges |
| Map rendering | SVG (dynamically generated, no external libraries) |
| Web UI | Streamlit |
| Data processing | Pandas · NumPy |
| Prototyping | Jupyter Notebook |

<br>

## 📊 Performance

| Metric | Value |
|--------|-------|
| Products indexed | 1,351 |
| Embedding dimensions | 384 |
| FAISS search latency | < 5ms |
| Dijkstra pathfinding | < 1ms |
| Shelf coverage | 15 / 16 shelves |
| Classification accuracy (high confidence) | 67% (≥ 0.85 score) |
| Graph connectivity | 100% (all nodes reachable) |

<br>

## 🔬 Improvements

### Search Quality

The current model `all-MiniLM-L6-v2` is English-only. Switching to a multilingual model adds Arabic support with no other changes required:

```python
# In SmartStore.ipynb and app.py
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

After changing the model, re-run the notebook to regenerate `products_faiss.index`.

### Model Comparison

| Model | Size | Languages | Accuracy |
|-------|------|-----------|----------|
| `all-MiniLM-L6-v2` *(current)* | 80MB | English only | baseline |
| `paraphrase-multilingual-MiniLM-L12-v2` *(recommended)* | 420MB | 50+ languages | +15% |
| `paraphrase-multilingual-mpnet-base-v2` | 1.1GB | 50+ languages | +25% |

### Known Limitations

- 33% of products have a shelf confidence score below 0.85 — shelf assignment may occasionally be incorrect
- `IndexFlatL2` searches all vectors sequentially — sufficient for 1,351 products but will slow down beyond 50,000+
- Search text averages 336 characters which can exceed the model's token limit for some products

<br>

## 📄 License

MIT License — see `LICENSE` for details.

---

<p align="center">Built with Python · sentence-transformers · FAISS · Streamlit · SVG</p>
