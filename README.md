# FACTGUARD


> Uncover Truths, Empower Decisions, Transform Reality


![last-commit](https://img.shields.io/github/last-commit/vla2oc/FactGuard?style=flat&logo=git&logoColor=white&color=0080ff)
![top-language](https://img.shields.io/github/languages/top/vla2oc/FactGuard?style=flat&color=0080ff)
![lang-count](https://img.shields.io/github/languages/count/vla2oc/FactGuard?style=flat&color=0080ff)


**FactGuard** is a platform for exploring event data on an interactive map with calendar-based date filtering, region filters, and event-type filters. The frontend is **React + Vite**; maps are powered by **Leaflet**; the backend is **Node.js (Express)** with **PostgreSQL**. Core views include point events (strikes), optional frontline lines, heatmaps and trend charts.


---


## ✨ Features
- 🗓️ **Date range** (calendar): events and the map update dynamically.
- 🗺️ **Interactive map** (Leaflet): scalable markers with tooltips/popups, highlighted Ukraine borders/oblasts.
- 🚀 **Strike icons**: custom `DivIcon` markers (SVG/emoji/`react-icons`).
- 🔥 **Heatmap**: kernel density or H3 hex aggregation (`/api/heatmap`).
- 📈 **Trends**: daily/weekly aggregations for charts.
- 🧩 **Pure JS**: no TypeScript; Node.js + SQL.


---


## 🧱 Stack
**Frontend**: React (Vite), TailwindCSS, shadcn/ui, `react-leaflet`, Axios.
**Backend**: Node.js, Express, `pg`, `cors`, `dotenv` (optionally `h3-js` for hex bins).
**DB**: PostgreSQL (recommended). For quick demos you can swap to SQLite.


---




## ⚙️ Getting Started
### Prerequisites
- **Node.js** ≥ 18, **npm** or **pnpm**
- **PostgreSQL** 14+ (local or cloud)


### Installation
```bash
# 1) Clone
git clone https://github.com/vla2oc/FactGuard
cd FactGuard


# 2) Backend deps
cd server
npm i


# 3) Backend env
cp .env.example .env
# edit .env → DATABASE_URL, PORT, CORS_ORIGIN


# 4) Init DB
psql "$DATABASE_URL" -f sql/schema.sql


# 5) (Optional) import Excel
node scripts/import_excel.js scripts/sample.xlsx


# 6) Run API
npm run dev # or: node index.js
# API: http://localhost:5174


# 7) Frontend
cd ../client
npm i
cp .env.example .env
# edit VITE_API_BASE_URL, VITE_TILE_URL
npm run dev
# UI: http://localhost:5173



Code: MIT. Data may be subject to original data-source licences; include attribution and dates when publishing derivatives.
