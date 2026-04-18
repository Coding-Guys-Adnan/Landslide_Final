# Landslide Intelligence Platform 🌍

The Landslide Intelligence Platform is a comprehensive, real-time Machine Learning application designed to predict and visualize landslide susceptibility across the Indian subcontinent based on live seismic triggers (USGS/NCS feeds) and antecedent rainfall (IMD/ERA5).

## Architecture Overview
This ecosystem is decoupled into three primary components:
1. **The Machine Learning Poller Engine** (`Live_Landslide_Predictor_19.py`)
2. **The FastAPI Backend Server** (`backend/main.py`)
3. **The Frontend Interactive Dashboard** (`frontend/`)

---

## 1. The Core ML Engine (`Live_Landslide_Predictor_19.py`)

This standalone Python script forms the mathematical backbone of our system. It is responsible for training, caching, and continuously querying live global APIs.

- **Data Harvesting:** Native fetchers interact with `USGS GeoJSON` and `NCS` (National Center for Seismology) endpoints to retrieve live earthquakes. It simultaneously pulls gridded rainfall data.
- **Topographic & Soil Mapping:** Upon a live trigger event, it calculates derived physics constraints, pulling elevations via Copernicus GLO-30 DEM APIs, soil compositions via SoilGrids REST, and NDVI vegetation indexes via Sentinel-2.
- **The Prediction Matrix:** The data is pushed through a Soft-Voting Ensemble model fusing Random Forest (RF), Gradient Boosting (GB), and Logistic Regression (LR) algorithms to output a localized landslide probability.
- **Automated Mail Alerts:** The `send_report_email` pipeline dynamically formats high-risk alerts and generates localized HTML reports seamlessly sent to target inboxes via Google SMTP when a >35% threshold is crossed.

---

## 2. API Backend & Cache Server (`backend/main.py`)

The backend spins up using FastAPI and completely insulates the heavy-lifting daemon thread from the frontend.

- **Daemon Decoupling:** `main.py` imports and spins the ML engine inside an isolated background daemon thread. This allows the backend to endlessly poll global APIs for earthquakes without blocking website visitors.
- **In-Memory Caching:** As predictions execute, they are cached into a threaded `predictions_queue` and locked via `threading.Lock()`.
- **REST Endpoints:** Exposes `/api/data` for the master dataset, `/api/earthquakes`, `/api/rainfall`, `/api/predictions` for live rolling data, and `/api/history` for historical High-Risk event parsing.
- **Subscription Engine:** The `/api/subscribe` route allows users to pass in an email. It hooks dynamically back into our SMTP mailer to fire instantaneous "Welcome" and "Data Context" reports out of the active memory queue.

---

## 3. Frontend Web Dashboard (`frontend/`)

A completely static frontend layer utilizing raw CSS, Vanilla JavaScript, and Leaflet Maps, meaning it compiles instantly with zero framework overhead.

### Key Pages
- `index.html`: The Historical Atlas. Leverages `data.js` to asynchronously ingest the master CSV payload, compiling thousands of data points onto the Leaflet map and providing rich filtering mechanics (by year, region, magnitude).
- `dashboard.html`: The Analytics interface. Implements Chart.js alongside the Leaflet layer, providing deep quantitative analytics into terrain variations, seismic distributions, and risk predictions. It also houses the **Live Risk Prediction** pane for manual input testing against the `/predict` route.
- `live.html`: The Real-Time Radar. Hooks into the fast-paced memory queue, polling the API every 10 seconds. It structurally visualizes new earthquakes and rainfall flags the moment the daemon finishes computing the probabilities. It dynamically renders risk badges, updates debug stats, and allows mapping history queries by fetching from from the new `/api/history` route.

## Execution
1. Install requirements.
2. Ensure you have the `landslide_pipeline_model.pkl` located near the execution directory.
3. Open `backend` and run `python -m uvicorn main:app --port 8001`.
4. Open the frontend by running `python -m http.server 8002` locally!
