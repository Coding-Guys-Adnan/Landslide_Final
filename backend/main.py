import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math
import os
import threading
import json
import time
import sqlite3
from datetime import datetime
from contextlib import asynccontextmanager
import builtins
import sys
import io

# Fix Windows cp1252 UnicodeEncodeError for emojis
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Landslide_Labeled_ResearchGrade.csv")

model_path = os.path.join(BASE_DIR, "landslide_pipeline_model.pkl")


data = []
ml_bundle = None
model_loaded = False
startup_err = ""

state_lock = threading.Lock()
predictions_queue = []
last_updated = "Never"
refresh_lock = threading.Lock()

DB_FILE = os.path.join(BASE_DIR, "live_data.db")

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS earthquakes (
                        id TEXT PRIMARY KEY,
                        lat REAL,
                        lon REAL,
                        mag REAL,
                        place TEXT,
                        timestamp TEXT
                     )''')
        c.execute('''CREATE TABLE IF NOT EXISTS rainfall_grids (
                        id TEXT PRIMARY KEY,
                        lat REAL,
                        lon REAL,
                        region TEXT,
                        rain REAL,
                        timestamp TEXT
                     )''')
        conn.commit()

init_db()

class PredictRequest(BaseModel):
    rainfall: float
    slope: float
    elevation: float
    ndvi: float
    soil_moisture: float
    magnitude: float = 0.0
    depth: float = 10.0
    pga: float = 0.0
    mmi: float = 1.0

class SubscribeRequest(BaseModel):
    email: str

SUBSCRIBERS_FILE = os.path.join(BASE_DIR, "subscribers.json")
if not os.path.exists(SUBSCRIBERS_FILE):
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump([], f)

class SoftEnsemble:
    def __init__(self):
        self.models = []

    def predict_proba(self, X):
        probs = np.array([m.predict_proba(X) for m in self.models])
        return probs.mean(axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def run_live_loop():
    global last_updated
    import sys
    parent_dir = os.path.abspath(os.path.join(BASE_DIR, ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        import Live_Landslide_Predictor_19 as lp
        import schedule
        
        # Override PKL path so background thread doesn't crash
        lp.CONFIG["pkl_model_path"] = model_path
        lp.CONFIG[r"C:\Users\User\Downloads\Machine Learning\landslide_pipeline_model.pkl"] = model_path
        
        # Prevent input hangs & configure email
        lp.CONFIG["report_email"] = ""
        builtins.input = lambda prompt="": "mocked"
        
        orig_send_email = lp.send_email_alert
        orig_send_report = lp.send_report_email
        
        def patched_send_email(subject, body):
            try:
                with open(SUBSCRIBERS_FILE, "r") as f:
                    subs = json.load(f)
            except:
                subs = []
            for sub in subs:
                lp.CONFIG["alert_email"] = sub
                orig_send_email(subject, body)
                
        def patched_send_report(cycle_results):
            try:
                with open(SUBSCRIBERS_FILE, "r") as f:
                    subs = json.load(f)
            except:
                subs = []
            for sub in subs:
                lp.CONFIG["report_email"] = sub
                orig_send_report(cycle_results)
                
        lp.send_email_alert = patched_send_email
        lp.send_report_email = patched_send_report

        # Intercept Data Fetching for Endpoints
        orig_fetch_eq = lp.fetch_recent_earthquakes
        def patched_fetch_eq():
            raw_eqs = orig_fetch_eq()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with sqlite3.connect(DB_FILE) as conn:
                c = conn.cursor()
                for eq in raw_eqs:
                    eid = str(eq.get("id", ""))
                    lat = float(eq.get("latitude", 0))
                    lon = float(eq.get("longitude", 0))
                    mag = float(eq.get("mag", 0))
                    place = str(eq.get("place", "Unknown"))
                    c.execute("INSERT OR REPLACE INTO earthquakes (id, lat, lon, mag, place, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                              (eid, lat, lon, mag, place, now_str))
                conn.commit()

            with state_lock:
                global last_updated
                last_updated = now_str
            return raw_eqs
        lp.fetch_recent_earthquakes = patched_fetch_eq

        orig_fetch_rain = lp.fetch_antecedent_rainfall
        def patched_fetch_rain(lat, lon, eq_time):
            res = orig_fetch_rain(lat, lon, eq_time)
            rain_val = float(res.get("rainfall_30d_mm", 0))
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            grid_id = f"{lat}_{lon}"
            
            with sqlite3.connect(DB_FILE) as conn:
                c = conn.cursor()
                c.execute("INSERT OR REPLACE INTO rainfall_grids (id, lat, lon, region, rain, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                          (grid_id, lat, lon, "Grid", rain_val, now_str))
                conn.commit()
                
            with state_lock:
                global last_updated
                last_updated = now_str
            return res
        lp.fetch_antecedent_rainfall = patched_fetch_rain
        
        # Intercept Predictions
        orig_log_pred = lp.log_prediction_to_csv
        def patched_log_pred(eq, prob, risk, shakemap=None, topo=None, rain=None, soil=None, veg=None, landcover=None, hydro=None):
            orig_log_pred(eq, prob, risk, shakemap, topo, rain, soil, veg, landcover, hydro)
            with state_lock:
                predictions_queue.append({
                    "lat": float(eq.get("latitude", 0)),
                    "lon": float(eq.get("longitude", 0)),
                    "place": str(eq.get("place", "Grid Area")),
                    "magnitude": eq.get("mag"),
                    "probability": float(prob * 100) if prob <= 1.0 else float(prob),
                    "risk_level": risk,
                    "trigger_type": "seismic" if eq.get("mag") is not None else "rainfall",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        lp.log_prediction_to_csv = patched_log_pred
        
        # Safe ML Load inside thread
        model = lp.train_or_load_model()
        lp.poll_and_predict(model)
        lp.poll_rainfall_grid(model)
        
        schedule.every(lp.CONFIG["poll_interval_sec"]).seconds.do(lp.poll_and_predict, model)
        schedule.every(lp.CONFIG["rainfall_poll_interval_sec"]).seconds.do(lp.poll_rainfall_grid, model)
        
        while True:
            schedule.run_pending()
            time.sleep(10)
    except Exception as e:
        print(f"[BACKGROUND THREAD ERROR]: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global data, ml_bundle, model_loaded, startup_err
    print("--- STARTUP SEQUENCE INITIATED ---")
    
    # 1. Try to load CSV
    try:
        df = pd.read_csv(file_path)
        dsrc_mapping = {
            'Below all research thresholds': 'None',
            'Combined-Kanungo2009': 'Combined',
            'USGS-Direct': 'USGS',
            'Seismic-Newmark/Jibson2007': 'Seismic',
            'Rainfall-Guzzetti2008': 'Rainfall'
        }
        def safe_val(x):
            if pd.isna(x): return None
            if isinstance(x, float) and math.isnan(x): return None
            return x

        for _, row in df.iterrows():
            item = {
                "year": safe_val(row["Year"]),
                "lat": safe_val(row.iloc[5]),
                "lon": safe_val(row.iloc[6]),
                "place": safe_val(row["Place"]),
                "region": safe_val(row["Region"]),
                "mag": safe_val(row["Magnitude (Mw)"]),
                "depth": safe_val(row["Depth (km)"]),
                "pga": safe_val(row["PGA (g)"]),
                "mmi": safe_val(row["MMI"]),
                "elev": safe_val(row["Elevation (m)"]),
                "slope": safe_val(row.iloc[17]),
                "steep": safe_val(row.iloc[19]),
                "vsteep": safe_val(row.iloc[20]),
                "rain": safe_val(row["Rainfall 30d (mm)"]),
                "prain": safe_val(row["Peak Daily Rain (mm)"]),
                "clay": safe_val(row["Clay (%)"]),
                "sand": safe_val(row["Sand (%)"]),
                "ph": safe_val(row["pH"]),
                "ndvi": safe_val(row["NDVI"]),
                "lc": safe_val(row["Land Cover Class"]),
                "rdist": safe_val(row["River Dist (km)"]),
                "stord": safe_val(row["Stream Order"]),
                "nriver": safe_val(row["Near River"]),
                "locc": 1 if row["Landslide Occurred"] == "YES" else 0 if row["Landslide Occurred"] == "NO" else safe_val(row["Landslide Occurred"]),
                "dsrc": dsrc_mapping.get(row["Detection Source"], safe_val(row["Detection Source"])) if pd.notna(row["Detection Source"]) else None
            }
            # Typecase properly to json primitives
            for k, v in item.items():
                if pd.api.types.is_integer(v):
                    item[k] = int(v)
                elif pd.api.types.is_float(v) and not math.isnan(v):
                    item[k] = float(v)
                    
            data.append(item)
        print(f"[OK] CSV Loaded gracefully. Records: {len(data)}")
    except Exception as e:
        startup_err += f"CSV Error: {str(e)} | "
        print(f"[FAIL] CSV Load Failed: {e}")

    # 2. Try to load PKL Model
    try:
        import __main__
        __main__.SoftEnsemble = SoftEnsemble
        ml_bundle = joblib.load(model_path)
        model_loaded = True
        print(f"[OK] Model Loaded gracefully: {model_path}")
    except Exception as e:
        startup_err += f"PKL Error: {str(e)}"
        print(f"[FAIL] Model Load Failed: {e}")

    # 3. Spin up daemon safely
    if model_loaded:
        threading.Thread(target=run_live_loop, daemon=True).start()
        print("[OK] Background Live Poller Daemon spun up.")
    else:
        print("[WARN] Background polling CANCELLED: Model missing.")

    print("--- STARTUP SEQUENCE COMPLETED ---")
    yield
    print("Application shutdown requested. Closing active threads.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/force_refresh")
def force_refresh():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model unassigned")
        
    def do_refresh():
        if refresh_lock.acquire(blocking=False):
            try:
                import sys
                parent_dir = os.path.abspath(os.path.join(BASE_DIR, ".."))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                import Live_Landslide_Predictor_19 as lp
                
                # Fetch fresh data using patched functions
                model = lp.train_or_load_model()
                lp.poll_and_predict(model)
            except Exception as e:
                print(f"Manual force refresh failed: {e}")
            finally:
                refresh_lock.release()
                
    t = threading.Thread(target=do_refresh)
    t.start()
    t.join(timeout=30.0) # wait up to 30s
    return {"status": "success"}

@app.get("/api/stats")
def health_check():
    eq_count = 0
    rain_count = 0
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        try:
            eq_count = c.execute("SELECT COUNT(*) FROM earthquakes").fetchone()[0]
            rain_count = c.execute("SELECT COUNT(*) FROM rainfall_grids").fetchone()[0]
        except Exception:
            pass

    with state_lock:
        return {
            "model_loaded": model_loaded,
            "errors": startup_err,
            "csv_records": len(data),
            "predictions_queue_count": len(predictions_queue),
            "earthquakes_cached": eq_count,
            "rainfall_grids_cached": rain_count,
            "last_updated": last_updated
        }

@app.get("/api/data")
def get_data():
    return data

@app.get("/api/predictions")
def get_predictions():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="ML Model not loaded on backend. Live data unavailable.")
    
    with state_lock:
        queue_copy = predictions_queue.copy()[-50:] # Restrict to last 50
    
    if queue_copy:
        return queue_copy
        
    # Fallback if queue empty but file exists
    parent_dir = os.path.abspath(os.path.join(BASE_DIR, ".."))
    pred_file = os.path.join(parent_dir, "all_predictions.csv")
    if not os.path.exists(pred_file):
        return []
    try:
        df = pd.read_csv(pred_file)
        df = df.tail(50).fillna("")
        out = []
        for _, row in df.iterrows():
            out.append({
                "lat": float(row.get("latitude", 0)),
                "lon": float(row.get("longitude", 0)),
                "place": str(row.get("place", "Grid")),
                "magnitude": float(row.get("magnitude", 0)) if row.get("magnitude") else None,
                "probability": float(row.get("probability", 0) * 100) if row.get("probability") <= 1.0 else float(row.get("probability", 0)),
                "risk_level": str(row.get("risk", "LOW")),
                "trigger_type": "seismic" if row.get("magnitude") else "rainfall",
                "timestamp": str(row.get("time", ""))
            })
        return out
    except Exception:
        return []

@app.get("/api/earthquakes")
def get_earthquakes():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Service unavailable (Model unassigned)")
    out = []
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        rows = c.execute("SELECT lat, lon, mag, place FROM earthquakes ORDER BY timestamp DESC LIMIT 200").fetchall()
        for r in rows:
            out.append({"lat": r[0], "lon": r[1], "mag": r[2], "place": r[3]})
    return out

@app.get("/api/rainfall")
def get_rainfall():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Service unavailable (Model unassigned)")
    out = []
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        rows = c.execute("SELECT lat, lon, rain, region FROM rainfall_grids ORDER BY timestamp DESC LIMIT 200").fetchall()
        for r in rows:
            out.append({"lat": r[0], "lon": r[1], "rain": r[2], "region": r[3]})
    return out
    
@app.get("/api/database_dump")
def get_database_dump():
    eqs = []
    rains = []
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        try:
            erows = c.execute("SELECT id, lat, lon, mag, place, timestamp FROM earthquakes ORDER BY timestamp DESC LIMIT 500").fetchall()
            for r in erows:
                eqs.append({"id": r[0], "lat": r[1], "lon": r[2], "mag": r[3], "place": r[4], "timestamp": r[5]})
                
            rrows = c.execute("SELECT id, lat, lon, region, rain, timestamp FROM rainfall_grids ORDER BY timestamp DESC LIMIT 500").fetchall()
            for r in rrows:
                rains.append({"id": r[0], "lat": r[1], "lon": r[2], "region": r[3], "rain": r[4], "timestamp": r[5]})
        except Exception as e:
            print(f"Error dumping db: {e}")
            pass
    return {"earthquakes": eqs, "rainfall_grids": rains}

@app.post("/predict")
def predict(payload: PredictRequest):
    if not ml_bundle:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")
    
    rain_30d = payload.rainfall
    slope = payload.slope
    elev = payload.elevation
    ndvi = payload.ndvi
    clay = payload.soil_moisture
    mag = payload.magnitude
    depth = payload.depth
    pga = payload.pga
    mmi = payload.mmi
    
    silt = max(100.0 - clay - 40.0, 5.0)
    pga_x_slope = pga * slope
    seismic_hazard_idx = pga * mag
    soil_instability = (clay / 100.0) * slope
    clay_silt_ratio = clay / max(silt, 1.0)
    
    feature_cols = ml_bundle.get("feature_cols", [])
    
    import numpy as np
    row = {
        "Year": 2024, "Month": 6, "Latitude (°N)": 30.0, "Longitude (°E)": 75.0,
        "Magnitude (Mw)": mag, "Depth (km)": depth, "Azimuthal Gap (°)": 180.0,
        "Min Station Dist": 1.0, "RMS Residual (s)": 0.5, "PGA (g)": pga,
        "MMI": mmi, "Elevation (m)": elev, "Slope (°)": slope,
        "Rain Intensity": 1.0, "Clay (%)": clay, "Sand (%)": 40.0,
        "Silt (%)": silt, "SOC (g/kg)": 12.0, "Bulk Density": 120.0,
        "pH": 6.5, "NDVI": ndvi, "LC Susceptibility": 0.5,
        "Stream Order": 2.0, "Aspect_sin": 0.0, "Aspect_cos": -1.0,
        "Region_Himalayan Belt": 1.0,
        "Land Cover Class_Tree cover": 1.0,
        "log_River Dist (km)": 0.69,
        "log_Rainfall 30d (mm)": float(np.log1p(max(rain_30d, 0.0))),
        "log_Peak Daily Rain (mm)": float(np.log1p(max(rain_30d / 3.0, 0.0))), # Peak daily rain estimated at ~1/3 monthly
        "Rain_x_Slope": rain_30d * slope,
        "PGA_x_Slope": pga_x_slope,
        "Clay_Silt_ratio": clay_silt_ratio,
        "Seismic_Hazard_Idx": seismic_hazard_idx,
        "Soil_Instability": soil_instability,
    }

    if feature_cols:
        X_input = pd.DataFrame([{col: row.get(col, 0.0) for col in feature_cols}], columns=feature_cols)
    else:
        X_input = pd.DataFrame([row])

    threshold = ml_bundle.get("threshold", 0.35)
    ensemble = ml_bundle["models"]["ensemble"]
    prob = float(ensemble.predict_proba(X_input)[0, 1])
    prediction = int(prob >= threshold)
    
    return {
        "prediction": prediction,
        "probability": round(prob * 100.0, 2)
    }

@app.post("/api/subscribe")
def subscribe(req: SubscribeRequest):
    email = req.email
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email format")
    try:
        with open(SUBSCRIBERS_FILE, "r") as f:
            subs = json.load(f)
    except:
        subs = []
    
    is_new = False
    if email not in subs:
        subs.append(email)
        with open(SUBSCRIBERS_FILE, "w") as f:
            json.dump(subs, f)
        is_new = True
        
    if is_new:
        try:
            import sys
            parent_dir = os.path.abspath(os.path.join(BASE_DIR, ".."))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            import Live_Landslide_Predictor_19 as lp
            
            # Send welcome email explicitly
            original_alert = lp.CONFIG["alert_email"]
            original_report = lp.CONFIG["report_email"]
            
            lp.CONFIG["alert_email"] = email
            lp.send_email_alert(
                "Successfully Subscribed to Live Radar",
                "You are now subscribed to automated high-risk landslide alerts.\nWe will dispatch live detection events immediately to this address."
            )
            
            # Dispatch full data report immediately by parsing the queue
            lp.CONFIG["report_email"] = email
            with state_lock:
                cycle = predictions_queue.copy()[-10:] # Take last 10 predictions to build an initial report
            lp.send_report_email(cycle)
            
            # Restore
            lp.CONFIG["alert_email"] = original_alert
            lp.CONFIG["report_email"] = original_report
        except Exception as e:
            print(f"Failed to send welcome email to {email}: {e}")

    return {"status": "subscribed", "email": email}

@app.get("/api/history")
def get_history():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Service unavailable (Model unassigned)")
        
    parent_dir = os.path.abspath(os.path.join(BASE_DIR, ".."))
    pred_file = os.path.join(parent_dir, "all_predictions.csv")
    if not os.path.exists(pred_file):
        return []
    try:
        df = pd.read_csv(pred_file)
        
        # Take latest 50 historical alerts, reverse chronologically, regardless of risk
        df_hist = df.tail(50).iloc[::-1].fillna("")
        
        out = []
        for idx, row in df_hist.iterrows():
            out.append({
                "lat": float(row.get("latitude", 0)),
                "lon": float(row.get("longitude", 0)),
                "place": str(row.get("place", "Grid Area")),
                "magnitude": float(row.get("magnitude", 0)) if row.get("magnitude") else None,
                "probability": float(row.get("probability", 0) * 100) if row.get("probability") <= 1.0 else float(row.get("probability", 0)),
                "risk_level": str(row.get("risk", "HIGH")),
                "trigger_type": "seismic" if row.get("magnitude") else "rainfall",
                "timestamp": str(row.get("time", ""))
            })
        return out
    except Exception as e:
        print(f"Error reading history: {e}")
        return []
