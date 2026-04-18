# 🌍 Landslide Intelligence Platform - Official Documentation

This document serves as a comprehensive breakdown of the entire platform's codebase and a step-by-step tutorial on how to host and deploy the website to the public internet for free using GitHub and Render.com.

---

## 📂 1. Directory Structure & File Breakdown

The project operates under a completely decoupled architecture, separating the heavy Machine Learning background calculations from the website interface.

```text
/RGEM
│
├── Live_Landslide_Predictor_19.py   # 🧠 The Core ML Daemon
│     # This is the mathematical brain of the project. It handles background polling,
│     # downloading data from the USGS/NCS APIs, gathering terrain properties 
│     # (elevation, slope, NDVI, soil), and making risk predictions using the AI model.
│
├── landslide_pipeline_model.pkl     # 🤖 The Machine Learning Model
│     # Trained AI model file loaded into memory by the Python backend.
│
├── Landslide_Labeled_ResearchGrade.csv # 📊 Master Historical Dataset
│     # Contains 15,000+ data points for charting on index.html and dashboard.html.
│
├── backend/
│    ├── main.py                     # ⚙️ The FastAPI Server
│    │     # Handles all network traffic. It starts up the background polling loop 
│    │     # and provides URL endpoints (/api/data, /api/earthquakes) for the frontend.
│    ├── live_data.db                # 🗄️ SQLite Live Database
│    │     # Automatically generated database that permanently saves Earthquakes and 
│    │     # Rainfall grids found by the ML Daemon.
│    └── requirements.txt            # 📦 Python Dependencies
│          # Lists the server dependencies (FastAPI, uvicorn, scikit-learn, etc.).
│
└── frontend/                        # 🎨 Visual Interface (Static Site)
     ├── style.css                   # The unified design aesthetic across all pages.
     │
     ├── index.html & script.js      # 🗺️ Historical Atlas Tab
     │     # Loads the massive CSV data and renders colored map nodes and filters.
     │
     ├── dashboard.html & dashboard.js # 📈 Analytics Tab
     │     # Renders the Chart.js graphs and the Manual Risk Prediction UI block.
     │
     ├── live.html & live.js         # 📡 Live Radar Tab
     │     # Constantly polls the backend every 15 minutes to plot recent earthquakes 
     │     # and features the new manual "Refresh" button.
     │
     └── database.html & database.js # 🗃️ Live Database Log Tab
           # A tabular viewer that looks up the historical records preserved cleanly 
           # in the `live_data.db` SQLite core.
```

---

## 🚀 2. Deployment Guide: GitHub & Render.com

To share this website with everyone, we will host the **Frontend** using **GitHub Pages** (Free) and host the **FastAPI Backend** using **Render.com** (Free).

### Step 1: Push your Code to GitHub
Both Render and GitHub Pages require your code to be on GitHub.
1. Create a free account on [GitHub.com](https://github.com/)
2. Create a new "Public" repository (e.g., `landslide-intelligence`)
3. Open your VS Code terminal in the `e:\RGEM` directory and run:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Landslide Intelligence"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/landslide-intelligence.git
   git push -u origin main
   ```

### Step 2: Host the Python Backend on Render.com
Render will act as your "Always On" server processing the live Python code and Machine Learning predictions.
1. Sign up on [Render.com](https://render.com/) and connect your GitHub account.
2. Click **New +** -> **Web Service**.
3. Select your `landslide-intelligence` GitHub repository.
4. Fill in the deployment details:
   - **Name**: `landslide-backend`
   - **Region**: Select whatever is closest to you.
   - **Branch**: `main`
   - **Root Directory**: `backend` *(This tells Render where `main.py` is)*
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Select the **Free** tier.
5. Click **Create Web Service**. Render will take a few minutes to install the Python libraries and start your server.
6. Once deployed, Render will give you a live URL *(e.g., `https://landslide-backend-xyz.onrender.com`)*. **Copy this URL!**

### Step 3: Link the Frontend to the New Live Backend
Your Javascript files check `localhost` right now, so you need to point them to your new Render.com backend.
1. Open `frontend/script.js`, `frontend/dashboard.js`, `frontend/live.js`, and `frontend/database.js`.
2. Look at the top of the file for the `API_BASE` variable:
   ```javascript
   const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
       ? 'http://127.0.0.1:8001' 
       : 'https://landslide-backend-xyz.onrender.com'; // <--- PASTE YOUR RENDER URL HERE
   ```
3. Save the files and run:
   ```bash
   git add .
   git commit -m "Updated API URL for Render"
   git push
   ```

### Step 4: Host the Website on GitHub Pages
1. Go back to your Repository on GitHub.com.
2. Click the **Settings** tab -> click **Pages** on the left menu.
3. Under **Source**, select `Deploy from a branch`.
4. Under **Branch**, select `main` and then select the `/frontend` folder (if it lets you select a folder, otherwise just select `/root`).
   - *(Note: Since your frontend files are in a folder named `frontend`, it is highly recommended to move `index.html` and its peers to the main root folder if GitHub Pages complains, or alternatively, your website will be available at `https://YOUR_USER.github.io/landslide-intelligence/frontend/index.html`)*
5. Click **Save**. 
6. Wait 2-3 minutes. GitHub will display a live URL linking to your new website!

---
> [!NOTE]  
> **A Note on the SQLite Database for Free Tier Hosting:**
> Because Render.com's FREE tier uses ephemeral disks, your `live_data.db` file might reset when Render automatically spins down the server for inactivity. If you want permanent database persistence for years to come, consider mounting a "Render Disk" (Starter Tier) or switching from SQLite to a free cloud database provider like *Supabase/PostgreSQL* in the future!
