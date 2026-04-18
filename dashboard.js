// Base API URL - easily toggled between localhost and prod
const API_BASE = 'https://landslide-final.onrender.com';

Chart.defaults.color = '#dde2f0';
Chart.defaults.font.family = "'Space Mono', monospace";

function setLoader(pct, msg) {
    const lbar = document.getElementById('lbar');
    const lsub = document.getElementById('lsub');
    if (lbar) lbar.style.width = pct + '%';
    if (lsub) lsub.textContent = msg;
}

async function initDashboard() {
    setLoader(30, 'Fetching analytics data...');
    try {
        const res = await fetch(`${API_BASE}/api/data`);
        if (!res.ok) throw new Error('API failed');
        const data = await res.json();
        
        setLoader(80, 'Rendering charts...');
        renderCharts(data);
        
        setLoader(100, 'Done');
        setTimeout(() => {
            const ld = document.getElementById('loader');
            if(ld) {
                ld.style.opacity = '0';
                setTimeout(() => ld.style.display = 'none', 400);
            }
        }, 300);
    } catch (e) {
        setLoader(100, 'Error loading data: ' + e.message);
        console.error(e);
    }
}

function renderCharts(data) {
    // 1. Line Chart: Landslides over time
    // Only count confirmed landslides (locc == 1) for the timeframe
    const lsData = data.filter(d => d.locc === 1 && d.year != null);
    const yrCounts = {};
    lsData.forEach(d => {
        yrCounts[d.year] = (yrCounts[d.year] || 0) + 1;
    });
    const years = Object.keys(yrCounts).sort();
    const yearVals = years.map(y => yrCounts[y]);

    new Chart(document.getElementById('timeChart'), {
        type: 'line',
        data: {
            labels: years,
            datasets: [{
                label: 'Confirmed Landslides',
                data: yearVals,
                borderColor: '#f5a623',
                backgroundColor: 'rgba(245, 166, 35, 0.2)',
                tension: 0.3,
                fill: true,
                pointBackgroundColor: '#ff3d6b'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            },
            scales: {
                y: { beginAtZero: true, grid: { color: '#242a3e' } },
                x: { grid: { display: false } }
            }
        }
    });

    // 2. Pie Chart: Risk/Occurrence Distribution
    const landslideCount = lsData.length;
    const nonLandslideCount = data.filter(d => d.locc === 0).length;
    
    new Chart(document.getElementById('riskChart'), {
        type: 'doughnut',
        data: {
            labels: ['Landslide Occurred', 'No Landslide'],
            datasets: [{
                data: [landslideCount, nonLandslideCount],
                backgroundColor: ['#ff3d6b', '#3a4060'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });

    // 3. Bar Chart: Events by Region
    const regCounts = {};
    data.forEach(d => {
        const r = d.region || 'Unknown';
        regCounts[r] = (regCounts[r] || 0) + 1;
    });
    
    // Sort regions by count
    const regions = Object.keys(regCounts).sort((a,b)=>regCounts[b]-regCounts[a]);
    const rVals = regions.map(r => regCounts[r]);

    new Chart(document.getElementById('regionChart'), {
        type: 'bar',
        data: {
            labels: regions,
            datasets: [{
                label: 'Total Monitored Events',
                data: rVals,
                backgroundColor: '#00cfff',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true, grid: { color: '#242a3e' } },
                x: { grid: { display: false } }
            }
        }
    });
}

// Prediction Form Handler
document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btn = document.getElementById('predictBtn');
    const resDiv = document.getElementById('predictResult');
    
    btn.textContent = 'Evaluating...';
    btn.disabled = true;
    resDiv.style.display = 'none';
    resDiv.className = 'predict-result'; // reset classes

    const payload = {
        rainfall: parseFloat(document.getElementById('pred-rain').value),
        slope: parseFloat(document.getElementById('pred-slope').value),
        elevation: parseFloat(document.getElementById('pred-elev').value),
        ndvi: parseFloat(document.getElementById('pred-ndvi').value),
        soil_moisture: parseFloat(document.getElementById('pred-soil').value),
        magnitude: parseFloat(document.getElementById('pred-mag').value),
        depth: parseFloat(document.getElementById('pred-depth').value),
        pga: parseFloat(document.getElementById('pred-pga').value),
        mmi: parseFloat(document.getElementById('pred-mmi').value),
    };

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Prediction API failed');
        
        const result = await response.json();
        
        resDiv.style.display = 'block';
        if (result.prediction === 1 || result.probability >= 35.0) {
            resDiv.className = 'predict-result result-high';
            resDiv.innerHTML = `⚠️ HIGH RISK<br><span style="font-size:0.8rem;color:#dde2f0;font-family:var(--font-mono)">Probability: ${result.probability}%</span>`;
        } else if (result.probability >= 20.0) {
            resDiv.className = 'predict-result result-moderate';
            resDiv.innerHTML = `🟠 MODERATE RISK<br><span style="font-size:0.8rem;color:#dde2f0;font-family:var(--font-mono)">Probability: ${result.probability}%</span>`;
        } else {
            resDiv.className = 'predict-result result-low';
            resDiv.innerHTML = `✅ LOW RISK<br><span style="font-size:0.8rem;color:#dde2f0;font-family:var(--font-mono)">Probability: ${result.probability}%</span>`;
        }
    } catch (err) {
        resDiv.style.display = 'block';
        resDiv.style.color = '#ff3d6b';
        resDiv.textContent = 'Error: Make sure the local python backend is running.';
        console.error(err);
    } finally {
        btn.textContent = 'Evaluate Risk';
        btn.disabled = false;
    }
});

initDashboard();
