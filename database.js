const API_BASE = 'https://landslide-final.onrender.com';
async function fetchDB() {
    try {
        const res = await fetch(`${API_BASE}/api/database_dump`);
        if (!res.ok) throw new Error('Failed to load db dump');
        const data = await res.json();
        
        renderEqTable(data.earthquakes || []);
        renderRainTable(data.rainfall_grids || []);
    } catch (e) {
        console.error(e);
        document.getElementById('eqBody').innerHTML = '<tr><td colspan="6" style="text-align: center; color: #ff3d6b;">Error loading database. Make sure backend is running.</td></tr>';
        document.getElementById('rainBody').innerHTML = '<tr><td colspan="6" style="text-align: center; color: #ff3d6b;">Error loading database. Make sure backend is running.</td></tr>';
    }
}

function renderEqTable(eqs) {
    const tbody = document.getElementById('eqBody');
    if (eqs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #888;">No earthquake records found in database.</td></tr>';
        return;
    }
    
    tbody.innerHTML = eqs.map(eq => {
        let magClass = '';
        if (eq.mag >= 4.5) magClass = 'mag-high';
        else if (eq.mag >= 3.0) magClass = 'mag-mod';
        return `
        <tr>
            <td style="color: #888; font-size: 0.75rem;">${eq.id}</td>
            <td style="color: #aaa;">${eq.timestamp}</td>
            <td class="${magClass}">M${parseFloat(eq.mag).toFixed(1)}</td>
            <td>${eq.place}</td>
            <td>${parseFloat(eq.lat).toFixed(3)}</td>
            <td>${parseFloat(eq.lon).toFixed(3)}</td>
        </tr>
        `;
    }).join("");
}

function renderRainTable(rains) {
    const tbody = document.getElementById('rainBody');
    if (rains.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #888;">No rainfall records found in database.</td></tr>';
        return;
    }
    
    tbody.innerHTML = rains.map(r => {
        let rainClass = r.rain >= 100 ? 'rain-high' : '';
        return `
        <tr>
            <td style="color: #888; font-size: 0.75rem;">${r.id}</td>
            <td style="color: #aaa;">${r.timestamp}</td>
            <td>${r.region}</td>
            <td class="${rainClass}">${parseFloat(r.rain).toFixed(1)}</td>
            <td>${parseFloat(r.lat).toFixed(3)}</td>
            <td>${parseFloat(r.lon).toFixed(3)}</td>
        </tr>
        `;
    }).join("");
}

// Init
fetchDB();
