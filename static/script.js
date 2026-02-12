const API = "";

// Navigation
document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", e => {
        e.preventDefault();
        document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
        document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
        link.classList.add("active");
        const section = link.dataset.section;
        document.getElementById(section).classList.add("active");

        if (section === "dashboard") loadDashboard();
        if (section === "transactions") loadTransactions();
        if (section === "alerts") loadAlerts();
        if (section === "model") loadModelInfo();
    });
});

// Health check
async function checkHealth() {
    const dot = document.getElementById("statusDot");
    const text = document.getElementById("statusText");
    try {
        const res = await fetch(`${API}/api/health`);
        const data = await res.json();
        dot.className = "status-dot online";
        text.textContent = data.model_loaded ? "Model Ready" : "No Model";
    } catch {
        dot.className = "status-dot offline";
        text.textContent = "Offline";
    }
}

// Dashboard
async function loadDashboard() {
    try {
        const [statsRes, txnRes] = await Promise.all([
            fetch(`${API}/api/stats`),
            fetch(`${API}/api/transactions?limit=10`)
        ]);
        const stats = await statsRes.json();
        const txns = await txnRes.json();

        document.getElementById("totalTxns").textContent = stats.total_transactions;
        document.getElementById("fraudTxns").textContent = stats.fraud_transactions;
        document.getElementById("legitTxns").textContent = stats.legit_transactions;
        document.getElementById("fraudRate").textContent = stats.fraud_rate + "%";
        document.getElementById("openAlerts").textContent = stats.open_alerts;
        document.getElementById("avgFraudAmt").textContent = "$" + stats.avg_fraud_amount.toLocaleString();

        renderTxnTable("recentTxnBody", txns, false);
    } catch (err) {
        console.error("Failed to load dashboard:", err);
    }
}

// Transactions
async function loadTransactions() {
    try {
        const res = await fetch(`${API}/api/transactions?limit=100`);
        const txns = await res.json();
        renderTxnTable("allTxnBody", txns, true);
    } catch (err) {
        console.error("Failed to load transactions:", err);
    }
}

function renderTxnTable(bodyId, txns, detailed) {
    const body = document.getElementById(bodyId);
    if (!txns.length) {
        body.innerHTML = `<tr><td colspan="${detailed ? 9 : 7}" style="text-align:center;color:var(--text-secondary);padding:2rem;">No transactions yet. Click "Simulate Transactions" to generate data.</td></tr>`;
        return;
    }
    body.innerHTML = txns.map(t => {
        const riskClass = (t.risk_level || "LOW").toLowerCase();
        const fraudBadge = t.is_fraud ? '<span class="badge badge-fraud">FRAUD</span>' : '<span class="badge badge-legit">LEGIT</span>';
        const time = t.timestamp ? new Date(t.timestamp).toLocaleString() : "-";
        const base = `
            <td>${t.transaction_id || "-"}</td>
            <td>$${(t.amount || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
            <td>${t.merchant_category || "-"}</td>`;
        const extra = detailed ? `<td>${t.transaction_type || "-"}</td><td>${t.distance_from_home || 0} km</td>` : "";
        return `<tr>${base}${extra}
            <td><span class="badge badge-${riskClass}">${t.risk_level || "N/A"}</span></td>
            <td>${t.fraud_probability !== null ? (t.fraud_probability * 100).toFixed(1) + "%" : "-"}</td>
            <td>${fraudBadge}</td>
            <td>${time}</td>
        </tr>`;
    }).join("");
}

// Alerts
async function loadAlerts() {
    try {
        const res = await fetch(`${API}/api/alerts`);
        const alerts = await res.json();
        const body = document.getElementById("alertsBody");
        if (!alerts.length) {
            body.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-secondary);padding:2rem;">No alerts.</td></tr>';
            return;
        }
        body.innerHTML = alerts.map(a => {
            const statusClass = a.status === "open" ? "open" : "resolved";
            const action = a.status === "open"
                ? `<button class="btn btn-success" onclick="resolveAlert('${a.transaction_id}')">Resolve</button>`
                : "-";
            return `<tr>
                <td>${a.transaction_id}</td>
                <td>$${(a.amount || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
                <td>${(a.fraud_probability * 100).toFixed(1)}%</td>
                <td><span class="badge badge-${a.risk_level.toLowerCase()}">${a.risk_level}</span></td>
                <td><span class="badge badge-${statusClass}">${a.status.toUpperCase()}</span></td>
                <td>${new Date(a.timestamp).toLocaleString()}</td>
                <td>${action}</td>
            </tr>`;
        }).join("");
    } catch (err) {
        console.error("Failed to load alerts:", err);
    }
}

async function resolveAlert(id) {
    try {
        await fetch(`${API}/api/alerts/${id}/resolve`, { method: "POST" });
        showToast("Alert resolved", "success");
        loadAlerts();
        loadDashboard();
    } catch (err) {
        showToast("Failed to resolve alert", "error");
    }
}

// Predict
async function predictTransaction(e) {
    e.preventDefault();
    const form = document.getElementById("predictForm");
    const data = Object.fromEntries(new FormData(form));
    // Convert numerics
    for (const key in data) {
        if (key !== "merchant_category" && key !== "transaction_type") {
            data[key] = parseFloat(data[key]);
        }
    }

    try {
        const res = await fetch(`${API}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        const result = await res.json();
        if (result.error) {
            showToast(result.error, "error");
            return;
        }

        const div = document.getElementById("predictionResult");
        div.style.display = "block";
        div.className = `prediction-result ${result.is_fraud ? "fraud" : "legit"}`;
        div.innerHTML = `
            <h3>${result.is_fraud ? "FRAUDULENT TRANSACTION DETECTED" : "LEGITIMATE TRANSACTION"}</h3>
            <p>Transaction ID: <span class="value">${result.transaction_id}</span></p>
            <p>Fraud Probability: <span class="value">${(result.fraud_probability * 100).toFixed(2)}%</span></p>
            <p>Risk Level: <span class="value">${result.risk_level}</span></p>
        `;
        loadDashboard();
    } catch (err) {
        showToast("Prediction failed: " + err.message, "error");
    }
}

// Simulate
async function simulateTransactions() {
    try {
        showToast("Simulating transactions...", "info");
        const res = await fetch(`${API}/api/simulate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ count: 20 })
        });
        const data = await res.json();
        if (data.error) {
            showToast(data.error, "error");
            return;
        }
        showToast(`Generated ${data.total} transactions (${data.fraud_detected} fraud detected)`, "success");
        loadDashboard();
    } catch (err) {
        showToast("Simulation failed", "error");
    }
}

// Model Info
async function loadModelInfo() {
    try {
        const res = await fetch(`${API}/api/model/info`);
        const info = await res.json();
        if (info.error) {
            document.getElementById("modelInfoContent").innerHTML = `<div class="card"><p style="color:var(--text-secondary);">No model info available. Train the model first.</p></div>`;
            return;
        }

        const container = document.getElementById("modelInfoContent");
        container.innerHTML = Object.entries(info.results).map(([name, metrics]) => {
            const isBest = name === info.best_model;
            return `<div class="model-card ${isBest ? "best" : ""}">
                <h3>${name.replace(/_/g, " ")}</h3>
                ${Object.entries(metrics).map(([k, v]) => `
                    <div class="metric-row">
                        <span>${k.charAt(0).toUpperCase() + k.slice(1)}</span>
                        <span>${(v * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric-bar"><div class="metric-bar-fill" style="width:${v * 100}%"></div></div>
                `).join("")}
            </div>`;
        }).join("");
    } catch (err) {
        console.error("Failed to load model info:", err);
    }
}

// Retrain
async function retrainModel() {
    if (!confirm("Retrain the model? This may reload data and update model weights.")) return;
    try {
        showToast("Retraining model...", "info");
        const res = await fetch(`${API}/api/model/retrain`, { method: "POST" });
        const data = await res.json();
        if (data.error) {
            showToast("Retrain failed: " + data.error, "error");
            return;
        }
        showToast(`Retrained! Best model: ${data.best_model}`, "success");
        loadModelInfo();
        checkHealth();
    } catch (err) {
        showToast("Retrain failed", "error");
    }
}

// Toast
function showToast(msg, type) {
    const existing = document.querySelector(".toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3500);
}

// Init
checkHealth();
loadDashboard();
