// R2 Training Studio - Frontend Logic

let ws = null;
let lossHistory = [];
let stepHistory = [];
let chart = null;
let selectedFilters = [];
let availableFilters = [];
let latestStatus = null;  // Track current training status
let disableAutoFill = false;  // Flag to prevent auto-fill after clearing form

// DOM Elements
const connectionStatus = document.getElementById('connectionStatus');
const connectionPanel = document.getElementById('connectionPanel');
const trainingPanel = document.getElementById('trainingPanel');
const progressPanel = document.getElementById('progressPanel');
const connectBtn = document.getElementById('connectBtn');
const newModelBtn = document.getElementById('newModelBtn');
const startBtn = document.getElementById('startBtn');
const cancelBtn = document.getElementById('cancelBtn');
const exportBtn = document.getElementById('exportBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const hardResetBtn = document.getElementById('hardResetBtn');
const viewModelsBtn = document.getElementById('viewModelsBtn');
const modelsSidebar = document.getElementById('modelsSidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const closeSidebarBtn = document.getElementById('closeSidebarBtn');
const modelSearch = document.getElementById('modelSearch');
const modelsList = document.getElementById('modelsList');
const modelNameInput = document.getElementById('modelName');
const modelNameDropdown = document.getElementById('modelNameDropdown');
const filterSearch = document.getElementById('filterSearch');
const addFilterBtn = document.getElementById('addFilterBtn');
const filterDropdown = document.getElementById('filterDropdown');
const selectedFiltersContainer = document.getElementById('selectedFilters');

// Load server info on page load
(async function() {
    try {
        const response = await fetch('/api/server_info');
        const data = await response.json();
        const serverInfo = document.getElementById('serverInfo');
        if (serverInfo) {
            serverInfo.textContent = `UI Server: ${data.hostname}`;
            serverInfo.title = `This web UI is running on: ${data.hostname}\nAccessing via: ${window.location.host}`;
        }
    } catch (error) {
        console.error('Failed to load server info:', error);
    }
})();

// Entry Filter Dropdown Logic
filterSearch.addEventListener('input', debounce(async (e) => {
    const search = e.target.value.trim();
    await loadAvailableFilters(search);
}, 300));

filterSearch.addEventListener('focus', async () => {
    await loadAvailableFilters(filterSearch.value.trim());
    filterDropdown.classList.add('visible');
});

filterSearch.addEventListener('blur', () => {
    // Delay to allow clicking on dropdown items
    setTimeout(() => filterDropdown.classList.remove('visible'), 200);
});

addFilterBtn.addEventListener('click', () => {
    const pattern = filterSearch.value.trim();
    if (pattern && !selectedFilters.includes(pattern)) {
        addFilter(pattern);
        filterSearch.value = '';
        filterDropdown.classList.remove('visible');
    }
});

filterSearch.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        addFilterBtn.click();
    }
});

async function loadAvailableFilters(search = '') {
    try {
        const response = await fetch(`/api/entry_filters?search=${encodeURIComponent(search)}`);
        const data = await response.json();

        if (data.success) {
            availableFilters = data.filters;
            renderFilterDropdown();
        }
    } catch (error) {
        console.error('Error loading filters:', error);
    }
}

function renderFilterDropdown() {
    filterDropdown.innerHTML = '';

    if (availableFilters.length === 0) {
        filterDropdown.innerHTML = '<div class="filter-option" style="color: var(--text-secondary);">No matches found</div>';
        return;
    }

    availableFilters.forEach(filter => {
        const option = document.createElement('div');
        option.className = 'filter-option';
        if (selectedFilters.includes(filter + '*')) {
            option.classList.add('selected');
        }
        option.textContent = filter + '*';
        option.addEventListener('click', () => {
            const pattern = filter + '*';
            if (!selectedFilters.includes(pattern)) {
                addFilter(pattern);
            }
        });
        filterDropdown.appendChild(option);
    });
}

function addFilter(pattern) {
    selectedFilters.push(pattern);
    renderSelectedFilters();
}

function removeFilter(pattern) {
    selectedFilters = selectedFilters.filter(f => f !== pattern);
    renderSelectedFilters();
}

function renderSelectedFilters() {
    selectedFiltersContainer.innerHTML = '';

    selectedFilters.forEach(filter => {
        const tag = document.createElement('div');
        tag.className = 'filter-tag';
        tag.innerHTML = `
            <span>${filter}</span>
            <span class="filter-tag-remove" onclick="removeFilter('${filter}')">×</span>
        `;
        selectedFiltersContainer.appendChild(tag);
    });
}

function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// Connection
connectBtn.addEventListener('click', async () => {
    const host = document.getElementById('serverHost').value;
    const port = parseInt(document.getElementById('serverPort').value);

    // Show AI loading screen
    setLoadingText('Connecting to Training Server', 'Establishing neural link...');
    showLoadingScreen();
    setConnectionStatus('connecting');
    connectBtn.disabled = true;
    connectBtn.textContent = 'Connecting...';

    // Random duration between 3-5 seconds for premium feel
    const minDuration = 3000 + Math.random() * 2000;
    const startTime = Date.now();

    try {
        const response = await fetch('/api/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ host, port })
        });

        const data = await response.json();

        // Wait for minimum duration to complete
        const elapsed = Date.now() - startTime;
        const remainingTime = Math.max(0, minDuration - elapsed);
        await new Promise(resolve => setTimeout(resolve, remainingTime));

        // Hide loading screen
        hideLoadingScreen();

        if (data.success) {
            setConnectionStatus('connected', `${host}:${port}`);
            connectionPanel.style.display = 'none';
            trainingPanel.style.display = 'block';
            progressPanel.style.display = 'block';
            viewModelsBtn.style.display = 'block';
            hardResetBtn.style.display = 'block';
            disconnectBtn.style.display = 'block';

            // Auto-fill form if training is running
            await loadCurrentTrainingConfig();

            connectWebSocket();
        } else {
            setConnectionStatus('disconnected');
            alert('Connection failed: ' + data.error);
        }
    } catch (error) {
        // Even on error, wait for minimum duration
        const elapsed = Date.now() - startTime;
        const remainingTime = Math.max(0, minDuration - elapsed);
        await new Promise(resolve => setTimeout(resolve, remainingTime));

        hideLoadingScreen();
        setConnectionStatus('disconnected');
        alert('Connection error: ' + error);
    } finally {
        connectBtn.disabled = false;
        connectBtn.textContent = 'Connect';
    }
});

// WebSocket Connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/status`;

    ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
        const status = JSON.parse(event.data);
        updateStatus(status);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
        console.log('WebSocket closed');
        setTimeout(connectWebSocket, 3000); // Reconnect after 3s
    };
}

// Start Training
startBtn.addEventListener('click', async () => {
    const modelNameSuffix = document.getElementById('modelName').value;
    const modelName = 'rectify_' + modelNameSuffix;
    const trainingSteps = parseInt(document.getElementById('trainingSteps').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    const predictionHorizon = parseInt(document.getElementById('predictionHorizon').value);
    const forceRebuild = document.getElementById('forceRebuild').checked;

    if (!modelName || !selectedFilters.length) {
        alert('Please fill in model name and select at least one entry filter');
        return;
    }

    startBtn.disabled = true;
    startBtn.innerHTML = '<span class="btn-icon">⏳</span> Starting...';

    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: modelName,
                training_steps: trainingSteps,
                entry_filters: selectedFilters,
                batch_size: batchSize,
                prediction_horizon: predictionHorizon,
                force_rebuild: forceRebuild
            })
        });

        const data = await response.json();

        if (data.success) {
            // Re-enable auto-fill ONLY after training starts successfully
            disableAutoFill = false;
            console.log('Training started - auto-fill re-enabled');

            lossHistory = [];
            stepHistory = [];
            cancelBtn.disabled = false;
        } else {
            alert('Failed to start training: ' + data.error);
            startBtn.disabled = false;
            startBtn.innerHTML = '<span class="btn-icon">▶</span> Start Training';
        }
    } catch (error) {
        alert('Error: ' + error);
        startBtn.disabled = false;
        startBtn.innerHTML = '<span class="btn-icon">▶</span> Start Training';
    }
});

// Cancel Training
cancelBtn.addEventListener('click', async () => {
    const currentPhase = latestStatus ? latestStatus.phase : 'idle';

    if (!confirm('Are you sure you want to cancel?')) {
        return;
    }

    // Detect if model is compiling (early training steps)
    const currentSteps = latestStatus ? latestStatus.steps_completed : 0;
    const isCompiling = currentPhase === 'training' && currentSteps < 10;
    const isExporting = currentPhase === 'preparing_dataset';

    // Show loading overlay with appropriate message
    let message, maxWaitTime;
    if (isCompiling) {
        message = 'Model was compiling - cancelling may take longer (up to 3 minutes)...';
        maxWaitTime = 180000; // 3 minutes for compilation
    } else if (isExporting) {
        message = 'Dataset export must complete first - this may take a minute...';
        maxWaitTime = 120000; // 2 minutes
    } else {
        message = 'Saving checkpoint and stopping training...';
        maxWaitTime = 120000; // 2 minutes
    }

    setLoadingText('Cancelling', message);
    showLoadingScreen();

    cancelBtn.disabled = true;

    try {
        // Trigger cancel (don't wait for response - it might timeout)
        fetch('/api/cancel', { method: 'POST' }).catch(() => console.log('Cancel request sent'));

        // Poll status until training stops
        const startTime = Date.now();

        while (true) {
            await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second

            // Check if timeout
            if (Date.now() - startTime > maxWaitTime) {
                hideLoadingScreen();
                const timeoutMins = Math.floor(maxWaitTime / 60000);
                alert(`Cancel timeout after ${timeoutMins} minutes. Training may still be stopping on server.`);
                break;
            }

            // Fetch fresh status directly (don't rely on WebSocket)
            try {
                const statusResponse = await fetch('/api/status');
                if (!statusResponse.ok) {
                    console.error('Status fetch failed:', statusResponse.status);
                    continue;
                }

                const statusData = await statusResponse.json();
                console.log('Cancel polling - phase:', statusData.phase, 'connected:', statusData.connected);

                const phase = statusData.phase;
                const isStopped = phase === 'idle' || phase === 'failed' || phase === 'finished';

                if (isStopped) {
                    hideLoadingScreen();
                    console.log('Training cancelled successfully - phase is', phase);
                    break;
                }

                // Update loading message with elapsed time
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                setLoadingText('Cancelling', `${message} (${elapsed}s)`);
            } catch (err) {
                console.error('Error checking status during cancel:', err);
                // Continue polling even if one check fails
            }
        }
    } catch (error) {
        hideLoadingScreen();
        alert('Error: ' + error);
    } finally {
        cancelBtn.disabled = false;
        cancelBtn.innerHTML = '<span class="btn-icon">■</span> Cancel';
    }
});

// Disconnect
disconnectBtn.addEventListener('click', async () => {
    if (!confirm('Disconnect from training server?')) {
        return;
    }

    try {
        // Close WebSocket
        if (ws) {
            ws.close();
            ws = null;
        }

        // Call disconnect API
        await fetch('/api/disconnect', { method: 'POST' });

        // Reset UI
        setConnectionStatus('disconnected');
        connectionPanel.style.display = 'block';
        trainingPanel.style.display = 'none';
        progressPanel.style.display = 'none';
        viewModelsBtn.style.display = 'none';
        hardResetBtn.style.display = 'none';
        disconnectBtn.style.display = 'none';

        // Reset state
        lossHistory = [];
        stepHistory = [];
        selectedFilters = [];
        renderSelectedFilters();
    } catch (error) {
        alert('Disconnect error: ' + error);
    }
});

// Hard Reset
hardResetBtn.addEventListener('click', async () => {
    if (!confirm('Hard Reset will destroy the current trainer and create a fresh connection.\n\nThis will cancel any running training and clear all state.\n\nContinue?')) {
        return;
    }

    // Show loading overlay
    setLoadingText('Hard Reset', 'Destroying trainer and creating fresh connection...');
    showLoadingScreen();

    try {
        // Call hard reset API
        const response = await fetch('/api/hard_reset', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            // Reset all UI state
            lossHistory = [];
            stepHistory = [];
            selectedFilters = [];
            latestStatus = null;
            disableAutoFill = true;  // Block WebSocket from re-filling form

            // Clear form
            document.getElementById('modelName').value = '';
            document.getElementById('trainingSteps').value = '40000';
            document.getElementById('batchSize').value = '32';
            document.getElementById('predictionHorizon').value = '32';
            document.getElementById('forceRebuild').checked = false;
            renderSelectedFilters();

            // Clear stats
            document.getElementById('stepsValue').textContent = '0 / 0';
            document.getElementById('lossValue').textContent = '—';
            document.getElementById('speedValue').textContent = '—';
            document.getElementById('etaValue').textContent = '—';
            document.getElementById('phaseValue').textContent = 'Idle';
            document.getElementById('progressBar').style.width = '0%';
            document.getElementById('chartContainer').style.display = 'none';

            // Reset buttons
            startBtn.disabled = false;
            startBtn.innerHTML = '<span class="btn-icon">▶</span> Start Training';
            cancelBtn.disabled = true;
            exportBtn.disabled = true;
            newModelBtn.disabled = false;

            // Enable form
            setFormReadOnly(false);

            hideLoadingScreen();
            alert('Hard reset successful! Fresh trainer created.');
        } else {
            hideLoadingScreen();
            alert('Hard reset failed: ' + data.error);
        }
    } catch (error) {
        hideLoadingScreen();
        alert('Hard reset error: ' + error);
    }
});

// New Model - Reset to clean slate
newModelBtn.addEventListener('click', async () => {
    if (!confirm('Start fresh with a new model? This will reset the server and clear all state.')) {
        return;
    }

    // Show loading
    setLoadingText('New Model', 'Resetting server state...');
    showLoadingScreen();

    try {
        // Call hard reset API (resets server + clears UI)
        const response = await fetch('/api/hard_reset', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            // Reset all UI state
            lossHistory = [];
            stepHistory = [];
            selectedFilters = [];
            latestStatus = null;
            disableAutoFill = true;

            // Clear form
            document.getElementById('modelName').value = '';
            document.getElementById('trainingSteps').value = '40000';
            document.getElementById('batchSize').value = '32';
            document.getElementById('predictionHorizon').value = '32';
            document.getElementById('forceRebuild').checked = false;
            renderSelectedFilters();

            // Clear stats
            document.getElementById('stepsValue').textContent = '0 / 0';
            document.getElementById('lossValue').textContent = '—';
            document.getElementById('speedValue').textContent = '—';
            document.getElementById('etaValue').textContent = '—';
            document.getElementById('phaseValue').textContent = 'Idle';
            document.getElementById('progressBar').style.width = '0%';
            document.getElementById('chartContainer').style.display = 'none';

            // Reset buttons
            startBtn.disabled = false;
            startBtn.innerHTML = '<span class="btn-icon">▶</span> Start Training';
            cancelBtn.disabled = true;
            exportBtn.disabled = true;
            newModelBtn.disabled = false;

            setFormReadOnly(false);
            hideLoadingScreen();
        } else {
            hideLoadingScreen();
            alert('Reset failed: ' + data.error);
        }
    } catch (error) {
        hideLoadingScreen();
        alert('Error: ' + error);
    }
});

// Export Model
exportBtn.addEventListener('click', async () => {
    exportBtn.disabled = true;
    exportBtn.innerHTML = '<span class="btn-icon">⏳</span> Exporting...';

    setLoadingText('Exporting Model', 'This may take a minute...');
    showLoadingScreen();

    try {
        const response = await fetch('/api/export', { method: 'POST' });
        const data = await response.json();

        hideLoadingScreen();

        if (data.success) {
            alert('Model exported successfully!\nModel ID: ' + data.model_id);
        } else {
            alert('Export failed: ' + data.error);
        }
    } catch (error) {
        hideLoadingScreen();
        alert('Error: ' + error);
    } finally {
        exportBtn.disabled = false;
        exportBtn.innerHTML = '<span class="btn-icon">↗</span> Export Model';
    }
});

// Update Status Display
function updateStatus(status) {
    console.log('Status update received:', {
        phase: status.phase,
        export_processed: status.export_entries_processed,
        export_total: status.export_entries_total,
        steps: status.steps_completed,
        max_steps: status.max_steps
    });

    if (!status.connected) {
        return;
    }

    // Skip all updates if user clicked "New Model" (form is cleared)
    if (disableAutoFill) {
        // Still save latestStatus for cancel button phase check
        latestStatus = status;
        return;
    }

    latestStatus = status;  // Save for export handler
    const phase = status.phase;
    const isTrainingActive = phase !== 'idle' && phase !== 'finished' && phase !== 'failed';

    // Form is read-only if a model is loaded (even if finished)
    const hasLoadedModel = status.model_name && status.model_name.length > 0;
    setFormReadOnly(hasLoadedModel);

    // Auto-fill form if training config is available (unless user cleared it)
    if (!disableAutoFill) {
        if (status.model_name) {
            const modelNameInput = document.getElementById('modelName');
            // Strip "rectify_" prefix if present (input only shows suffix)
            const displayName = status.model_name.startsWith('rectify_')
                ? status.model_name.substring(8)
                : status.model_name;
            if (!modelNameInput.value || modelNameInput.value !== displayName) {
                modelNameInput.value = displayName;
            }
        }

        // Always update entry filters if provided and different from current selection
        if (status.entry_filters && status.entry_filters.length > 0) {
            const currentFiltersStr = JSON.stringify(selectedFilters.sort());
            const newFiltersStr = JSON.stringify([...status.entry_filters].sort());
            if (currentFiltersStr !== newFiltersStr) {
                selectedFilters = [...status.entry_filters];
                renderSelectedFilters();
                console.log('Loaded entry filters:', selectedFilters);
            }
        }

        if (status.batch_size) {
            document.getElementById('batchSize').value = status.batch_size;
        }
        if (status.prediction_horizon) {
            document.getElementById('predictionHorizon').value = status.prediction_horizon;
        }
    }

    // Update phase indicator
    const phaseValue = document.getElementById('phaseValue');
    phaseValue.textContent = formatPhase(phase);
    phaseValue.className = 'phase-value ' + phase;

    // Update progress bar
    const progressBar = document.getElementById('progressBar');
    let progressPct = 0;

    if (phase === 'preparing_dataset') {
        const total = Math.max(status.export_entries_total, 1);
        progressPct = (status.export_entries_processed / total) * 100;
        progressBar.style.background = 'var(--orange)';
    } else if (phase === 'training' || phase === 'finished') {
        const total = Math.max(status.max_steps, 1);
        progressPct = (status.steps_completed / total) * 100;
        progressBar.style.background = phase === 'finished' ? 'var(--green)' : 'var(--blue)';
    }

    progressBar.style.width = progressPct + '%';

    // Update metrics
    if (phase === 'preparing_dataset') {
        document.getElementById('stepsValue').textContent =
            `${status.export_entries_processed.toLocaleString()} / ${status.export_entries_total.toLocaleString()}`;
        document.getElementById('lossValue').textContent = '—';
        document.getElementById('speedValue').textContent = 'Exporting...';
        document.getElementById('etaValue').textContent = '—';
    } else {
        document.getElementById('stepsValue').textContent =
            `${status.steps_completed.toLocaleString()} / ${status.max_steps.toLocaleString()}`;
        document.getElementById('lossValue').textContent =
            status.loss !== null ? status.loss.toFixed(6) : '—';
        document.getElementById('speedValue').textContent =
            status.fps ? `${status.fps.toFixed(1)} steps/s` : '—';

        // Calculate ETA
        if (status.fps && status.fps > 0 && status.max_steps > status.steps_completed) {
            const remaining = status.max_steps - status.steps_completed;
            const etaSeconds = remaining / status.fps;
            document.getElementById('etaValue').textContent = formatETA(etaSeconds);
        } else {
            document.getElementById('etaValue').textContent = '—';
        }

        // Update loss chart
        if (phase === 'training' && status.loss !== null && status.loss > 0) {
            updateLossChart(status.steps_completed, status.loss);
        }
    }

    // Update button states
    if (phase === 'finished') {
        startBtn.disabled = false;
        startBtn.innerHTML = '<span class="btn-icon">▶</span> Resume Training';
        cancelBtn.disabled = true;
        exportBtn.disabled = false;
        newModelBtn.disabled = false;
    } else if (phase === 'training') {
        startBtn.disabled = true;
        // Show "Started" once training actually begins (steps > 0)
        if (status.steps_completed > 0) {
            startBtn.innerHTML = '<span class="btn-icon">✓</span> Started';
        }
        cancelBtn.disabled = false;
        exportBtn.disabled = true;  // Disable during training - user must cancel first
        newModelBtn.disabled = true;  // Disable during training
    } else if (phase === 'preparing_dataset') {
        startBtn.disabled = true;
        cancelBtn.disabled = false;  // Allow cancel during export
        exportBtn.disabled = true;  // Disable export - no checkpoint exists yet
        newModelBtn.disabled = true;  // Disable during dataset export
    } else if (phase === 'idle') {
        startBtn.disabled = false;
        // Show "Resume" if model is already configured, otherwise "Start"
        const hasModel = status.model_name && status.model_name.length > 0;
        startBtn.innerHTML = hasModel
            ? '<span class="btn-icon">▶</span> Resume Training'
            : '<span class="btn-icon">▶</span> Start Training';
        cancelBtn.disabled = true;
        exportBtn.disabled = status.model_name ? false : true;  // Only allow export if model exists
        newModelBtn.disabled = false;
    }
}

// Update Loss Chart (with EMA smoothing)
function updateLossChart(step, loss) {
    // Filter out near-zero values (before training starts)
    if (loss > 1e-10) {
        stepHistory.push(step);
        lossHistory.push(loss);
    }

    // Keep last 500 points
    if (stepHistory.length > 500) {
        stepHistory.shift();
        lossHistory.shift();
    }

    const chartContainer = document.getElementById('chartContainer');
    const canvas = document.getElementById('lossChart');

    if (lossHistory.length < 2) {
        return;
    }

    chartContainer.style.display = 'block';

    // Smooth loss with EMA
    function smoothEMA(values, alpha = 0.3) {
        const smoothed = [values[0]];
        for (let i = 1; i < values.length; i++) {
            smoothed.push(alpha * values[i] + (1 - alpha) * smoothed[i - 1]);
        }
        return smoothed;
    }

    const lossSmoothed = smoothEMA(lossHistory, 0.3);

    // Setup canvas
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = {left: 20, right: 20, top: 40, bottom: 30};

    ctx.clearRect(0, 0, width, height);

    // Log scale for Y-axis
    const minLoss = Math.min(...lossHistory);
    const maxLoss = Math.max(...lossHistory);
    const logMin = Math.log10(Math.max(minLoss, 1e-10));
    const logMax = Math.log10(maxLoss);
    const minStep = Math.min(...stepHistory);
    const maxStep = Math.max(...stepHistory);

    function toLogY(loss) {
        const logLoss = Math.log10(Math.max(loss, 1e-10));
        return height - padding.bottom - ((logLoss - logMin) / (logMax - logMin)) * (height - padding.top - padding.bottom);
    }

    function toX(step) {
        return padding.left + ((step - minStep) / (maxStep - minStep)) * (width - padding.left - padding.right);
    }

    // Draw gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, 'rgba(0, 122, 255, 0.15)');
    gradient.addColorStop(1, 'rgba(0, 122, 255, 0.02)');

    ctx.beginPath();
    ctx.moveTo(toX(stepHistory[0]), height - padding.bottom);
    lossSmoothed.forEach((loss, i) => {
        ctx.lineTo(toX(stepHistory[i]), toLogY(loss));
    });
    ctx.lineTo(toX(stepHistory[stepHistory.length - 1]), height - padding.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw smoothed line
    ctx.beginPath();
    ctx.strokeStyle = '#007aff';
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    lossSmoothed.forEach((loss, i) => {
        const x = toX(stepHistory[i]);
        const y = toLogY(loss);
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();

    // Find max loss for annotation
    const maxIdx = lossHistory.indexOf(maxLoss);
    const maxLossStep = stepHistory[maxIdx];

    // Format loss values
    function formatLoss(x) {
        if (x >= 1) return x.toFixed(2);
        if (x >= 0.01) return x.toFixed(3);
        return x.toFixed(4);
    }

    // Set font
    ctx.font = '11px -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif';
    ctx.fillStyle = '#86868b';

    // Max loss annotation (top)
    const maxText = `${formatLoss(maxLoss)} (max, step ${maxLossStep.toLocaleString()})`;
    const maxX = toX(maxLossStep);
    const maxY = toLogY(lossSmoothed[maxIdx]);
    ctx.fillText(maxText, maxX + 8, maxY - 8);

    // Current loss annotation (with background box)
    const currentLoss = lossHistory[lossHistory.length - 1];
    const currentStep = stepHistory[stepHistory.length - 1];
    const currentX = toX(currentStep);
    const currentY = toLogY(lossSmoothed[lossSmoothed.length - 1]);
    const currentText = ` ${formatLoss(currentLoss)} `;

    // Measure text for background box
    ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif';
    const textMetrics = ctx.measureText(currentText);
    const boxPadding = 6;
    const boxWidth = textMetrics.width + boxPadding * 2;

    // Smart positioning: flip to left if would overflow right edge
    const rightEdge = width - padding.right;
    const wouldOverflow = (currentX + 8 + boxWidth) > rightEdge;
    const boxX = wouldOverflow ? (currentX - boxWidth - 8) : (currentX + 8);
    const boxY = currentY - 8;

    // Draw white background box (rounded rectangle)
    ctx.fillStyle = 'white';
    const bx = boxX - boxPadding;
    const by = boxY - 12;
    const bw = textMetrics.width + boxPadding * 2;
    const bh = 20;
    const radius = 4;

    ctx.beginPath();
    ctx.moveTo(bx + radius, by);
    ctx.lineTo(bx + bw - radius, by);
    ctx.arcTo(bx + bw, by, bx + bw, by + radius, radius);
    ctx.lineTo(bx + bw, by + bh - radius);
    ctx.arcTo(bx + bw, by + bh, bx + bw - radius, by + bh, radius);
    ctx.lineTo(bx + radius, by + bh);
    ctx.arcTo(bx, by + bh, bx, by + bh - radius, radius);
    ctx.lineTo(bx, by + radius);
    ctx.arcTo(bx, by, bx + radius, by, radius);
    ctx.closePath();
    ctx.fill();

    // Draw current loss text
    ctx.fillStyle = '#007aff';
    ctx.fillText(currentText, boxX, boxY);

    // Draw subtle grid (horizontal only)
    ctx.strokeStyle = 'rgba(134, 134, 139, 0.1)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 4; i++) {
        const y = padding.top + (i / 3) * (height - padding.top - padding.bottom);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
    }

    // Draw X-axis labels (steps)
    ctx.fillStyle = '#86868b';
    ctx.font = '10px -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif';
    ctx.textAlign = 'center';
    const numTicks = 5;
    for (let i = 0; i <= numTicks; i++) {
        const step = minStep + (i / numTicks) * (maxStep - minStep);
        const x = toX(step);
        ctx.fillText(Math.round(step).toLocaleString(), x, height - 10);
    }
}

// Helper Functions
function setConnectionStatus(status, server = '') {
    const statusDot = connectionStatus.querySelector('.status-dot');
    const statusText = connectionStatus.querySelector('.status-text');

    statusDot.className = 'status-dot ' + status;

    if (status === 'connected') {
        statusText.textContent = `Connected to ${server}`;
    } else if (status === 'connecting') {
        statusText.textContent = 'Connecting...';
    } else {
        statusText.textContent = 'Not Connected';
    }
}

function formatPhase(phase) {
    const labels = {
        'idle': 'Idle',
        'preparing_dataset': 'Preparing Dataset',
        'training': 'Training',
        'finished': 'Complete',
        'failed': 'Failed'
    };
    return labels[phase] || phase;
}

function formatETA(seconds) {
    if (seconds < 60) {
        return `${Math.floor(seconds)}s`;
    } else if (seconds < 3600) {
        return `${(seconds / 60).toFixed(1)}m`;
    } else {
        return `${(seconds / 3600).toFixed(1)}h`;
    }
}

function showLoadingScreen() {
    const loadingScreen = document.getElementById('loadingScreen');
    loadingScreen.classList.add('active');
}

function hideLoadingScreen() {
    const loadingScreen = document.getElementById('loadingScreen');
    loadingScreen.classList.remove('active');
}

function setLoadingText(title, subtitle = '') {
    const loadingText = document.querySelector('.loading-text');
    const loadingSubtext = document.querySelector('.loading-subtext');
    if (loadingText) loadingText.textContent = title;
    if (loadingSubtext) loadingSubtext.textContent = subtitle;
}

// Models Sidebar
let allModels = [];

viewModelsBtn.addEventListener('click', async () => {
    modelsSidebar.classList.add('active');
    sidebarOverlay.classList.add('active');
    await loadModels();
});

closeSidebarBtn.addEventListener('click', () => {
    modelsSidebar.classList.remove('active');
    sidebarOverlay.classList.remove('active');
});

sidebarOverlay.addEventListener('click', () => {
    modelsSidebar.classList.remove('active');
    sidebarOverlay.classList.remove('active');
});

async function loadModels() {
    modelsList.innerHTML = '<div class="loading-message">Loading models...</div>';

    try {
        const response = await fetch('/api/list_models');
        const data = await response.json();

        if (data.success) {
            allModels = data.models;
            renderModels(allModels);
        } else {
            modelsList.innerHTML = '<div class="loading-message">Error: ' + data.error + '</div>';
        }
    } catch (error) {
        modelsList.innerHTML = '<div class="loading-message">Error loading models: ' + error + '</div>';
    }
}

function renderModels(models) {
    if (models.length === 0) {
        modelsList.innerHTML = '<div class="loading-message">No models found</div>';
        return;
    }

    modelsList.innerHTML = models.map(model => {
        const tagsHtml = model.tags && model.tags.length > 0
            ? '<div class="model-tags">' + model.tags.map(tag => '<span class="model-tag">' + tag + '</span>').join('') + '</div>'
            : '';
        const descHtml = model.description
            ? '<div class="model-description">' + model.description + '</div>'
            : '';

        return '<div class="model-item" onclick="copyModelId(\'' + model.model_id + '\')">' +
            '<div class="model-id">' + model.model_id + '</div>' +
            '<div class="model-meta"><span>' + model.timestamp + '</span></div>' +
            descHtml +
            tagsHtml +
            '</div>';
    }).join('');
}

function copyModelId(modelId) {
    navigator.clipboard.writeText(modelId).then(() => {
        alert('Model ID copied to clipboard!\n' + modelId);
    }).catch(() => {
        prompt('Copy this model ID:', modelId);
    });
}

modelSearch.addEventListener('input', (e) => {
    const search = e.target.value.toLowerCase();
    const filtered = allModels.filter(model =>
        model.model_id.toLowerCase().includes(search) ||
        (model.description && model.description.toLowerCase().includes(search)) ||
        (model.tags && model.tags.some(tag => tag.toLowerCase().includes(search)))
    );
    renderModels(filtered);
});

async function loadCurrentTrainingConfig() {
    // Get current status to check if training is running and get config
    try {
        // The WebSocket will provide status, but we need it now
        // Make a direct API call through the status endpoint
        // For now, we'll rely on the first WebSocket message
        // which will trigger updateStatus and populate the form
    } catch (error) {
        console.error('Failed to load training config:', error);
    }
}

function setFormReadOnly(readonly) {
    // Form inputs
    document.getElementById('modelName').disabled = readonly;
    document.getElementById('trainingSteps').disabled = readonly;
    document.getElementById('batchSize').disabled = readonly;
    document.getElementById('predictionHorizon').disabled = readonly;
    document.getElementById('forceRebuild').disabled = readonly;

    // Filter selection controls
    document.getElementById('filterSearch').disabled = readonly;
    addFilterBtn.disabled = readonly;

    // Remove filter tags functionality when readonly
    const filterTags = document.querySelectorAll('.filter-tag-remove');
    filterTags.forEach(tag => {
        tag.style.pointerEvents = readonly ? 'none' : 'auto';
        tag.style.opacity = readonly ? '0.3' : '0.6';
    });

    // Add visual indicator
    const trainingCard = document.querySelector('.training-card');
    if (readonly) {
        trainingCard.classList.add('readonly');
    } else {
        trainingCard.classList.remove('readonly');
    }
}

// Model Name Autocomplete
const debouncedFetchCheckpointNames = debounce(async (search) => {
    try {
        const response = await fetch('/api/checkpoint_names?search=' + encodeURIComponent(search));
        const data = await response.json();
        if (data.success && data.names.length > 0) {
            modelNameDropdown.innerHTML = data.names.map(name => {
                // Strip "rectify_" prefix for display
                const displayName = name.startsWith('rectify_') ? name.substring(8) : name;
                return '<div class="dropdown-item" onclick="selectCheckpointName(\'' + name + '\')">' + displayName + '</div>';
            }).join('');
            modelNameDropdown.style.display = 'block';
        } else {
            modelNameDropdown.style.display = 'none';
        }
    } catch (error) {
        console.error('Error fetching checkpoint names:', error);
    }
}, 300);

modelNameInput.addEventListener('input', (e) => {
    const search = e.target.value;
    if (search.length > 0) {
        debouncedFetchCheckpointNames(search);
    } else {
        modelNameDropdown.style.display = 'none';
    }
});

modelNameInput.addEventListener('focus', (e) => {
    if (e.target.value.length > 0) {
        debouncedFetchCheckpointNames(e.target.value);
    }
});

modelNameInput.addEventListener('blur', () => {
    setTimeout(() => modelNameDropdown.style.display = 'none', 200);
});

function selectCheckpointName(name) {
    // Strip "rectify_" prefix (input only shows suffix)
    const displayName = name.startsWith('rectify_') ? name.substring(8) : name;
    modelNameInput.value = displayName;
    modelNameDropdown.style.display = 'none';
}

// Theme Toggle (Twitter/X Dark Mode)
(function() {
    const themeToggle = document.getElementById('themeToggle');
    if (!themeToggle) return;

    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    function updateIcon(theme) {
        const icon = themeToggle.querySelector('.theme-icon');
        if (icon) icon.textContent = theme === 'dark' ? '☀️' : '🌙';
    }

    updateIcon(savedTheme);

    themeToggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme') || 'light';
        const newTheme = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateIcon(newTheme);
    });
})();
