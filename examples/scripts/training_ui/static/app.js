// R2 Training Studio - Frontend Logic

let ws = null;
let lossHistory = [];
let stepHistory = [];
let chart = null;
let selectedFilters = [];
let availableFilters = [];
let latestStatus = null;  // Track current training status
let disableAutoFill = false;  // Flag to prevent auto-fill after clearing form

// Progress prediction training state
let ppLossHistory = [];
let ppStepHistory = [];
let ppLatestStatus = null;
let ppDisableAutoFill = false;
let progressSelectedFilters = [];

// DOM Elements
const connectionStatus = document.getElementById('connectionStatus');
const connectionPanel = document.getElementById('connectionPanel');
const trainerSelection = document.getElementById('trainerSelection');
const selectSkillTraining = document.getElementById('selectSkillTraining');
const selectProgressTraining = document.getElementById('selectProgressTraining');
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
            viewModelsBtn.style.display = 'block';
            hardResetBtn.style.display = 'block';
            disconnectBtn.style.display = 'block';

            // Show trainer type selection
            trainerSelection.style.display = 'flex';
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
    // Don't create duplicate connections
    if (ws && ws.readyState === WebSocket.OPEN) return;

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
        ws = null;
        setTimeout(connectWebSocket, 3000); // Reconnect after 3s
    };
}

// Start Training
startBtn.addEventListener('click', async () => {
    const modelNameSuffix = document.getElementById('modelName').value;
    const modelName = 'rectify_skill_' + modelNameSuffix;
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

                // During compilation, enforce minimum wait time even if phase shows stopped
                const elapsedMs = Date.now() - startTime;
                const minWaitDuringCompilation = 30000; // 30 seconds minimum
                if (isCompiling && elapsedMs < minWaitDuringCompilation) {
                    console.log('Compilation mode - waiting minimum 30s before accepting stop signal');
                    const elapsed = Math.floor(elapsedMs / 1000);
                    setLoadingText('Cancelling', `${message} (${elapsed}s)`);
                    continue;
                }

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
        // Close WebSockets
        if (ws) {
            ws.close();
            ws = null;
        }
        if (progressWs) {
            progressWs.close();
            progressWs = null;
        }

        // Call disconnect API
        await fetch('/api/disconnect', { method: 'POST' });

        // Reset UI
        setConnectionStatus('disconnected');
        connectionPanel.style.display = 'block';
        trainerSelection.style.display = 'none';
        trainingPanel.style.display = 'none';
        document.getElementById('progressTrainingPanel').style.display = 'none';
        progressPanel.style.display = 'none';
        document.getElementById('progressPredictionPanel').style.display = 'none';
        viewModelsBtn.style.display = 'none';
        hardResetBtn.style.display = 'none';
        disconnectBtn.style.display = 'none';

        // Reset state for skill training
        lossHistory = [];
        stepHistory = [];
        selectedFilters = [];
        renderSelectedFilters();

        // Reset state for progress prediction
        ppLossHistory = [];
        ppStepHistory = [];
        progressSelectedFilters = [];
        renderProgressFilters();
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
            // Reset all skill training UI state
            lossHistory = [];
            stepHistory = [];
            selectedFilters = [];
            latestStatus = null;
            disableAutoFill = true;  // Block WebSocket from re-filling form

            // Clear skill training form
            document.getElementById('modelName').value = '';
            document.getElementById('trainingSteps').value = '40000';
            document.getElementById('batchSize').value = '32';
            document.getElementById('predictionHorizon').value = '32';
            document.getElementById('forceRebuild').checked = false;
            renderSelectedFilters();

            // Clear skill training stats
            document.getElementById('stepsValue').textContent = '0 / 0';
            document.getElementById('lossValue').textContent = '—';
            document.getElementById('speedValue').textContent = '—';
            document.getElementById('etaValue').textContent = '—';
            document.getElementById('phaseValue').textContent = 'Idle';
            document.getElementById('progressBar').style.width = '0%';
            document.getElementById('chartContainer').style.display = 'none';

            // Clear the skill training chart canvas
            const skillCanvas = document.getElementById('lossChart');
            if (skillCanvas) {
                const skillCtx = skillCanvas.getContext('2d');
                skillCtx.clearRect(0, 0, skillCanvas.width, skillCanvas.height);
            }

            // Reset skill training buttons
            startBtn.disabled = false;
            startBtn.innerHTML = '<span class="btn-icon">▶</span> Start Training';
            cancelBtn.disabled = true;
            exportBtn.disabled = true;
            newModelBtn.disabled = false;

            // Enable skill training form
            setFormReadOnly(false);

            // Reset all progress prediction UI state
            ppLossHistory = [];
            ppStepHistory = [];
            progressSelectedFilters = [];
            ppLatestStatus = null;
            ppDisableAutoFill = true;  // Block WebSocket from re-filling form

            // Clear progress prediction form
            document.getElementById('progressModelName').value = '';
            document.getElementById('progressTrainingSteps').value = '10000';
            document.getElementById('progressBatchSize').value = '32';
            document.getElementById('progressTaskType').value = 'classification';
            document.getElementById('progressCheckpointInterval').value = '1000';
            document.getElementById('progressMaxCheckpoints').value = '10';
            document.getElementById('progressForceRebuild').checked = false;
            renderProgressFilters();

            // Clear progress prediction stats
            document.getElementById('ppStepsValue').textContent = '0 / 0';
            document.getElementById('ppLossValue').textContent = '—';
            document.getElementById('ppAccuracyValue').textContent = '—';
            document.getElementById('ppF1Value').textContent = '—';
            document.getElementById('ppValLossValue').textContent = '—';
            document.getElementById('ppValAccuracyValue').textContent = '—';
            document.getElementById('ppSpeedValue').textContent = '—';
            document.getElementById('ppEtaValue').textContent = '—';
            document.getElementById('ppPhaseValue').textContent = 'Idle';
            document.getElementById('ppProgressBar').style.width = '0%';
            document.getElementById('ppChartContainer').style.display = 'none';

            // Clear the progress prediction chart canvas
            const ppCanvas = document.getElementById('ppLossChart');
            if (ppCanvas) {
                const ctx = ppCanvas.getContext('2d');
                ctx.clearRect(0, 0, ppCanvas.width, ppCanvas.height);
            }

            // Reset progress prediction buttons
            progressStartBtn.disabled = false;
            progressStartBtn.innerHTML = '<span class="btn-icon">▶</span> Start Training';
            progressCancelBtn.disabled = true;
            progressExportBtn.disabled = true;
            progressNewModelBtn.disabled = false;

            // Enable progress prediction form
            setProgressFormReadOnly(false);

            // Reset sidebar charts
            resetSidebarCharts();

            hideLoadingScreen();
            alert('Hard reset successful! Both trainers reset.');
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

            // Reset sidebar charts (skill only)
            sidebarSkillLossHistory = [];
            sidebarSkillStepHistory = [];

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
            // Strip "rectify_skill_" prefix if present (input only shows suffix)
            const displayName = status.model_name.startsWith('rectify_skill_')
                ? status.model_name.substring(14)
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
        gameBtn.style.display = 'none';
    } else if (phase === 'training') {
        startBtn.disabled = true;
        // Show "Started" once training actually begins (steps > 0)
        if (status.steps_completed > 0) {
            startBtn.innerHTML = '<span class="btn-icon">✓</span> Started';
        }
        cancelBtn.disabled = false;
        exportBtn.disabled = true;  // Disable during training - user must cancel first
        newModelBtn.disabled = true;  // Disable during training
        gameBtn.style.display = 'inline-flex';  // Show game during training
    } else if (phase === 'preparing_dataset') {
        startBtn.disabled = true;
        cancelBtn.disabled = false;  // Allow cancel during export
        exportBtn.disabled = true;  // Disable export - no checkpoint exists yet
        newModelBtn.disabled = true;  // Disable during dataset export
        gameBtn.style.display = 'none';
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
        gameBtn.style.display = 'none';
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

// Model Name Autocomplete (filtered to skill models only)
const debouncedFetchCheckpointNames = debounce(async (search) => {
    try {
        const response = await fetch('/api/checkpoint_names?search=' + encodeURIComponent(search) + '&prefix=rectify_skill_');
        const data = await response.json();
        if (data.success && data.names.length > 0) {
            modelNameDropdown.innerHTML = data.names.map(name => {
                // Strip "rectify_skill_" prefix for display
                const displayName = name.startsWith('rectify_skill_') ? name.substring(14) : name;
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
    // Strip "rectify_skill_" prefix (input only shows suffix)
    const displayName = name.startsWith('rectify_skill_') ? name.substring(14) : name;
    modelNameInput.value = displayName;
    modelNameDropdown.style.display = 'none';
}

// Theme Toggle (iOS-style switch)
(function() {
    const themeToggle = document.getElementById('themeToggle');
    if (!themeToggle) return;

    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    // Set initial checkbox state (checked = dark mode)
    themeToggle.checked = savedTheme === 'dark';

    themeToggle.addEventListener('change', () => {
        const newTheme = themeToggle.checked ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });
})();

// Robot Jump Game
const gameBtn = document.getElementById('gameBtn');
const gameModal = document.getElementById('gameModal');
const closeGameBtn = document.getElementById('closeGameBtn');
const gameCanvas = document.getElementById('gameCanvas');
const gameScoreEl = document.getElementById('gameScore');
const ctx = gameCanvas.getContext('2d');

let gameRunning = false;
let gameScore = 0;
let gameSpeed = 4;
let robot = { x: 50, y: 300, width: 40, height: 40, velocityY: 0, jumping: false };
let obstacles = [];
let gameFrame = 0;

gameBtn.addEventListener('click', () => {
    gameModal.classList.add('active');
    startGame();
});

// Progress training game button
document.getElementById('progressGameBtn').addEventListener('click', () => {
    gameModal.classList.add('active');
    startGame();
});

closeGameBtn.addEventListener('click', () => {
    gameModal.classList.remove('active');
    gameRunning = false;
});

gameModal.addEventListener('click', (e) => {
    if (e.target === gameModal) {
        gameModal.classList.remove('active');
        gameRunning = false;
    }
});

document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && gameRunning && !robot.jumping) {
        e.preventDefault();
        robot.velocityY = -12;
        robot.jumping = true;
    }
});

function startGame() {
    gameRunning = true;
    gameScore = 0;
    gameSpeed = 4;
    robot = { x: 50, y: 300, width: 40, height: 40, velocityY: 0, jumping: false };
    obstacles = [];
    gameFrame = 0;
    gameLoop();
}

function gameLoop() {
    if (!gameRunning) return;

    ctx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);

    // Update robot
    robot.velocityY += 0.6; // Gravity
    robot.y += robot.velocityY;

    // Ground collision
    if (robot.y > 300) {
        robot.y = 300;
        robot.velocityY = 0;
        robot.jumping = false;
    }

    // Draw robot
    drawRobot(robot.x, robot.y, robot.width, robot.height);

    // Spawn obstacles
    if (gameFrame % 100 === 0) {
        obstacles.push({ x: 800, y: 310, width: 30, height: 30 });
    }

    // Update and draw obstacles
    for (let i = obstacles.length - 1; i >= 0; i--) {
        const obs = obstacles[i];
        obs.x -= gameSpeed;

        drawObstacle(obs.x, obs.y, obs.width, obs.height);

        // Remove off-screen obstacles
        if (obs.x + obs.width < 0) {
            obstacles.splice(i, 1);
            gameScore++;
            gameScoreEl.textContent = gameScore;

            // Increase speed every 5 points
            if (gameScore % 5 === 0) {
                gameSpeed += 0.5;
            }
        }

        // Collision detection
        if (
            robot.x < obs.x + obs.width &&
            robot.x + robot.width > obs.x &&
            robot.y < obs.y + obs.height &&
            robot.y + robot.height > obs.y
        ) {
            gameRunning = false;
            ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
            ctx.fillRect(0, 0, gameCanvas.width, gameCanvas.height);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 48px -apple-system';
            ctx.textAlign = 'center';
            ctx.fillText('Game Over!', 400, 180);
            ctx.font = '24px -apple-system';
            ctx.fillText('Score: ' + gameScore, 400, 230);
            ctx.fillText('Press SPACE to restart', 400, 270);

            document.addEventListener('keydown', function restartHandler(e) {
                if (e.code === 'Space') {
                    document.removeEventListener('keydown', restartHandler);
                    startGame();
                }
            });
            return;
        }
    }

    // Draw ground
    ctx.fillStyle = '#86868b';
    ctx.fillRect(0, 340, 800, 2);

    gameFrame++;
    requestAnimationFrame(gameLoop);
}
// Draw robot (cute robot with head, body, arms, legs)
function drawRobot(x, y, width, height) {
    // Body
    ctx.fillStyle = '#007aff';
    ctx.fillRect(x + 8, y + 15, 24, 20);

    // Head
    ctx.fillStyle = '#0051d5';
    ctx.fillRect(x + 10, y, 20, 15);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(x + 13, y + 4, 5, 5); // Eye
    ctx.fillRect(x + 22, y + 4, 5, 5); // Eye

    // Arms
    ctx.fillStyle = '#007aff';
    ctx.fillRect(x + 2, y + 18, 6, 12); // Left arm
    ctx.fillRect(x + 32, y + 18, 6, 12); // Right arm

    // Legs
    ctx.fillRect(x + 12, y + 35, 6, 5); // Left leg
    ctx.fillRect(x + 22, y + 35, 6, 5); // Right leg

    // Antenna
    ctx.strokeStyle = '#0051d5';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x + 20, y);
    ctx.lineTo(x + 20, y - 5);
    ctx.stroke();
    ctx.fillStyle = '#ff3b30';
    ctx.beginPath();
    ctx.arc(x + 20, y - 5, 2, 0, Math.PI * 2);
    ctx.fill();
}

// Draw obstacle (warning box with stripes)
function drawObstacle(x, y, width, height) {
    // Box
    ctx.fillStyle = '#ff3b30';
    ctx.fillRect(x, y, width, height);

    // Warning stripes
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    for (let i = 0; i < 3; i++) {
        ctx.beginPath();
        ctx.moveTo(x + i * 10, y);
        ctx.lineTo(x + i * 10 + 10, y + height);
        ctx.stroke();
    }

    // Border
    ctx.strokeStyle = '#cc0000';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
}

// Trainer Type Selection
selectSkillTraining.addEventListener('click', async () => {
    trainerSelection.style.display = 'none';
    trainingPanel.style.display = 'block';
    progressPanel.style.display = 'block';

    // Auto-fill form if training is running
    await loadCurrentTrainingConfig();
    connectWebSocket();
});

selectProgressTraining.addEventListener('click', async () => {
    trainerSelection.style.display = 'none';
    document.getElementById('progressTrainingPanel').style.display = 'block';
    document.getElementById('progressPredictionPanel').style.display = 'block';

    // Connect WebSocket for progress training
    connectProgressWebSocket();
});

// Back Buttons
const backFromSkillBtn = document.getElementById('backFromSkillBtn');
const backFromProgressBtn = document.getElementById('backFromProgressBtn');

backFromSkillBtn.addEventListener('click', () => {
    trainingPanel.style.display = 'none';
    progressPanel.style.display = 'none';
    if (ws) {
        ws.close();
        ws = null;
    }
    trainerSelection.style.display = 'flex';
});

backFromProgressBtn.addEventListener('click', () => {
    document.getElementById('progressTrainingPanel').style.display = 'none';
    document.getElementById('progressPredictionPanel').style.display = 'none';
    if (progressWs) {
        progressWs.close();
        progressWs = null;
    }
    trainerSelection.style.display = 'flex';
});

// Progress Prediction Training
const progressStartBtn = document.getElementById('progressStartBtn');
const progressCancelBtn = document.getElementById('progressCancelBtn');
const progressExportBtn = document.getElementById('progressExportBtn');
const progressNewModelBtn = document.getElementById('progressNewModelBtn');

progressStartBtn.addEventListener('click', async () => {
    const modelNameSuffix = document.getElementById('progressModelName').value;
    const modelName = 'rectify_progress_' + modelNameSuffix;
    const trainingSteps = parseInt(document.getElementById('progressTrainingSteps').value);
    const batchSize = parseInt(document.getElementById('progressBatchSize').value);
    const taskType = document.getElementById('progressTaskType').value;
    const checkpointInterval = parseInt(document.getElementById('progressCheckpointInterval').value);
    const maxCheckpoints = parseInt(document.getElementById('progressMaxCheckpoints').value);
    const forceRebuild = document.getElementById('progressForceRebuild').checked;

    // Hardcode cameras to wrist and right
    const cameras = ["wrist_camera", "right_camera"];

    if (!modelName || progressSelectedFilters.length === 0) {
        alert('Please fill in model name and select at least one entry filter');
        return;
    }

    progressStartBtn.disabled = true;
    progressStartBtn.innerHTML = '<span class="btn-icon">⏳</span> Starting...';

    // Clear chart history
    ppLossHistory = [];
    ppStepHistory = [];

    try {
        const response = await fetch('/api/progress/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: modelName,
                training_steps: trainingSteps,
                batch_size: batchSize,
                task_type: taskType,
                cameras: cameras,
                entry_filters: progressSelectedFilters,
                force_rebuild: forceRebuild,
                checkpoint_interval_steps: checkpointInterval,
                max_checkpoints_to_keep: maxCheckpoints,
            })
        });

        const data = await response.json();

        if (data.success) {
            progressCancelBtn.disabled = false;
            ppDisableAutoFill = false;
        } else {
            alert('Failed to start training: ' + data.error);
            progressStartBtn.disabled = false;
            progressStartBtn.innerHTML = '<span class="btn-icon">▶</span> Start Training';
        }
    } catch (error) {
        alert('Error: ' + error);
        progressStartBtn.disabled = false;
        progressStartBtn.innerHTML = '<span class="btn-icon">▶</span> Start Training';
    }
});

// Progress Prediction Entry Filters
const progressFilterSearch = document.getElementById('progressFilterSearch');
const progressAddFilterBtn = document.getElementById('progressAddFilterBtn');
const progressFilterDropdown = document.getElementById('progressFilterDropdown');
const progressSelectedFiltersContainer = document.getElementById('progressSelectedFilters');
const progressModelNameInput = document.getElementById('progressModelName');
const progressModelNameDropdown = document.getElementById('progressModelNameDropdown');

// Reuse the same filter search logic as skill training
const debouncedProgressFilterSearch = debounce(async (search = '') => {
    try {
        const response = await fetch('/api/entry_filters?search=' + encodeURIComponent(search));
        const data = await response.json();

        if (data.success && data.filters.length > 0) {
            progressFilterDropdown.innerHTML = data.filters.map(filter => {
                const isSelected = progressSelectedFilters.includes(filter + '*');
                return `<div class="filter-option${isSelected ? ' selected' : ''}" onclick="addProgressFilter('${filter}')">${filter}${isSelected ? ' ✓' : ''}</div>`;
            }).join('');
            progressFilterDropdown.style.display = 'block';
        } else {
            progressFilterDropdown.innerHTML = '<div class="filter-option" style="color: var(--text-secondary);">No matches found</div>';
            progressFilterDropdown.style.display = 'block';
        }
    } catch (error) {
        console.error('Error fetching filters:', error);
    }
}, 300);

progressFilterSearch.addEventListener('input', (e) => debouncedProgressFilterSearch(e.target.value));
progressFilterSearch.addEventListener('focus', () => {
    // Show dropdown on focus, even if empty (like flow matching)
    debouncedProgressFilterSearch(progressFilterSearch.value);
});
progressFilterSearch.addEventListener('blur', () => {
    setTimeout(() => progressFilterDropdown.style.display = 'none', 200);
});
progressFilterSearch.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        progressAddFilterBtn.click();
    }
});

progressAddFilterBtn.addEventListener('click', () => {
    let filter = progressFilterSearch.value.trim();
    if (filter) {
        // Add wildcard if not present
        if (!filter.endsWith('*')) filter = filter + '*';
        if (!progressSelectedFilters.includes(filter)) {
            progressSelectedFilters.push(filter);
            renderProgressFilters();
        }
        progressFilterSearch.value = '';
        progressFilterDropdown.style.display = 'none';
    }
});

function addProgressFilter(filter) {
    // Add wildcard if not present
    if (!filter.endsWith('*')) {
        filter = filter + '*';
    }
    if (!progressSelectedFilters.includes(filter)) {
        progressSelectedFilters.push(filter);
        renderProgressFilters();
    }
    progressFilterSearch.value = '';
    progressFilterDropdown.style.display = 'none';
}

function renderProgressFilters() {
    if (progressSelectedFilters.length === 0) {
        progressSelectedFiltersContainer.innerHTML = '<div class="no-filters">No filters selected</div>';
        return;
    }
    progressSelectedFiltersContainer.innerHTML = progressSelectedFilters.map(filter =>
        '<div class="filter-tag">' + filter + '<button class="remove-filter" onclick="removeProgressFilter(\'' + filter + '\')">×</button></div>'
    ).join('');
}

function removeProgressFilter(filter) {
    progressSelectedFilters = progressSelectedFilters.filter(f => f !== filter);
    renderProgressFilters();
}

// Initialize empty state
renderProgressFilters();

// Progress Model Name Autocomplete (filtered to progress models only)
const debouncedProgressModelNameSearch = debounce(async (search) => {
    try {
        const response = await fetch('/api/checkpoint_names?search=' + encodeURIComponent(search) + '&prefix=rectify_progress_');
        const data = await response.json();
        if (data.success && data.names.length > 0) {
            progressModelNameDropdown.innerHTML = data.names.map(name => {
                // Strip "rectify_progress_" prefix for display
                const displayName = name.startsWith('rectify_progress_') ? name.substring(17) : name;
                return '<div class="dropdown-item" onclick="selectProgressCheckpointName(\'' + name + '\')">' + displayName + '</div>';
            }).join('');
            progressModelNameDropdown.style.display = 'block';
        } else {
            progressModelNameDropdown.style.display = 'none';
        }
    } catch (error) {
        console.error('Error fetching checkpoint names:', error);
    }
}, 300);

progressModelNameInput.addEventListener('input', (e) => {
    const search = e.target.value;
    if (search.length > 0) {
        debouncedProgressModelNameSearch(search);
    } else {
        progressModelNameDropdown.style.display = 'none';
    }
});

progressModelNameInput.addEventListener('focus', (e) => {
    if (e.target.value.length > 0) {
        debouncedProgressModelNameSearch(e.target.value);
    }
});

progressModelNameInput.addEventListener('blur', () => {
    setTimeout(() => progressModelNameDropdown.style.display = 'none', 200);
});

function selectProgressCheckpointName(name) {
    // Strip "rectify_progress_" prefix (input only shows suffix)
    const displayName = name.startsWith('rectify_progress_') ? name.substring(17) : name;
    progressModelNameInput.value = displayName;
    progressModelNameDropdown.style.display = 'none';
}

// Progress Prediction WebSocket
let progressWs = null;

function connectProgressWebSocket() {
    // Don't create duplicate connections
    if (progressWs && progressWs.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = protocol + '//' + window.location.host + '/ws/progress_status';

    progressWs = new WebSocket(wsUrl);

    progressWs.onmessage = (event) => {
        const status = JSON.parse(event.data);
        updateProgressStatus(status);
    };

    progressWs.onclose = () => {
        progressWs = null;
    };
}

function updateProgressStatus(status) {
    if (!status.connected) return;

    // Skip all updates if user clicked "New Model" (form is cleared)
    if (ppDisableAutoFill) {
        ppLatestStatus = status;
        return;
    }

    ppLatestStatus = status;
    const phase = status.phase;
    const isTrainingActive = phase !== 'idle' && phase !== 'finished' && phase !== 'failed';

    // Form is read-only if a model is loaded (even if finished)
    const hasLoadedModel = status.model_name && status.model_name.length > 0;
    setProgressFormReadOnly(hasLoadedModel);

    // Auto-fill form if training config is available
    if (status.model_name) {
        const modelNameInput = document.getElementById('progressModelName');
        // Strip "rectify_progress_" prefix if present (input only shows suffix)
        const displayName = status.model_name.startsWith('rectify_progress_')
            ? status.model_name.substring(17)
            : status.model_name;
        if (!modelNameInput.value || modelNameInput.value !== displayName) {
            modelNameInput.value = displayName;
        }
    }

    // Always update entry filters if provided and different from current selection
    if (status.entry_filters && status.entry_filters.length > 0) {
        const currentFiltersStr = JSON.stringify(progressSelectedFilters.sort());
        const newFiltersStr = JSON.stringify([...status.entry_filters].sort());
        if (currentFiltersStr !== newFiltersStr) {
            progressSelectedFilters = [...status.entry_filters];
            renderProgressFilters();
            console.log('[Progress] Loaded entry filters:', progressSelectedFilters);
        }
    }

    if (status.batch_size) {
        document.getElementById('progressBatchSize').value = status.batch_size;
    }
    if (status.task_type) {
        document.getElementById('progressTaskType').value = status.task_type;
    }

    // Update phase indicator
    const phaseValue = document.getElementById('ppPhaseValue');
    phaseValue.textContent = formatPhase(phase);
    phaseValue.className = 'phase-value ' + phase;

    // Update progress bar
    const progressBar = document.getElementById('ppProgressBar');
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

    // Update metrics based on phase
    if (phase === 'preparing_dataset') {
        document.getElementById('ppStepsValue').textContent =
            status.export_entries_processed.toLocaleString() + ' / ' + status.export_entries_total.toLocaleString();
        document.getElementById('ppLossValue').textContent = '—';
        document.getElementById('ppAccuracyValue').textContent = '—';
        document.getElementById('ppF1Value').textContent = '—';
        document.getElementById('ppValLossValue').textContent = '—';
        document.getElementById('ppValAccuracyValue').textContent = '—';
        document.getElementById('ppSpeedValue').textContent = 'Exporting...';
        document.getElementById('ppEtaValue').textContent = '—';
    } else {
        document.getElementById('ppStepsValue').textContent =
            status.steps_completed.toLocaleString() + ' / ' + status.max_steps.toLocaleString();
        document.getElementById('ppLossValue').textContent =
            status.loss !== null && status.loss !== undefined ? status.loss.toFixed(4) : '—';
        document.getElementById('ppAccuracyValue').textContent =
            status.accuracy !== null && status.accuracy !== undefined ? (status.accuracy * 100).toFixed(1) + '%' : '—';
        document.getElementById('ppF1Value').textContent =
            status.f1 !== null && status.f1 !== undefined ? status.f1.toFixed(3) : '—';
        document.getElementById('ppValLossValue').textContent =
            status.val_loss !== null && status.val_loss !== undefined ? status.val_loss.toFixed(4) : '—';
        document.getElementById('ppValAccuracyValue').textContent =
            status.val_accuracy !== null && status.val_accuracy !== undefined ? (status.val_accuracy * 100).toFixed(1) + '%' : '—';
        document.getElementById('ppSpeedValue').textContent =
            status.fps ? status.fps.toFixed(1) + ' steps/s' : '—';

        // Calculate ETA
        if (status.fps && status.fps > 0 && status.max_steps > status.steps_completed) {
            const remaining = status.max_steps - status.steps_completed;
            const etaSeconds = remaining / status.fps;
            document.getElementById('ppEtaValue').textContent = formatETA(etaSeconds);
        } else {
            document.getElementById('ppEtaValue').textContent = '—';
        }

        // Update loss chart
        if (phase === 'training' && status.loss !== null && status.loss > 0) {
            updatePPLossChart(status.steps_completed, status.loss);
        }
    }

    // Update button states
    if (phase === 'finished') {
        progressStartBtn.disabled = false;
        progressStartBtn.innerHTML = '<span class="btn-icon">▶</span> Resume Training';
        progressCancelBtn.disabled = true;
        progressExportBtn.disabled = false;
        progressNewModelBtn.disabled = false;
        const gameBtn = document.getElementById('progressGameBtn');
        if (gameBtn) gameBtn.style.display = 'none';
    } else if (phase === 'training') {
        progressStartBtn.disabled = true;
        if (status.steps_completed > 0) {
            progressStartBtn.innerHTML = '<span class="btn-icon">✓</span> Training...';
        }
        progressCancelBtn.disabled = false;
        progressExportBtn.disabled = true;
        progressNewModelBtn.disabled = true;
        const gameBtn = document.getElementById('progressGameBtn');
        if (gameBtn) gameBtn.style.display = 'inline-flex';
    } else if (phase === 'preparing_dataset') {
        progressStartBtn.disabled = true;
        progressStartBtn.innerHTML = '<span class="btn-icon">⏳</span> Preparing...';
        progressCancelBtn.disabled = false;
        progressExportBtn.disabled = true;
        progressNewModelBtn.disabled = true;
        const gameBtn = document.getElementById('progressGameBtn');
        if (gameBtn) gameBtn.style.display = 'none';
    } else if (phase === 'idle') {
        progressStartBtn.disabled = false;
        progressStartBtn.innerHTML = '<span class="btn-icon">▶</span> Start Training';
        progressCancelBtn.disabled = true;
        progressExportBtn.disabled = false; // Allow export from checkpoints
        progressNewModelBtn.disabled = false;
        const gameBtn = document.getElementById('progressGameBtn');
        if (gameBtn) gameBtn.style.display = 'none';
    } else if (phase === 'failed') {
        progressStartBtn.disabled = false;
        progressStartBtn.innerHTML = '<span class="btn-icon">▶</span> Retry Training';
        progressCancelBtn.disabled = true;
        progressExportBtn.disabled = false;
        progressNewModelBtn.disabled = false;
        const gameBtn = document.getElementById('progressGameBtn');
        if (gameBtn) gameBtn.style.display = 'none';
    }
}

function setProgressFormReadOnly(readonly) {
    document.getElementById('progressModelName').disabled = readonly;
    document.getElementById('progressTrainingSteps').disabled = readonly;
    document.getElementById('progressBatchSize').disabled = readonly;
    document.getElementById('progressTaskType').disabled = readonly;
    document.getElementById('progressCheckpointInterval').disabled = readonly;
    document.getElementById('progressMaxCheckpoints').disabled = readonly;
    document.getElementById('progressForceRebuild').disabled = readonly;
    document.getElementById('progressFilterSearch').disabled = readonly;
    progressAddFilterBtn.disabled = readonly;

    // Visual indicator
    const trainingCard = document.getElementById('progressTrainingPanel');
    if (readonly) {
        trainingCard.classList.add('readonly');
    } else {
        trainingCard.classList.remove('readonly');
    }
}

// Progress Prediction Loss Chart
function updatePPLossChart(step, loss) {
    if (loss > 1e-10) {
        ppStepHistory.push(step);
        ppLossHistory.push(loss);
    }

    // Keep last 500 points
    if (ppStepHistory.length > 500) {
        ppStepHistory.shift();
        ppLossHistory.shift();
    }

    const chartContainer = document.getElementById('ppChartContainer');
    const canvas = document.getElementById('ppLossChart');

    if (ppLossHistory.length < 2) {
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

    const lossSmoothed = smoothEMA(ppLossHistory, 0.3);

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

    const minLoss = Math.min(...ppLossHistory);
    const maxLoss = Math.max(...ppLossHistory);
    const logMin = Math.log10(Math.max(minLoss, 1e-10));
    const logMax = Math.log10(maxLoss);
    const minStep = Math.min(...ppStepHistory);
    const maxStep = Math.max(...ppStepHistory);

    function toLogY(loss) {
        const logLoss = Math.log10(Math.max(loss, 1e-10));
        return height - padding.bottom - ((logLoss - logMin) / (logMax - logMin)) * (height - padding.top - padding.bottom);
    }

    function toX(step) {
        return padding.left + ((step - minStep) / (maxStep - minStep)) * (width - padding.left - padding.right);
    }

    // Draw gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, 'rgba(52, 199, 89, 0.15)');
    gradient.addColorStop(1, 'rgba(52, 199, 89, 0.02)');

    ctx.beginPath();
    ctx.moveTo(toX(ppStepHistory[0]), height - padding.bottom);
    lossSmoothed.forEach((loss, i) => {
        ctx.lineTo(toX(ppStepHistory[i]), toLogY(loss));
    });
    ctx.lineTo(toX(ppStepHistory[ppStepHistory.length - 1]), height - padding.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw smoothed line
    ctx.beginPath();
    ctx.strokeStyle = '#34c759';
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    lossSmoothed.forEach((loss, i) => {
        const x = toX(ppStepHistory[i]);
        const y = toLogY(loss);
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();

    // Current loss annotation
    const currentLoss = ppLossHistory[ppLossHistory.length - 1];
    const currentStep = ppStepHistory[ppStepHistory.length - 1];
    const currentX = toX(currentStep);
    const currentY = toLogY(lossSmoothed[lossSmoothed.length - 1]);

    ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif';
    ctx.fillStyle = '#34c759';
    ctx.fillText(currentLoss.toFixed(4), currentX - 30, currentY - 8);
}

// Progress Cancel
progressCancelBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to cancel?')) return;

    setLoadingText('Cancelling', 'Saving checkpoint and stopping...');
    showLoadingScreen();
    progressCancelBtn.disabled = true;

    try {
        fetch('/api/progress/cancel', { method: 'POST' }).catch(() => {});

        const startTime = Date.now();
        while (true) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            if (Date.now() - startTime > 120000) {
                hideLoadingScreen();
                alert('Cancel timeout after 2 minutes.');
                break;
            }

            const statusResp = await fetch('/api/progress/status');
            const statusData = await statusResp.json();
            if (statusData.phase === 'idle' || statusData.phase === 'finished') {
                hideLoadingScreen();
                break;
            }

            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            setLoadingText('Cancelling', 'Saving checkpoint... (' + elapsed + 's)');
        }
    } catch (error) {
        hideLoadingScreen();
        alert('Error: ' + error);
    } finally {
        progressCancelBtn.disabled = false;
    }
});

// Progress Export
progressExportBtn.addEventListener('click', async () => {
    progressExportBtn.disabled = true;
    progressExportBtn.innerHTML = '<span class="btn-icon">⏳</span> Exporting...';

    setLoadingText('Exporting Model', 'This may take a minute...');
    showLoadingScreen();

    try {
        const response = await fetch('/api/progress/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
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
        progressExportBtn.disabled = false;
        progressExportBtn.innerHTML = '<span class="btn-icon">↗</span> Export Model';
    }
});

// Progress New Model - Reset server and clear UI
progressNewModelBtn.addEventListener('click', async () => {
    if (!confirm('Start fresh with a new model? This will reset the server and clear all state.')) return;

    // Show loading
    setLoadingText('New Model', 'Resetting server state...');
    showLoadingScreen();

    try {
        // Call reset API (resets server + clears UI)
        const response = await fetch('/api/progress/reset', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            // Reset all UI state
            ppLossHistory = [];
            ppStepHistory = [];
            progressSelectedFilters = [];
            ppLatestStatus = null;
            ppDisableAutoFill = true;

            // Clear form
            document.getElementById('progressModelName').value = '';
            document.getElementById('progressTrainingSteps').value = '10000';
            document.getElementById('progressBatchSize').value = '32';
            document.getElementById('progressTaskType').value = 'classification';
            document.getElementById('progressCheckpointInterval').value = '1000';
            document.getElementById('progressMaxCheckpoints').value = '10';
            document.getElementById('progressForceRebuild').checked = false;
            renderProgressFilters();

            // Clear chart
            document.getElementById('ppChartContainer').style.display = 'none';

            // Reset metrics display
            document.getElementById('ppPhaseValue').textContent = 'Idle';
            document.getElementById('ppPhaseValue').className = 'phase-value';
            document.getElementById('ppProgressBar').style.width = '0%';
            document.getElementById('ppStepsValue').textContent = '0 / 0';
            document.getElementById('ppLossValue').textContent = '—';
            document.getElementById('ppAccuracyValue').textContent = '—';
            document.getElementById('ppF1Value').textContent = '—';
            document.getElementById('ppValLossValue').textContent = '—';
            document.getElementById('ppValAccuracyValue').textContent = '—';
            document.getElementById('ppSpeedValue').textContent = '—';
            document.getElementById('ppEtaValue').textContent = '—';

            // Reset buttons
            progressStartBtn.disabled = false;
            progressStartBtn.innerHTML = '<span class="btn-icon">▶</span> Start Training';
            progressCancelBtn.disabled = true;
            progressExportBtn.disabled = true;
            progressNewModelBtn.disabled = false;

            // Reset sidebar charts (progress only)
            sidebarPPLossHistory = [];
            sidebarPPStepHistory = [];

            // Enable form
            setProgressFormReadOnly(false);
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

// ============================================
// Settings Modal & Claude Chat Integration
// ============================================

// Settings Modal Elements
const settingsBtn = document.getElementById('settingsBtn');
const settingsModal = document.getElementById('settingsModal');
const closeSettingsBtn = document.getElementById('closeSettingsBtn');
const claudeApiKeyInput = document.getElementById('claudeApiKey');
const toggleApiKeyVisibility = document.getElementById('toggleApiKeyVisibility');
const apiKeyStatus = document.getElementById('apiKeyStatus');
const clearApiKeyBtn = document.getElementById('clearApiKeyBtn');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');

// Chat Interface Elements
const chatInterface = document.getElementById('chatInterface');
const chatModeBtn = document.getElementById('chatModeBtn');
const switchToTraditionalBtn = document.getElementById('switchToTraditionalBtn');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const chatSendBtn = document.getElementById('chatSendBtn');
const chatStatusText = document.getElementById('chatStatusText');

// Chat State
let chatHistory = [];
let isProcessing = false;
let chatMode = false; // true = chat interface, false = traditional UI
let chatStatusInterval = null;

// Initialize on load
function initializeSettings() {
    const savedKey = localStorage.getItem('claude_api_key');
    if (savedKey) {
        claudeApiKeyInput.value = savedKey;
        updateApiKeyStatus(true);
        // Show the chat mode button but stay in traditional UI by default
        chatModeBtn.style.display = 'flex';
    } else {
        updateApiKeyStatus(false);
        chatModeBtn.style.display = 'none';
    }
}

function updateApiKeyStatus(configured) {
    const indicator = apiKeyStatus.querySelector('.status-indicator');
    const text = apiKeyStatus.querySelector('.status-text');

    if (configured) {
        indicator.className = 'status-indicator configured';
        text.textContent = 'API key configured';
    } else {
        indicator.className = 'status-indicator not-configured';
        text.textContent = 'Not configured';
    }
}

function showChatInterface() {
    chatMode = true;
    chatInterface.style.display = 'flex';
    chatModeBtn.style.display = 'none';
    document.querySelector('.app-container').style.display = 'none';

    // Ensure both WebSockets are connected for sidebar updates
    if (!ws) {
        connectWebSocket();
    }
    if (!progressWs) {
        connectProgressWebSocket();
    }

    startChatStatusUpdates();
    // Initialize sidebar with current state
    updateTrainingSidebar();
}

function hideChatInterface() {
    chatMode = false;
    chatInterface.style.display = 'none';
    chatModeBtn.style.display = 'flex';
    document.querySelector('.app-container').style.display = 'block';
    stopChatStatusUpdates();

    // Auto-show the correct training panel if training is active
    const skillActive = latestStatus && latestStatus.phase &&
        !['idle'].includes(latestStatus.phase);
    const progressActive = ppLatestStatus && ppLatestStatus.phase &&
        !['idle'].includes(ppLatestStatus.phase);

    if (skillActive && trainerSelection.style.display !== 'none') {
        // Training is active but user is on trainer selection - switch to skill panel
        trainerSelection.style.display = 'none';
        trainingPanel.style.display = 'block';
        progressPanel.style.display = 'block';
    } else if (progressActive && trainerSelection.style.display !== 'none') {
        // Progress training is active - switch to progress panel
        trainerSelection.style.display = 'none';
        document.getElementById('progressTrainingPanel').style.display = 'block';
        document.getElementById('progressPredictionPanel').style.display = 'block';
    }
}

// Settings Modal Controls
settingsBtn.addEventListener('click', () => {
    settingsModal.style.display = 'flex';
    setTimeout(() => settingsModal.classList.add('active'), 10);
});

closeSettingsBtn.addEventListener('click', () => {
    settingsModal.classList.remove('active');
    setTimeout(() => settingsModal.style.display = 'none', 300);
});

settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
        settingsModal.classList.remove('active');
        setTimeout(() => settingsModal.style.display = 'none', 300);
    }
});

toggleApiKeyVisibility.addEventListener('click', () => {
    if (claudeApiKeyInput.type === 'password') {
        claudeApiKeyInput.type = 'text';
    } else {
        claudeApiKeyInput.type = 'password';
    }
});

clearApiKeyBtn.addEventListener('click', () => {
    localStorage.removeItem('claude_api_key');
    claudeApiKeyInput.value = '';
    updateApiKeyStatus(false);
    chatModeBtn.style.display = 'none';
    hideChatInterface();
    chatHistory = [];
    renderChatMessages();
});

saveSettingsBtn.addEventListener('click', () => {
    const apiKey = claudeApiKeyInput.value.trim();
    if (apiKey) {
        localStorage.setItem('claude_api_key', apiKey);
        updateApiKeyStatus(true);
        chatModeBtn.style.display = 'flex';

        // Close modal
        settingsModal.classList.remove('active');
        setTimeout(() => settingsModal.style.display = 'none', 300);
    } else {
        localStorage.removeItem('claude_api_key');
        updateApiKeyStatus(false);
        chatModeBtn.style.display = 'none';
        hideChatInterface();
    }
});

// Chat Mode Toggle
chatModeBtn.addEventListener('click', () => {
    showChatInterface();
});

switchToTraditionalBtn.addEventListener('click', () => {
    hideChatInterface();
});

// Auto-resize chat input
chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 100) + 'px';
    chatSendBtn.disabled = !chatInput.value.trim();

    // Check for # autocomplete trigger
    handleChatAutocomplete();
});

// Chat autocomplete state
const chatAutocompleteDropdown = document.getElementById('chatAutocompleteDropdown');
let chatAutocompleteItems = [];
let chatAutocompleteSelectedIndex = -1;
let chatAutocompleteActive = false;
let chatAutocompleteStartPos = -1;
let chatAutocompleteType = null; // 'filter' or 'command'

// Available slash commands with state requirements
const CHAT_COMMANDS = [
    { command: '/train', label: 'Train Skill Model', description: 'Start training a new skill model', icon: '🤖', prompt: 'Start training a skill model named ', requiresIdle: true },
    { command: '/progress', label: 'Train Progress Predictor', description: 'Start training a progress prediction model', icon: '📊', prompt: 'Start training a progress predictor named ', requiresIdle: true },
    { command: '/status', label: 'Check Status', description: 'View current training status and progress', icon: '📈', prompt: 'Show me the current training status', alwaysAvailable: true },
    { command: '/cancel', label: 'Cancel Training', description: 'Stop the current training job', icon: '⏹️', prompt: 'Cancel the current training', requiresRunning: true },
    { command: '/export', label: 'Export Model', description: 'Export the trained model to warehouse', icon: '📦', prompt: 'Export the current model', requiresFinished: true },
    { command: '/models', label: 'List Models', description: 'Show available exported models', icon: '📋', prompt: 'List all exported models', alwaysAvailable: true },
    { command: '/reset', label: 'Reset Trainer', description: 'Reset trainer to initial state', icon: '🔄', prompt: 'Reset the trainer', alwaysAvailable: true },
];

// Get current training state for command availability
async function getTrainingState() {
    try {
        const [skillRes, progressRes] = await Promise.all([
            fetch('/api/training_status').then(r => r.json()).catch(() => ({ connected: false })),
            fetch('/api/progress_training_status').then(r => r.json()).catch(() => ({ connected: false }))
        ]);

        const skillPhase = skillRes.connected ? skillRes.phase : null;
        const progressPhase = progressRes.connected ? progressRes.phase : null;

        const runningPhases = ['preparing_dataset', 'exporting_dataset', 'training'];
        const isSkillRunning = runningPhases.includes(skillPhase);
        const isProgressRunning = runningPhases.includes(progressPhase);
        const isSkillFinished = skillPhase === 'finished';
        const isProgressFinished = progressPhase === 'finished';

        return {
            connected: skillRes.connected || progressRes.connected,
            isRunning: isSkillRunning || isProgressRunning,
            isIdle: !isSkillRunning && !isProgressRunning,
            isFinished: isSkillFinished || isProgressFinished,
            skillPhase,
            progressPhase
        };
    } catch (error) {
        console.error('Error getting training state:', error);
        return { connected: false, isRunning: false, isIdle: true, isFinished: false };
    }
}

// Filter commands based on current state
function getAvailableCommands(state, searchTerm = '') {
    return CHAT_COMMANDS.map(cmd => {
        let available = false;
        let reason = '';

        if (cmd.alwaysAvailable) {
            available = true;
        } else if (cmd.requiresIdle) {
            available = state.isIdle;
            reason = state.isRunning ? 'Training in progress' : '';
        } else if (cmd.requiresRunning) {
            available = state.isRunning;
            reason = state.isIdle ? 'No active training' : '';
        } else if (cmd.requiresFinished) {
            available = state.isFinished;
            reason = !state.isFinished ? 'No finished training to export' : '';
        }

        return { ...cmd, available, reason };
    }).filter(cmd =>
        cmd.command.toLowerCase().includes(searchTerm) ||
        cmd.label.toLowerCase().includes(searchTerm)
    );
}

async function handleChatAutocomplete() {
    const value = chatInput.value;
    const cursorPos = chatInput.selectionStart;

    // Check for / command at start of input or after newline
    if (value.startsWith('/') || value.includes('\n/')) {
        let slashPos = value.lastIndexOf('\n/');
        if (slashPos === -1 && value.startsWith('/')) {
            slashPos = 0;
        } else if (slashPos !== -1) {
            slashPos += 1; // Skip the newline
        }

        // Only trigger if cursor is after the slash
        if (slashPos !== -1 && cursorPos > slashPos) {
            const searchTerm = value.substring(slashPos + 1, cursorPos).toLowerCase();

            // Dismiss autocomplete if there's a space (command is complete)
            if (searchTerm.includes(' ')) {
                hideChatAutocomplete();
                return;
            }

            chatAutocompleteStartPos = slashPos;
            chatAutocompleteType = 'command';
            chatAutocompleteActive = true;

            // Get current state and filter commands
            const state = await getTrainingState();
            const filteredCommands = getAvailableCommands(state, searchTerm);

            chatAutocompleteItems = filteredCommands;
            // Select first available command
            const firstAvailableIdx = filteredCommands.findIndex(cmd => cmd.available);
            chatAutocompleteSelectedIndex = firstAvailableIdx >= 0 ? firstAvailableIdx : 0;
            renderChatAutocomplete();
            return;
        }
    }

    // Find the @ trigger before cursor for entry filters
    let atPos = -1;
    for (let i = cursorPos - 1; i >= 0; i--) {
        if (value[i] === '@') {
            atPos = i;
            break;
        }
        // Stop if we hit a space or newline (no @ in current word)
        if (value[i] === ' ' || value[i] === '\n') {
            break;
        }
    }

    if (atPos === -1) {
        hideChatAutocomplete();
        return;
    }

    // Get the search term after @
    const searchTerm = value.substring(atPos + 1, cursorPos);
    chatAutocompleteStartPos = atPos;
    chatAutocompleteType = 'filter';
    chatAutocompleteActive = true;

    // Fetch filters from API
    try {
        const response = await fetch(`/api/entry_filters?search=${encodeURIComponent(searchTerm)}`);
        const data = await response.json();

        if (data.filters && data.filters.length > 0) {
            chatAutocompleteItems = data.filters.slice(0, 8); // Limit to 8 results
            chatAutocompleteSelectedIndex = 0;
            renderChatAutocomplete();
        } else {
            chatAutocompleteItems = [];
            renderChatAutocomplete();
        }
    } catch (error) {
        console.error('Error fetching filters:', error);
        hideChatAutocomplete();
    }
}

function renderChatAutocomplete() {
    if (!chatAutocompleteActive) {
        hideChatAutocomplete();
        return;
    }

    if (chatAutocompleteType === 'command') {
        renderCommandAutocomplete();
    } else {
        renderFilterAutocomplete();
    }

    chatAutocompleteDropdown.classList.add('visible');

    // Add click handlers
    chatAutocompleteDropdown.querySelectorAll('.chat-autocomplete-item').forEach(item => {
        item.addEventListener('click', () => {
            // Check if item is available (for commands)
            if (item.dataset.available === 'false') return;
            selectChatAutocompleteItem(parseInt(item.dataset.index));
        });
    });
}

function renderCommandAutocomplete() {
    if (chatAutocompleteItems.length === 0) {
        chatAutocompleteDropdown.innerHTML = `
            <div class="chat-autocomplete-header">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M4 17l6-6-6-6M12 19h8"/>
                </svg>
                Commands
            </div>
            <div class="chat-autocomplete-empty">No matching commands</div>
        `;
        return;
    }

    const itemsHtml = chatAutocompleteItems.map((cmd, index) => {
        const isSelected = index === chatAutocompleteSelectedIndex;
        const disabledClass = cmd.available ? '' : 'disabled';
        const selectedClass = isSelected && cmd.available ? 'selected' : '';

        return `
            <div class="chat-autocomplete-item ${selectedClass} ${disabledClass}"
                 data-index="${index}" data-available="${cmd.available}">
                <div class="chat-autocomplete-icon ${disabledClass}">${cmd.icon}</div>
                <div class="chat-autocomplete-content">
                    <span class="chat-autocomplete-label">${escapeHtml(cmd.label)}</span>
                    <span class="chat-autocomplete-desc">${cmd.available ? escapeHtml(cmd.description) : escapeHtml(cmd.reason)}</span>
                </div>
                ${cmd.available
                    ? `<span class="chat-autocomplete-command">${escapeHtml(cmd.command)}</span>`
                    : `<span class="chat-autocomplete-unavailable">Unavailable</span>`
                }
            </div>
        `;
    }).join('');

    chatAutocompleteDropdown.innerHTML = `
        <div class="chat-autocomplete-header">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M4 17l6-6-6-6M12 19h8"/>
            </svg>
            Commands
        </div>
        ${itemsHtml}
    `;
}

function renderFilterAutocomplete() {
    if (chatAutocompleteItems.length === 0) {
        chatAutocompleteDropdown.innerHTML = `
            <div class="chat-autocomplete-header">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
                </svg>
                Entry Filters
            </div>
            <div class="chat-autocomplete-empty">No matching filters found</div>
        `;
        return;
    }

    const itemsHtml = chatAutocompleteItems.map((filter, index) => `
        <div class="chat-autocomplete-item ${index === chatAutocompleteSelectedIndex ? 'selected' : ''}"
             data-index="${index}" data-filter="${escapeHtml(filter)}">
            <div class="chat-autocomplete-icon">@</div>
            <span class="chat-autocomplete-text">${escapeHtml(filter)}</span>
            <span class="chat-autocomplete-hint">⏎ to select</span>
        </div>
    `).join('');

    chatAutocompleteDropdown.innerHTML = `
        <div class="chat-autocomplete-header">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                <circle cx="12" cy="7" r="4"/>
            </svg>
            Entry Filters
        </div>
        ${itemsHtml}
    `;
}

function selectChatAutocompleteItem(index) {
    if (index < 0 || index >= chatAutocompleteItems.length) return;

    const value = chatInput.value;
    const cursorPos = chatInput.selectionStart;

    if (chatAutocompleteType === 'command') {
        const cmd = chatAutocompleteItems[index];
        // Don't select unavailable commands
        if (!cmd.available) return;
        // Replace the entire input with the command's prompt
        chatInput.value = cmd.prompt;
        chatInput.setSelectionRange(cmd.prompt.length, cmd.prompt.length);
    } else {
        // Entry filter - insert at @ position
        const filter = chatAutocompleteItems[index];
        const before = value.substring(0, chatAutocompleteStartPos);
        const after = value.substring(cursorPos);

        chatInput.value = before + filter + ' ' + after;

        // Set cursor after the inserted filter
        const newCursorPos = chatAutocompleteStartPos + filter.length + 1;
        chatInput.setSelectionRange(newCursorPos, newCursorPos);
    }

    hideChatAutocomplete();
    chatInput.focus();
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 100) + 'px';
    chatSendBtn.disabled = !chatInput.value.trim();
}

function hideChatAutocomplete() {
    chatAutocompleteActive = false;
    chatAutocompleteItems = [];
    chatAutocompleteSelectedIndex = -1;
    chatAutocompleteType = null;
    chatAutocompleteDropdown.classList.remove('visible');
}

// Hide autocomplete when clicking outside
chatInput.addEventListener('blur', () => {
    // Delay to allow clicking on dropdown items
    setTimeout(() => hideChatAutocomplete(), 200);
});

// Find next available command index (for keyboard navigation)
function findNextAvailableIndex(currentIndex, direction) {
    if (chatAutocompleteType !== 'command') {
        // For filters, all items are available
        const newIndex = currentIndex + direction;
        if (newIndex < 0) return chatAutocompleteItems.length - 1;
        if (newIndex >= chatAutocompleteItems.length) return 0;
        return newIndex;
    }

    // For commands, skip unavailable ones
    let index = currentIndex;
    const itemCount = chatAutocompleteItems.length;
    for (let i = 0; i < itemCount; i++) {
        index = index + direction;
        if (index < 0) index = itemCount - 1;
        if (index >= itemCount) index = 0;
        if (chatAutocompleteItems[index].available) {
            return index;
        }
    }
    return currentIndex; // No available items found
}

// Send message on Enter (Shift+Enter for newline)
chatInput.addEventListener('keydown', (e) => {
    // Handle autocomplete navigation
    if (chatAutocompleteActive && chatAutocompleteItems.length > 0) {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            chatAutocompleteSelectedIndex = findNextAvailableIndex(chatAutocompleteSelectedIndex, 1);
            renderChatAutocomplete();
            return;
        }
        if (e.key === 'ArrowUp') {
            e.preventDefault();
            chatAutocompleteSelectedIndex = findNextAvailableIndex(chatAutocompleteSelectedIndex, -1);
            renderChatAutocomplete();
            return;
        }
        if (e.key === 'Enter' || e.key === 'Tab') {
            e.preventDefault();
            selectChatAutocompleteItem(chatAutocompleteSelectedIndex);
            return;
        }
        if (e.key === 'Escape') {
            e.preventDefault();
            hideChatAutocomplete();
            return;
        }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (chatInput.value.trim() && !isProcessing) {
            sendMessage();
        }
    }
});

chatSendBtn.addEventListener('click', () => {
    if (chatInput.value.trim() && !isProcessing) {
        sendMessage();
    }
});

// Suggestion cards (full screen chat)
function bindSuggestionCards() {
    document.querySelectorAll('.suggestion-card').forEach(card => {
        card.addEventListener('click', () => {
            chatInput.value = card.dataset.suggestion;
            chatInput.dispatchEvent(new Event('input'));
            sendMessage();
        });
    });
}

// Render chat messages
function renderChatMessages() {
    if (chatHistory.length === 0) {
        chatMessages.innerHTML = `
            <div class="chat-welcome-full">
                <div class="welcome-icon-large">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>
                        <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>
                        <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>
                    </svg>
                </div>
                <h2>Hi! I'm your R2 training assistant.</h2>
                <p>I can help you train robot skill models and progress predictors using natural language.</p>
                <div class="welcome-suggestions-grid">
                    <button class="suggestion-card" data-suggestion="Start training a skill model named my_skill with entry filter rectify_*">
                        <span class="suggestion-icon">🤖</span>
                        <span class="suggestion-text">Train a skill model</span>
                    </button>
                    <button class="suggestion-card" data-suggestion="Start training a progress predictor named my_progress with entry filter rectify_*">
                        <span class="suggestion-icon">📊</span>
                        <span class="suggestion-text">Train progress predictor</span>
                    </button>
                    <button class="suggestion-card" data-suggestion="What's the current training status?">
                        <span class="suggestion-icon">📈</span>
                        <span class="suggestion-text">Check training status</span>
                    </button>
                    <button class="suggestion-card" data-suggestion="List all exported models">
                        <span class="suggestion-icon">📦</span>
                        <span class="suggestion-text">List exported models</span>
                    </button>
                </div>
            </div>
        `;
        bindSuggestionCards();
        return;
    }

    chatMessages.innerHTML = chatHistory.map(msg => {
        if (msg.role === 'user') {
            return `
                <div class="chat-message user">
                    <div class="message-avatar">👤</div>
                    <div class="message-content">
                        <div class="message-text">${escapeHtml(msg.content)}</div>
                    </div>
                </div>
            `;
        } else if (msg.role === 'assistant') {
            let toolCallHtml = '';
            if (msg.toolCalls && msg.toolCalls.length > 0) {
                toolCallHtml = msg.toolCalls.map(tc => `
                    <div class="tool-call-embed">
                        <span class="tool-call-name">🔧 ${tc.name}</span>
                        ${tc.result ? `<div class="tool-call-result">✓ ${tc.result}</div>` : ''}
                    </div>
                `).join('');
            }

            return `
                <div class="chat-message assistant">
                    <div class="message-avatar">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                            <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2"/>
                        </svg>
                    </div>
                    <div class="message-content">
                        <div class="message-text">${formatMarkdown(msg.content)}</div>
                        ${toolCallHtml}
                    </div>
                </div>
            `;
        } else if (msg.role === 'system') {
            return `
                <div class="chat-message system">
                    <div class="message-avatar">ℹ️</div>
                    <div class="message-content">
                        <div class="message-text">${escapeHtml(msg.content)}</div>
                    </div>
                </div>
            `;
        }
        return '';
    }).join('');

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatEta(seconds) {
    if (!seconds || seconds < 0) return '—';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
}

// ============================================
// Chat Training Sidebar
// ============================================

const chatTrainingSidebar = document.getElementById('chatTrainingSidebar');
const sidebarSkillTab = document.getElementById('sidebarSkillTab');
const sidebarProgressTab = document.getElementById('sidebarProgressTab');
const sidebarSkillPanel = document.getElementById('sidebarSkillPanel');
const sidebarProgressPanel = document.getElementById('sidebarProgressPanel');
const sidebarIdleState = document.getElementById('sidebarIdleState');

// Sidebar chart state
let sidebarSkillLossHistory = [];
let sidebarSkillStepHistory = [];
let sidebarPPLossHistory = [];
let sidebarPPStepHistory = [];

// Track which trainer tab is selected
let sidebarActiveTrainer = 'skill';

// Initialize sidebar tab switching
sidebarSkillTab?.addEventListener('click', () => {
    sidebarActiveTrainer = 'skill';
    sidebarSkillTab.classList.add('active');
    sidebarProgressTab.classList.remove('active');
    sidebarSkillPanel.style.display = 'flex';
    sidebarProgressPanel.style.display = 'none';
});

sidebarProgressTab?.addEventListener('click', () => {
    sidebarActiveTrainer = 'progress';
    sidebarProgressTab.classList.add('active');
    sidebarSkillTab.classList.remove('active');
    sidebarProgressPanel.style.display = 'flex';
    sidebarSkillPanel.style.display = 'none';
});

// Update sidebar based on current training status
function updateTrainingSidebar() {
    if (!chatTrainingSidebar) return;

    const skillStatus = latestStatus;
    const progressStatus = ppLatestStatus;

    const skillActive = skillStatus && skillStatus.phase &&
        !['idle'].includes(skillStatus.phase);
    const progressActive = progressStatus && progressStatus.phase &&
        !['idle'].includes(progressStatus.phase);

    // Determine if any training is active
    const anyActive = skillActive || progressActive;

    // Toggle idle state
    if (anyActive) {
        chatTrainingSidebar.classList.remove('is-idle');
    } else {
        chatTrainingSidebar.classList.add('is-idle');
    }

    // Update tab indicators for activity
    if (skillActive && skillStatus.phase !== 'finished' && skillStatus.phase !== 'failed') {
        sidebarSkillTab?.classList.add('has-activity');
    } else {
        sidebarSkillTab?.classList.remove('has-activity');
    }

    if (progressActive && progressStatus.phase !== 'finished' && progressStatus.phase !== 'failed') {
        sidebarProgressTab?.classList.add('has-activity');
    } else {
        sidebarProgressTab?.classList.remove('has-activity');
    }

    // Auto-switch to active trainer if one just started
    if (skillActive && !progressActive && sidebarActiveTrainer !== 'skill' &&
        skillStatus.phase !== 'finished' && skillStatus.phase !== 'failed') {
        sidebarSkillTab?.click();
    } else if (progressActive && !skillActive && sidebarActiveTrainer !== 'progress' &&
        progressStatus.phase !== 'finished' && progressStatus.phase !== 'failed') {
        sidebarProgressTab?.click();
    }

    // Update skill panel
    if (skillStatus) {
        updateSidebarSkillPanel(skillStatus);
    }

    // Update progress panel
    if (progressStatus) {
        updateSidebarProgressPanel(progressStatus);
    }
}

function updateSidebarSkillPanel(status) {
    const phase = status.phase || 'idle';
    const isExporting = phase === 'exporting_dataset' || phase === 'preparing_dataset';

    // Model name
    const modelNameEl = document.getElementById('sidebarModelName');
    if (modelNameEl) {
        modelNameEl.textContent = status.model_name || '—';
        modelNameEl.title = status.model_name || '';
    }

    // Phase
    const phaseDot = document.getElementById('sidebarPhaseDot');
    const phaseText = document.getElementById('sidebarPhaseText');
    if (phaseDot) {
        phaseDot.className = 'phase-dot';
        if (phase === 'training') phaseDot.classList.add('training');
        else if (phase === 'preparing_dataset' || phase === 'exporting_dataset') phaseDot.classList.add('preparing');
        else if (phase === 'finished') phaseDot.classList.add('finished');
        else if (phase === 'failed') phaseDot.classList.add('failed');
        else phaseDot.classList.add('idle');
    }
    if (phaseText) {
        const phaseLabels = {
            'idle': 'Idle',
            'preparing_dataset': 'Preparing Dataset',
            'exporting_dataset': 'Exporting Dataset',
            'training': 'Training',
            'finished': 'Finished',
            'failed': 'Failed'
        };
        phaseText.textContent = phaseLabels[phase] || phase;
    }

    // Progress
    const progressFill = document.getElementById('sidebarProgressFill');
    const progressText = document.getElementById('sidebarProgressText');
    let progress = 0;
    if (isExporting) {
        const total = status.export_entries_total || 0;
        const processed = status.export_entries_processed || 0;
        progress = total > 0 ? Math.round((processed / total) * 100) : 0;
    } else {
        progress = status.max_steps > 0 ? Math.round((status.steps_completed / status.max_steps) * 100) : 0;
    }
    if (progressFill) progressFill.style.width = `${progress}%`;
    if (progressText) progressText.textContent = `${progress}%`;

    // Metrics
    const stepsEl = document.getElementById('sidebarSteps');
    const lossEl = document.getElementById('sidebarLoss');
    const speedEl = document.getElementById('sidebarSpeed');
    const etaEl = document.getElementById('sidebarEta');

    if (stepsEl) {
        if (isExporting) {
            stepsEl.textContent = `${status.export_entries_processed || 0} / ${status.export_entries_total || 0}`;
        } else {
            stepsEl.textContent = `${status.steps_completed || 0} / ${status.max_steps || 0}`;
        }
    }
    if (lossEl) lossEl.textContent = status.loss !== null && status.loss !== undefined ? status.loss.toFixed(4) : '—';
    if (speedEl) speedEl.textContent = status.fps !== null && status.fps !== undefined ? `${status.fps.toFixed(1)} it/s` : '—';
    if (etaEl) {
        if (status.fps && status.max_steps && status.steps_completed) {
            etaEl.textContent = formatEta((status.max_steps - status.steps_completed) / status.fps);
        } else {
            etaEl.textContent = '—';
        }
    }

    // Update chart (skip step 0 to avoid initial bump)
    if (phase === 'training' && status.loss !== null && status.loss !== undefined && status.steps_completed > 0) {
        const step = status.steps_completed;
        if (sidebarSkillStepHistory.length === 0 || sidebarSkillStepHistory[sidebarSkillStepHistory.length - 1] !== step) {
            sidebarSkillStepHistory.push(step);
            sidebarSkillLossHistory.push(status.loss);
            if (sidebarSkillLossHistory.length > 100) {
                sidebarSkillLossHistory.shift();
                sidebarSkillStepHistory.shift();
            }
            drawSidebarChart('sidebarLossChart', sidebarSkillLossHistory, sidebarSkillStepHistory);
        }
    }

    // Filters
    const filtersListEl = document.getElementById('sidebarFiltersList');
    if (filtersListEl && status.entry_filters && status.entry_filters.length > 0) {
        filtersListEl.innerHTML = status.entry_filters.map(f =>
            `<span class="filter-tag">${escapeHtml(f)}</span>`
        ).join('');
        document.getElementById('sidebarFilters').style.display = 'flex';
    } else if (filtersListEl) {
        document.getElementById('sidebarFilters').style.display = 'none';
    }
}

function updateSidebarProgressPanel(status) {
    const phase = status.phase || 'idle';
    const isExporting = phase === 'exporting_dataset' || phase === 'preparing_dataset';

    // Model name
    const modelNameEl = document.getElementById('sidebarPPModelName');
    if (modelNameEl) {
        modelNameEl.textContent = status.model_name || '—';
        modelNameEl.title = status.model_name || '';
    }

    // Phase
    const phaseDot = document.getElementById('sidebarPPPhaseDot');
    const phaseText = document.getElementById('sidebarPPPhaseText');
    if (phaseDot) {
        phaseDot.className = 'phase-dot';
        if (phase === 'training') phaseDot.classList.add('training');
        else if (phase === 'preparing_dataset' || phase === 'exporting_dataset') phaseDot.classList.add('preparing');
        else if (phase === 'finished') phaseDot.classList.add('finished');
        else if (phase === 'failed') phaseDot.classList.add('failed');
        else phaseDot.classList.add('idle');
    }
    if (phaseText) {
        const phaseLabels = {
            'idle': 'Idle',
            'preparing_dataset': 'Preparing Dataset',
            'exporting_dataset': 'Exporting Dataset',
            'training': 'Training',
            'finished': 'Finished',
            'failed': 'Failed'
        };
        phaseText.textContent = phaseLabels[phase] || phase;
    }

    // Progress
    const progressFill = document.getElementById('sidebarPPProgressFill');
    const progressText = document.getElementById('sidebarPPProgressText');
    let progress = 0;
    if (isExporting) {
        const total = status.export_entries_total || 0;
        const processed = status.export_entries_processed || 0;
        progress = total > 0 ? Math.round((processed / total) * 100) : 0;
    } else {
        progress = status.max_steps > 0 ? Math.round((status.steps_completed / status.max_steps) * 100) : 0;
    }
    if (progressFill) progressFill.style.width = `${progress}%`;
    if (progressText) progressText.textContent = `${progress}%`;

    // Metrics
    const stepsEl = document.getElementById('sidebarPPSteps');
    const lossEl = document.getElementById('sidebarPPLoss');
    const accuracyEl = document.getElementById('sidebarPPAccuracy');
    const speedEl = document.getElementById('sidebarPPSpeed');

    if (stepsEl) {
        if (isExporting) {
            stepsEl.textContent = `${status.export_entries_processed || 0} / ${status.export_entries_total || 0}`;
        } else {
            stepsEl.textContent = `${status.steps_completed || 0} / ${status.max_steps || 0}`;
        }
    }
    if (lossEl) lossEl.textContent = status.loss !== null && status.loss !== undefined ? status.loss.toFixed(4) : '—';
    if (accuracyEl) accuracyEl.textContent = status.accuracy !== null && status.accuracy !== undefined ? `${(status.accuracy * 100).toFixed(1)}%` : '—';
    if (speedEl) speedEl.textContent = status.fps !== null && status.fps !== undefined ? `${status.fps.toFixed(1)} it/s` : '—';

    // Update chart (skip step 0 to avoid initial bump)
    if (phase === 'training' && status.loss !== null && status.loss !== undefined && status.steps_completed > 0) {
        const step = status.steps_completed;
        if (sidebarPPStepHistory.length === 0 || sidebarPPStepHistory[sidebarPPStepHistory.length - 1] !== step) {
            sidebarPPStepHistory.push(step);
            sidebarPPLossHistory.push(status.loss);
            if (sidebarPPLossHistory.length > 100) {
                sidebarPPLossHistory.shift();
                sidebarPPStepHistory.shift();
            }
            drawSidebarChart('sidebarPPLossChart', sidebarPPLossHistory, sidebarPPStepHistory);
        }
    }

    // Filters
    const filtersListEl = document.getElementById('sidebarPPFiltersList');
    if (filtersListEl && status.entry_filters && status.entry_filters.length > 0) {
        filtersListEl.innerHTML = status.entry_filters.map(f =>
            `<span class="filter-tag">${escapeHtml(f)}</span>`
        ).join('');
        document.getElementById('sidebarPPFilters').style.display = 'flex';
    } else if (filtersListEl) {
        document.getElementById('sidebarPPFilters').style.display = 'none';
    }
}

function drawSidebarChart(canvasId, lossHistory, stepHistory) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || lossHistory.length < 2) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    // Set canvas size
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Padding
    const padding = { top: 20, right: 10, bottom: 25, left: 45 };

    // Calculate log scale
    const validLoss = lossHistory.filter(l => l > 0);
    if (validLoss.length < 2) return;

    const logMin = Math.floor(Math.log10(Math.min(...validLoss)));
    const logMax = Math.ceil(Math.log10(Math.max(...validLoss)));

    // Smooth the loss for display
    const smoothed = [];
    const smoothWindow = Math.max(1, Math.floor(lossHistory.length / 20));
    for (let i = 0; i < lossHistory.length; i++) {
        let sum = 0, count = 0;
        for (let j = Math.max(0, i - smoothWindow); j <= i; j++) {
            if (lossHistory[j] > 0) { sum += lossHistory[j]; count++; }
        }
        smoothed.push(count > 0 ? sum / count : lossHistory[i]);
    }

    const minStep = Math.min(...stepHistory);
    const maxStep = Math.max(...stepHistory);

    function toLogY(loss) {
        const logLoss = Math.log10(Math.max(loss, 1e-10));
        return height - padding.bottom - ((logLoss - logMin) / (logMax - logMin)) * (height - padding.top - padding.bottom);
    }

    function toX(step) {
        if (maxStep === minStep) return padding.left;
        return padding.left + ((step - minStep) / (maxStep - minStep)) * (width - padding.left - padding.right);
    }

    // Colors
    const isDark = document.body.classList.contains('dark-mode') ||
        window.matchMedia('(prefers-color-scheme: dark)').matches;
    const lineColor = '#007aff';
    const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
    const textColor = isDark ? '#71767b' : '#86868b';

    // Draw gradient fill
    const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    gradient.addColorStop(0, 'rgba(0, 122, 255, 0.2)');
    gradient.addColorStop(1, 'rgba(0, 122, 255, 0.02)');

    ctx.beginPath();
    ctx.moveTo(toX(stepHistory[0]), height - padding.bottom);
    smoothed.forEach((loss, i) => {
        ctx.lineTo(toX(stepHistory[i]), toLogY(loss));
    });
    ctx.lineTo(toX(stepHistory[stepHistory.length - 1]), height - padding.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw line
    ctx.beginPath();
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1.5;
    smoothed.forEach((loss, i) => {
        const x = toX(stepHistory[i]);
        const y = toLogY(loss);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw axes labels
    ctx.fillStyle = textColor;
    ctx.font = '9px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'right';

    // Y-axis labels
    const yTicks = 3;
    for (let i = 0; i <= yTicks; i++) {
        const logVal = logMin + (logMax - logMin) * (i / yTicks);
        const val = Math.pow(10, logVal);
        const y = height - padding.bottom - (i / yTicks) * (height - padding.top - padding.bottom);
        ctx.fillText(val.toExponential(0), padding.left - 4, y + 3);

        // Grid line
        ctx.strokeStyle = gridColor;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
    }

    // X-axis labels
    ctx.textAlign = 'center';
    ctx.fillText(Math.round(minStep).toString(), padding.left, height - 5);
    ctx.fillText(Math.round(maxStep).toString(), width - padding.right, height - 5);

    // Title
    ctx.fillStyle = textColor;
    ctx.font = '9px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Loss', padding.left, 10);
}

// Reset sidebar chart data and clear canvases
function resetSidebarCharts() {
    sidebarSkillLossHistory = [];
    sidebarSkillStepHistory = [];
    sidebarPPLossHistory = [];
    sidebarPPStepHistory = [];

    // Clear the canvas elements so old charts disappear
    ['sidebarLossChart', 'sidebarPPLossChart'].forEach(id => {
        const canvas = document.getElementById(id);
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    });
}

// Start periodic status updates in chat (updates sidebar only)
function startChatStatusUpdates() {
    if (chatStatusInterval) return;

    chatStatusInterval = setInterval(() => {
        if (!chatMode) return;
        // Update the sidebar with latest training status
        updateTrainingSidebar();
    }, 1000);
}

function stopChatStatusUpdates() {
    if (chatStatusInterval) {
        clearInterval(chatStatusInterval);
        chatStatusInterval = null;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatMarkdown(text) {
    // Basic markdown formatting
    return escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message assistant';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2"/>
            </svg>
        </div>
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isProcessing) return;

    const apiKey = localStorage.getItem('claude_api_key');
    if (!apiKey) {
        alert('Please configure your Claude API key in Settings');
        return;
    }

    // Add user message to history
    chatHistory.push({ role: 'user', content: message });
    renderChatMessages();

    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatSendBtn.disabled = true;

    // Show processing state
    isProcessing = true;
    if (chatStatusText) {
        chatStatusText.textContent = 'Thinking...';
        chatStatusText.classList.add('thinking');
    }
    showTypingIndicator();

    try {
        const response = await fetch('/api/claude/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                api_key: apiKey,
                messages: chatHistory.filter(m => m.role !== 'status'), // Don't send status messages to Claude
                context: getTrainingContext()
            })
        });

        const data = await response.json();
        hideTypingIndicator();

        if (data.success) {
            chatHistory.push({
                role: 'assistant',
                content: data.response,
                toolCalls: data.tool_calls || []
            });

            // If training was started, reset sidebar charts and re-enable traditional UI updates
            const startedTraining = data.tool_calls && data.tool_calls.some(tc =>
                tc.name.includes('start_') && tc.result && tc.result.includes('Started')
            );
            if (startedTraining) {
                resetSidebarCharts();
                // Re-enable traditional UI auto-fill so it updates when user switches back
                const isSkill = data.tool_calls.some(tc => tc.name === 'start_skill_training');
                if (isSkill) {
                    disableAutoFill = false;
                } else {
                    ppDisableAutoFill = false;
                }
            }

            // If training was cancelled or reset, clear sidebar charts
            const cancelledOrReset = data.tool_calls && data.tool_calls.some(tc =>
                tc.name === 'cancel_training' || tc.name === 'reset_trainer'
            );
            if (cancelledOrReset) {
                resetSidebarCharts();
            }
        } else {
            chatHistory.push({
                role: 'assistant',
                content: `Sorry, I encountered an error: ${data.error}`
            });
        }

        renderChatMessages();
    } catch (error) {
        hideTypingIndicator();
        chatHistory.push({
            role: 'assistant',
            content: `Sorry, I couldn't connect to the server: ${error.message}`
        });
        renderChatMessages();
    } finally {
        isProcessing = false;
        if (chatStatusText) {
            chatStatusText.textContent = 'Ready to help';
            chatStatusText.classList.remove('thinking');
        }
    }
}

// Get current training context for Claude
function getTrainingContext() {
    // Check connection status by looking at the status dot class
    const statusDot = document.querySelector('.connection-status .status-dot');
    const isConnected = statusDot && statusDot.classList.contains('connected');

    const context = {
        connected: isConnected,
        currentTrainer: null,
        skillStatus: null,
        progressStatus: null
    };

    // Determine which trainer panel is visible
    const trainingPanelEl = document.getElementById('trainingPanel');
    const progressTrainingPanelEl = document.getElementById('progressTrainingPanel');

    if (trainingPanelEl && trainingPanelEl.style.display !== 'none') {
        context.currentTrainer = 'skill';
    } else if (progressTrainingPanelEl && progressTrainingPanelEl.style.display !== 'none') {
        context.currentTrainer = 'progress';
    }

    // Get latest status if available (these are global variables set by WebSocket)
    if (typeof latestStatus !== 'undefined' && latestStatus) {
        context.skillStatus = {
            phase: latestStatus.phase,
            is_running: latestStatus.phase !== 'idle' && latestStatus.phase !== 'finished' && latestStatus.phase !== 'failed',
            steps: latestStatus.steps_completed,
            max_steps: latestStatus.max_steps,
            loss: latestStatus.loss,
            model_name: latestStatus.model_name,
            entry_filters: latestStatus.entry_filters
        };
    }

    if (typeof ppLatestStatus !== 'undefined' && ppLatestStatus) {
        context.progressStatus = {
            phase: ppLatestStatus.phase,
            is_running: ppLatestStatus.phase !== 'idle' && ppLatestStatus.phase !== 'finished' && ppLatestStatus.phase !== 'failed',
            steps: ppLatestStatus.steps_completed,
            max_steps: ppLatestStatus.max_steps,
            loss: ppLatestStatus.loss,
            model_name: ppLatestStatus.model_name,
            entry_filters: ppLatestStatus.entry_filters
        };
    }

    return context;
}

// Initialize settings on page load
initializeSettings();

// Bind suggestion cards on initial load
bindSuggestionCards();
