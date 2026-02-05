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
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = protocol + '//' + window.location.host + '/ws/progress_status';
    
    progressWs = new WebSocket(wsUrl);
    
    progressWs.onmessage = (event) => {
        const status = JSON.parse(event.data);
        console.log('[Progress] Status update:', {
            phase: status.phase,
            export_processed: status.export_entries_processed,
            export_total: status.export_entries_total,
            steps: status.steps_completed,
            max_steps: status.max_steps
        });
        updateProgressStatus(status);
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
