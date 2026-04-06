/**
 * app.js – Frontend application for the Research Synthesizer
 * 
 * Handles: file upload, WebSocket progress, report rendering, history
 */

// ══════════════════════════════════════════
// State
// ══════════════════════════════════════════

const state = {
    selectedFile: null,
    currentJobId: null,
    ws: null,
    ollamaConnected: false,
    reportMarkdown: '',
};

// ══════════════════════════════════════════
// DOM Elements
// ══════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
    // Status
    ollamaStatus: $('#ollamaStatus'),
    ollamaStatusText: $('#ollamaStatusText'),

    // Stats
    statModel: $('#statModel'),
    statReports: $('#statReports'),
    statThreshold: $('#statThreshold'),

    // Upload
    uploadSection: $('#uploadSection'),
    uploadZone: $('#uploadZone'),
    uploadBtn: $('#uploadBtn'),
    fileInput: $('#fileInput'),
    fileSelected: $('#fileSelected'),
    fileName: $('#fileName'),
    fileSize: $('#fileSize'),
    fileRemove: $('#fileRemove'),
    uploadOptions: $('#uploadOptions'),
    modelSelect: $('#modelSelect'),
    startBtn: $('#startBtn'),

    // Pipeline
    pipelineSection: $('#pipelineSection'),
    pipelineSpinner: $('#pipelineSpinner'),
    pipelineBadge: $('#pipelineBadge'),
    pipelineSteps: $('#pipelineSteps'),
    progressFill: $('#progressFill'),
    pipelineLog: $('#pipelineLog'),
    paperInfoCard: $('#paperInfoCard'),
    paperTitle: $('#paperTitle'),
    paperAbstract: $('#paperAbstract'),

    // Report
    reportSection: $('#reportSection'),
    reportContent: $('#reportContent'),
    reportMeta: $('#reportMeta'),
    copyReportBtn: $('#copyReportBtn'),
    downloadReportBtn: $('#downloadReportBtn'),
    newAnalysisBtn: $('#newAnalysisBtn'),

    // History
    historyGrid: $('#historyGrid'),
    emptyState: $('#emptyState'),

    // Toast
    toastContainer: $('#toastContainer'),
};

// ══════════════════════════════════════════
// Initialize
// ══════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    checkSystemStatus();
    loadHistory();
    setupEventListeners();
    setInterval(checkSystemStatus, 30000);
});

// ══════════════════════════════════════════
// System Status
// ══════════════════════════════════════════

async function checkSystemStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        state.ollamaConnected = data.ollama.connected;

        if (data.ollama.connected) {
            els.ollamaStatus.classList.add('connected');
            els.ollamaStatusText.textContent = `Ollama: ${data.ollama.model}`;
        } else {
            els.ollamaStatus.classList.remove('connected');
            els.ollamaStatusText.textContent = 'Ollama: Disconnected';
        }

        els.statModel.textContent = data.ollama.model || '—';
        els.statThreshold.textContent = `${(data.config.overlap_threshold * 100).toFixed(0)}%`;
    } catch {
        els.ollamaStatus.classList.remove('connected');
        els.ollamaStatusText.textContent = 'Server: Offline';
    }
}

// ══════════════════════════════════════════
// Event Listeners
// ══════════════════════════════════════════

function setupEventListeners() {
    // Upload zone drag & drop
    els.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        els.uploadZone.classList.add('dragover');
    });

    els.uploadZone.addEventListener('dragleave', () => {
        els.uploadZone.classList.remove('dragover');
    });

    els.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        els.uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].name.toLowerCase().endsWith('.pdf')) {
            selectFile(files[0]);
        } else {
            showToast('Please drop a PDF file.', 'error');
        }
    });

    // Browse button
    els.uploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        els.fileInput.click();
    });

    els.uploadZone.addEventListener('click', () => {
        if (!state.selectedFile) {
            els.fileInput.click();
        }
    });

    // File selection
    els.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectFile(e.target.files[0]);
        }
    });

    // File remove
    els.fileRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

    // Start analysis
    els.startBtn.addEventListener('click', startAnalysis);

    // Report actions
    els.copyReportBtn.addEventListener('click', copyReport);
    els.downloadReportBtn.addEventListener('click', downloadReport);
    els.newAnalysisBtn.addEventListener('click', resetUI);
}

// ══════════════════════════════════════════
// File Selection
// ══════════════════════════════════════════

function selectFile(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showToast('Only PDF files are supported.', 'error');
        return;
    }

    state.selectedFile = file;
    els.fileName.textContent = file.name;
    els.fileSize.textContent = formatFileSize(file.size);
    els.fileSelected.classList.add('visible');
    els.uploadOptions.classList.add('visible');
    els.uploadBtn.style.display = 'none';

    showToast(`Selected: ${file.name}`, 'info');
}

function clearFile() {
    state.selectedFile = null;
    els.fileInput.value = '';
    els.fileSelected.classList.remove('visible');
    els.uploadOptions.classList.remove('visible');
    els.uploadBtn.style.display = 'inline-flex';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ══════════════════════════════════════════
// Analysis Pipeline
// ══════════════════════════════════════════

async function startAnalysis() {
    if (!state.selectedFile) {
        showToast('Please select a PDF file first.', 'error');
        return;
    }

    if (!state.ollamaConnected) {
        showToast('Ollama is not running. Start it with: ollama serve', 'error');
        return;
    }

    els.startBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', state.selectedFile);
    formData.append('model', els.modelSelect.value);

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.error) {
            showToast(data.error, 'error');
            els.startBtn.disabled = false;
            return;
        }

        state.currentJobId = data.job_id;

        // Show pipeline section
        els.uploadSection.style.display = 'none';
        els.pipelineSection.classList.add('visible');

        // Connect WebSocket
        connectWebSocket(data.job_id);

        showToast('Analysis started! The agent crew is working...', 'success');
        addLog('🚀 Pipeline started for: ' + state.selectedFile.name);

    } catch (err) {
        showToast('Failed to start analysis: ' + err.message, 'error');
        els.startBtn.disabled = false;
    }
}

// ══════════════════════════════════════════
// WebSocket
// ══════════════════════════════════════════

function connectWebSocket(jobId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${jobId}`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        addLog('🔗 Connected to pipeline');
    };

    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWSMessage(data);
    };

    state.ws.onclose = () => {
        addLog('🔌 Connection closed');
    };

    state.ws.onerror = () => {
        addLog('❌ WebSocket error');
    };

    // Keep alive
    setInterval(() => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send('ping');
        }
    }, 15000);
}

function handleWSMessage(data) {
    switch (data.type) {
        case 'progress':
            updateProgress(data);
            break;

        case 'paper_info':
            showPaperInfo(data);
            break;

        case 'complete':
            handleComplete(data);
            break;

        case 'error':
            handleError(data);
            break;

        case 'status':
            updateProgress({
                step: data.step,
                total_steps: 5,
                status: data.status,
                message: `Status: ${data.step_name}`,
            });
            break;

        case 'pong':
            break;
    }
}

// ══════════════════════════════════════════
// Progress Updates
// ══════════════════════════════════════════

function updateProgress(data) {
    const { step, total_steps, message } = data;

    // Update step indicators
    const steps = $$('.pipeline-step');
    steps.forEach((el) => {
        const s = parseInt(el.dataset.step);
        el.classList.remove('active', 'done');
        if (s < step) el.classList.add('done');
        else if (s === step) el.classList.add('active');
    });

    // Update progress bar
    const pct = ((step) / total_steps) * 100;
    els.progressFill.style.width = `${Math.min(pct, 100)}%`;

    // Add log entry
    if (message) addLog(message);
}

function showPaperInfo(data) {
    els.paperTitle.textContent = data.title || 'Unknown Paper';
    els.paperAbstract.textContent = data.abstract || '';
    els.paperInfoCard.classList.add('visible');
    addLog(`📄 Paper: ${data.title}`);
}

function handleComplete(data) {
    state.reportMarkdown = data.report;

    // Update pipeline UI
    els.pipelineSpinner.style.display = 'none';
    els.pipelineBadge.classList.remove('running');
    els.pipelineBadge.classList.add('completed');
    els.pipelineBadge.textContent = 'Completed';
    els.progressFill.style.width = '100%';

    // Mark all steps done
    $$('.pipeline-step').forEach((el) => {
        el.classList.remove('active');
        el.classList.add('done');
    });

    addLog(`✅ Report completed! ${data.word_count} words in ${data.elapsed}s`);

    // Show report
    setTimeout(() => {
        showReport(data.report, data.word_count, data.elapsed);
    }, 800);

    showToast('Synthesis report is ready!', 'success');
    loadHistory();
}

function handleError(data) {
    els.pipelineSpinner.style.display = 'none';
    els.pipelineBadge.classList.remove('running');
    els.pipelineBadge.classList.add('failed');
    els.pipelineBadge.textContent = 'Failed';

    addLog(`❌ Error: ${data.message}`);
    showToast('Pipeline failed: ' + data.message, 'error');
}

// ══════════════════════════════════════════
// Report Display
// ══════════════════════════════════════════

function showReport(markdown, wordCount, elapsed) {
    els.reportSection.classList.add('visible');
    els.reportContent.innerHTML = renderMarkdown(markdown);
    els.reportMeta.textContent = `${wordCount || '—'} words • ${elapsed ? elapsed + 's' : ''}`;
    els.reportSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderMarkdown(md) {
    if (!md) return '<p>No report content available.</p>';

    // Simple markdown to HTML renderer
    let html = md
        // Headers
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')
        // Bold and italic
        .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // Code blocks
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Blockquotes
        .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
        // Unordered lists
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        // Ordered lists  
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
        // Horizontal rule
        .replace(/^---$/gm, '<hr>')
        // Links
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
        // Line breaks / paragraphs
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');

    // Wrap list items in ul
    html = html.replace(/(<li>.*<\/li>)/g, '<ul>$1</ul>');
    // Clean up consecutive uls
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    return `<p>${html}</p>`;
}

function copyReport() {
    navigator.clipboard.writeText(state.reportMarkdown).then(() => {
        showToast('Report copied to clipboard!', 'success');
        els.copyReportBtn.textContent = '✅ Copied!';
        setTimeout(() => {
            els.copyReportBtn.textContent = '📋 Copy';
        }, 2000);
    });
}

function downloadReport() {
    const blob = new Blob([state.reportMarkdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `synthesis_report_${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('Report downloaded!', 'success');
}

// ══════════════════════════════════════════
// History
// ══════════════════════════════════════════

async function loadHistory() {
    try {
        const [jobsRes, reportsRes] = await Promise.all([
            fetch('/api/jobs'),
            fetch('/api/reports'),
        ]);

        const jobsData = await jobsRes.json();
        const reportsData = await reportsRes.json();

        const jobs = jobsData.jobs || [];
        const reports = reportsData.reports || [];

        els.statReports.textContent = reports.length;

        if (jobs.length === 0 && reports.length === 0) {
            els.emptyState.style.display = 'block';
            return;
        }

        els.emptyState.style.display = 'none';
        els.historyGrid.innerHTML = '';

        // Show jobs
        jobs.forEach((job) => {
            const card = createHistoryCard(job);
            els.historyGrid.appendChild(card);
        });

        // Show orphan reports (not linked to a job)
        if (jobs.length === 0) {
            reports.forEach((report) => {
                const card = createReportCard(report);
                els.historyGrid.appendChild(card);
            });
        }

    } catch {
        // Silently fail on history load
    }
}

function createHistoryCard(job) {
    const card = document.createElement('div');
    card.className = 'history-card';
    card.innerHTML = `
        <div class="history-card-header">
            <div class="history-card-title">${job.paper_title || job.filename || 'Untitled'}</div>
            <span class="history-card-status ${job.status}">${job.status}</span>
        </div>
        <div class="history-card-meta">
            <span>📁 ${job.filename || '—'}</span>
            <span>📝 ${job.word_count ? job.word_count + ' words' : '—'}</span>
            <span>📅 ${job.started_at ? formatDate(job.started_at) : '—'}</span>
        </div>
    `;

    if (job.status === 'completed') {
        card.addEventListener('click', () => loadJobReport(job.id));
    }

    return card;
}

function createReportCard(report) {
    const card = document.createElement('div');
    card.className = 'history-card';
    card.innerHTML = `
        <div class="history-card-header">
            <div class="history-card-title">${report.filename}</div>
            <span class="history-card-status completed">Report</span>
        </div>
        <div class="history-card-meta">
            <span>📄 ${formatFileSize(report.size)}</span>
            <span>📅 ${formatDate(report.created)}</span>
        </div>
    `;

    card.addEventListener('click', () => loadSavedReport(report.filename));
    return card;
}

async function loadJobReport(jobId) {
    try {
        const res = await fetch(`/api/jobs/${jobId}`);
        const data = await res.json();

        if (data.report) {
            state.reportMarkdown = data.report;
            showReport(data.report, data.word_count, data.elapsed);
        }
    } catch {
        showToast('Failed to load report.', 'error');
    }
}

async function loadSavedReport(filename) {
    try {
        const res = await fetch(`/api/reports/${filename}`);
        const data = await res.json();

        if (data.markdown) {
            state.reportMarkdown = data.markdown;
            showReport(data.markdown);
        }
    } catch {
        showToast('Failed to load report.', 'error');
    }
}

// ══════════════════════════════════════════
// UI Helpers
// ══════════════════════════════════════════

function resetUI() {
    state.selectedFile = null;
    state.currentJobId = null;
    state.reportMarkdown = '';

    clearFile();

    els.uploadSection.style.display = 'block';
    els.pipelineSection.classList.remove('visible');
    els.reportSection.classList.remove('visible');

    // Reset pipeline
    els.pipelineSpinner.style.display = 'inline-block';
    els.pipelineBadge.className = 'pipeline-status-badge running';
    els.pipelineBadge.textContent = 'Running';
    els.progressFill.style.width = '0%';
    els.pipelineLog.innerHTML = `
        <div class="log-entry">
            <span class="timestamp">[--:--:--]</span>
            <span class="icon">⏳</span>
            Waiting for pipeline to start...
        </div>
    `;
    els.paperInfoCard.classList.remove('visible');
    $$('.pipeline-step').forEach((el) => el.classList.remove('active', 'done'));

    els.startBtn.disabled = false;

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function addLog(message) {
    const now = new Date();
    const ts = now.toTimeString().slice(0, 8);
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="timestamp">[${ts}]</span> ${message}`;
    els.pipelineLog.appendChild(entry);
    els.pipelineLog.scrollTop = els.pipelineLog.scrollHeight;
}

function formatDate(isoString) {
    if (!isoString) return '—';
    const d = new Date(isoString);
    return d.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

// ══════════════════════════════════════════
// Toast Notifications
// ══════════════════════════════════════════

function showToast(message, type = 'info') {
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span> ${message}`;
    els.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(40px)';
        toast.style.transition = 'all 0.3s ease-in';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
