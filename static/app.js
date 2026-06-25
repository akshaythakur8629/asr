const $ = id => document.getElementById(id);
let recorder, chunks = [];
let ws = null;
let audioContext = null;
let audioProcessor = null;
let audioStream = null;
let liveInterval = null;
let liveChunkCount = 0;
let liveStartTime = 0;
let isMuted = false;

// Setup Tab Navigation
$('tab-realtime').onclick = () => {
  $('tab-realtime').classList.add('active');
  $('tab-batch').classList.remove('active');
  $('tab-eval').classList.remove('active');
  $('stream-mode').value = 'websocket';
  $('realtime-actions').classList.remove('hidden');
  $('batch-actions').classList.add('hidden');
  $('mode-badge').textContent = 'Real-time Streaming';
  $('mode-badge').className = 'badge';
  $('audio-players-container').classList.add('hidden');
  $('eval').classList.add('hidden');
  $('batch-jobs-panel').classList.add('hidden');
  $('sessions-card').classList.add('hidden');
  $('onboarding-panel').classList.remove('hidden');
  $('status').classList.add('hidden');
  $('result').classList.add('hidden');
  resetTurnsConsole('Select options and click "Start Stream" to transcribe.');
};

$('tab-batch').onclick = () => {
  $('tab-batch').classList.add('active');
  $('tab-realtime').classList.remove('active');
  $('tab-eval').classList.remove('active');
  $('stream-mode').value = 'batch';
  $('batch-actions').classList.remove('hidden');
  $('realtime-actions').classList.add('hidden');
  $('mode-badge').textContent = 'Batch File Transcription';
  $('mode-badge').className = 'badge badge-indigo';
  $('audio-players-container').classList.add('hidden');
  $('eval').classList.add('hidden');
  $('batch-jobs-panel').classList.add('hidden');
  $('sessions-card').classList.remove('hidden');
  $('onboarding-panel').classList.remove('hidden');
  $('status').classList.add('hidden');
  $('result').classList.add('hidden');
  resetTurnsConsole('Drag & drop or select audio files to transcribe.');
};

$('tab-eval').onclick = () => {
  $('tab-eval').classList.add('active');
  $('tab-realtime').classList.remove('active');
  $('tab-batch').classList.remove('active');
  $('realtime-actions').classList.add('hidden');
  $('batch-actions').classList.add('hidden');
  $('mode-badge').textContent = 'Hindi Context Biasing Evaluation';
  $('mode-badge').className = 'badge badge-indigo';
  $('onboarding-panel').classList.add('hidden');
  $('status').classList.add('hidden');
  $('result').classList.add('hidden');
  $('batch-jobs-panel').classList.add('hidden');
  $('eval').classList.remove('hidden');
  $('sessions-card').classList.add('hidden');
};

// Diagnostics Panel Toggle
$('diagnostics-toggle-btn').onclick = () => {
  $('diagnostics-panel').classList.toggle('collapsed');
};

function resetTurnsConsole(msg) {
  $('onboarding-panel').classList.remove('hidden');
  $('result').classList.add('hidden');
  $('status').classList.add('hidden');
  document.querySelector('.onboarding-desc').textContent = msg;
}

// Session Creation / Refresh Evaluator
$('new-session-btn').onclick = () => {
  location.reload();
};

async function loadSamples() {
  const data = await fetch('/api/samples').then(r => r.json());
  if (data && data.length) {
    $('samples').innerHTML = data.map(x => `
      <button class="sample" data-name="${x.name}">
        <span>📄 ${x.name.substring(0, 15)}...</span>
        <small>${(x.size / 1024 / 1024).toFixed(1)} MB</small>
      </button>
    `).join('');
    document.querySelectorAll('.sample').forEach(b => b.onclick = () => submitSample(b.dataset.name));
  } else {
    $('samples').innerHTML = '<p class="muted" style="font-size:11px; padding:10px;">No sample files loaded.</p>';
  }
}

// Start WebSocket real-time streaming
async function startWebSocketStream() {
  const language = $('language').value;
  const denoise = $('denoise').value;
  const vad = $('vad').value;
  const model = $('model').value;
  
  // Clear previous results and show status
  $('onboarding-panel').classList.add('hidden');
  $('status').classList.remove('hidden');
  $('result').classList.add('hidden');
  $('error').textContent = '';
  $('stage').textContent = 'Connecting...';
  $('percent').textContent = 'Live';
  $('progress').value = 100;
  $('turns').innerHTML = '<div id="live-transcript" class="turn speaker-customer"><em>Listening... Speak into your microphone.</em></div>';
  $('result').classList.remove('hidden');

  // Diagnostics init
  liveChunkCount = 0;
  liveStartTime = null;
  $('diag-duration').textContent = '0.0s';
  $('diag-chunks').textContent = '0';
  $('diag-latency').textContent = '--ms';
  $('diag-language').textContent = language.split('-')[0].toUpperCase();

  liveInterval = setInterval(() => {
    const elapsed = liveStartTime ? ((Date.now() - liveStartTime) / 1000).toFixed(1) : '0.0';
    $('diag-duration').textContent = elapsed + 's';
  }, 200);

  const chunkMs = Number($('chunk').value) || 0;
  let scriptBufferSize = 2048;
  if (chunkMs === 0 || chunkMs <= 80) {
    scriptBufferSize = 1024; // ~64ms capture latency
  } else if (chunkMs <= 160) {
    scriptBufferSize = 2048; // ~128ms capture latency
  } else if (chunkMs <= 320) {
    scriptBufferSize = 4096; // ~256ms capture latency
  } else if (chunkMs <= 560) {
    scriptBufferSize = 8192;
  } else {
    scriptBufferSize = 16384;
  }

  const loc = window.location;
  const wsProtocol = loc.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProtocol}//${loc.host}/ws/stt?language-code=${language}&model=${encodeURIComponent(model)}&mode=transcribe&sample_rate=16000&vad_signals=true&flush_signal=false&input_audio_codec=pcm_s16le&binary_audio=1&call_code=frontend-session-${Date.now()}`;
  
  ws = new WebSocket(wsUrl);
  
  ws.onopen = async () => {
    $('stage').textContent = 'Streaming Live';
    
    try {
      // Start microphone recording and audio context
      audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(audioStream);
      
      // Create script processor for 16kHz mono audio
      audioProcessor = audioContext.createScriptProcessor(scriptBufferSize, 1, 1);
      source.connect(audioProcessor);
      audioProcessor.connect(audioContext.destination);
      
      const sampleRate = 16000;
      const targetSamples = chunkMs > 0 ? Math.round(sampleRate * (chunkMs / 1000)) : 0;
      let sampleBuffer = [];
      
      audioProcessor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        
        if (targetSamples > 0) {
          // Accumulate samples for exact-size chunking
          for (let i = 0; i < inputData.length; i++) {
            sampleBuffer.push(inputData[i]);
          }
          
          while (sampleBuffer.length >= targetSamples) {
            const chunk = sampleBuffer.splice(0, targetSamples);
            const pcm16 = new Int16Array(chunk.length);
            for (let i = 0; i < chunk.length; i++) {
              pcm16[i] = Math.min(1, Math.max(-1, chunk[i])) * 0x7FFF;
            }
            if (ws && ws.readyState === WebSocket.OPEN) {
              if (!liveStartTime) {
                liveStartTime = Date.now();
              }
              ws.send(pcm16.buffer);
              liveChunkCount++;
              $('diag-chunks').textContent = liveChunkCount;
            }
          }
        } else {
          // No buffering: send raw script processor chunk immediately
          const pcm16 = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            pcm16[i] = Math.min(1, Math.max(-1, inputData[i])) * 0x7FFF;
          }
          if (ws && ws.readyState === WebSocket.OPEN) {
            if (!liveStartTime) {
              liveStartTime = Date.now();
            }
            ws.send(pcm16.buffer);
            liveChunkCount++;
            $('diag-chunks').textContent = liveChunkCount;
          }
        }
      };
    } catch (err) {
      $('error').textContent = 'Microphone access failed: ' + err.message;
      stopWebSocketStream();
    }
  };

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.event === 'transcript') {
      const formattedText = data.text ? data.text.trim() : '';
      if (formattedText) {
        const speakerClass = data.speaker === 'customer' ? 'speaker-customer' : 'speaker-agent';
        const speakerName = data.speaker === 'customer' ? 'Customer' : 'Agent';
        const elapsedSinceStart = liveStartTime ? (Date.now() - liveStartTime) / 1000 : 0;
        const latencySec = elapsedSinceStart - Number(data.end);
        const latencyMs = Math.max(0, Math.round(latencySec * 1000));
        
        $('diag-latency').textContent = `${latencyMs}ms`;
        
        const timeRange = `${Number(data.start).toFixed(2)}–${Number(data.end).toFixed(2)}s (latency: ${latencyMs}ms)`;
        
        const turnDiv = document.getElementById('live-transcript');
        if (turnDiv) {
          turnDiv.className = `turn ${speakerClass}`;
          turnDiv.innerHTML = `
            <div class="msg-bubble">
              <div class="msg-header">
                <strong>${esc(speakerName)}</strong>
                <span>${timeRange}</span>
              </div>
              <div class="msg-content">
                <p>${esc(formattedText)}</p>
              </div>
            </div>
          `;
          
          // If final, create a new live transcript container and move on
          if (data.final) {
            turnDiv.removeAttribute('id');
            const newLiveDiv = document.createElement('div');
            newLiveDiv.id = 'live-transcript';
            newLiveDiv.className = 'turn speaker-customer';
            newLiveDiv.innerHTML = '<em>Listening...</em>';
            $('turns').appendChild(newLiveDiv);
          }
        }
      }
    }
  };

  ws.onerror = (err) => {
    console.error('WebSocket connection error details:', err);
    $('error').textContent = `WebSocket connection error. Target URL: ${wsUrl}. If accessing via HTTPS, browser policy requires secure WebSockets (wss://). Ensure port 8000 allows WebSocket connections and is not blocked by a firewall or mixed-content policies.`;
  };

  ws.onclose = () => {
    $('stage').textContent = 'Stream Closed';
    $('percent').textContent = '100%';
    stopAudioCapture();
  };
}

function stopAudioCapture() {
  if (liveInterval) {
    clearInterval(liveInterval);
    liveInterval = null;
  }
  if (audioProcessor) {
    audioProcessor.disconnect();
    audioProcessor = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (audioStream) {
    audioStream.getTracks().forEach(t => t.stop());
    audioStream = null;
  }
}

// Stop WebSocket recording
function stopWebSocketStream() {
  if (ws) {
    ws.close();
    ws = null;
  }
  stopAudioCapture();
  isMuted = false;
  $('mute').disabled = true;
  $('mute').textContent = 'Mute';
}

$('record').onclick = async () => {
  const mode = $('stream-mode').value;
  $('record').disabled = true;
  $('stop').disabled = false;
  
  if (mode === 'websocket') {
    $('mute').disabled = false;
    isMuted = false;
    $('mute').textContent = 'Mute';
    await startWebSocketStream();
  } else {
    $('mute').disabled = true;
    // Original Batch MediaRecorder Mode
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.start();
  }
};

$('stop').onclick = () => {
  const mode = $('stream-mode').value;
  $('record').disabled = false;
  $('stop').disabled = true;
  $('mute').disabled = true;
  
  if (mode === 'websocket') {
    stopWebSocketStream();
  } else if (recorder) {
    recorder.stop();
  }
};

$('mute').onclick = () => {
  if (!audioStream) return;
  isMuted = !isMuted;
  audioStream.getAudioTracks().forEach(track => {
    track.enabled = !isMuted;
  });
  if (isMuted) {
    $('mute').textContent = '🔇 Unmute';
  } else {
    $('mute').textContent = 'Mute';
  }
};

document.addEventListener('click', e => {
  if (e.target.id === 'stop' && recorder && $('stream-mode').value === 'batch') {
    recorder.onstop = () => {
      recorder.stream.getTracks().forEach(t => t.stop());
      submitFile(new File(chunks, 'recording.webm', { type: recorder.mimeType }));
    };
  }
});

$('file').onchange = e => {
  if (e.target.files.length > 1) {
    submitBatchFiles(Array.from(e.target.files));
  } else if (e.target.files[0]) {
    submitFile(e.target.files[0]);
  }
};

function options(fd) {
  fd.append('language', $('language').value);
  fd.append('chunk_ms', $('chunk').value);
  fd.append('itn_backend', $('itn').value);
  fd.append('model', $('model').value);
  [['name', 'bias-name'], ['institute_name', 'bias-institute'], ['total_due', 'bias-amount'], ['due_date', 'bias-date']].forEach(([field, id]) => {
    const v = $(id).value.trim();
    if (v) fd.append(field, v);
  });
  return fd;
}

async function submitFile(file) {
  const fd = options(new FormData());
  fd.append('file', file);
  start(await fetch('/api/jobs', { method: 'POST', body: fd }).then(r => r.json()));
}

async function submitSample(name) {
  start(await fetch('/api/jobs/sample/' + encodeURIComponent(name), { method: 'POST', body: options(new FormData()) }).then(r => r.json()));
}

function start(job) {
  $('onboarding-panel').classList.add('hidden');
  $('status').classList.remove('hidden');
  $('result').classList.add('hidden');
  $('error').textContent = '';
  poll(job.id);
}

async function poll(id) {
  const job = await fetch('/api/jobs/' + id).then(r => r.json());
  $('stage').textContent = job.stage;
  $('percent').textContent = job.progress + '%';
  $('progress').value = job.progress;
  if (job.status === 'complete') return show(job.result);
  if (job.status === 'failed') {
    $('error').textContent = job.error;
    return;
  }
  setTimeout(() => poll(id), 1500);
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}

function spanList(spans) {
  if (!spans?.length) return '';
  return `<details style="border:none; margin:0; padding:0;"><summary style="font-size:10px; font-weight:700; color:var(--text-light); margin-top:4px;">${spans.length} ITN span${spans.length === 1 ? '' : 's'}</summary><div class="spans" style="display:grid; gap:4px; margin-top:4px;">${spans.map(s => `<code style="display:block; font-size:10px; background:#f4f4f2; padding:4px; border-radius:4px;">${esc(s.raw)} → ${esc(s.canonical)} <small style="color:var(--text-light);">${esc(s.cls)} · ${esc(s.rule_id)}</small></code>`).join('')}</div></details>`;
}

function turnHtml(t, backend) {
  const normalized = t.canonical_text ?? t.text ?? '';
  const raw = t.text ?? '';
  const changed = normalized !== raw;
  
  const speakerClass = t.speaker === 'customer' ? 'speaker-customer' : 'speaker-agent';
  const speakerName = t.speaker === 'customer' ? 'Customer' : 'Agent';
  const timeRange = `${Number(t.start_sec).toFixed(2)}–${Number(t.end_sec).toFixed(2)}s`;
  
  const rawASR = changed ? `<p class="raw-asr-text"><small style="font-weight:700; font-size:10px;">Raw ASR: </small>${esc(raw)}</p>` : '';
  const compare = backend === 'compare' ? `
    <div class="compare" style="margin-top:8px; display:grid; grid-template-columns:1fr 1fr; gap:8px;">
      <div style="background:#f7f7f5; padding:6px; border-radius:6px;"><small style="font-size:9px; color:var(--text-light); font-weight:800; text-transform:uppercase;">Custom</small><p style="font-size:12px; margin-top:2px;">${esc(t.custom_canonical_text ?? normalized)}</p></div>
      <div style="background:#f7f7f5; padding:6px; border-radius:6px;"><small style="font-size:9px; color:var(--text-light); font-weight:800; text-transform:uppercase;">NeMo</small><p style="font-size:12px; margin-top:2px;">${esc(t.nemo_canonical_text ?? raw)}</p></div>
    </div>` : '';
    
  return `
    <div class="turn ${speakerClass}">
      <div class="msg-bubble">
        <div class="msg-header">
          <strong>${esc(speakerName)}</strong>
          <span>${timeRange}</span>
        </div>
        <div class="msg-content">
          <p>${esc(normalized)}</p>
          ${rawASR}
          ${compare}
          ${spanList(t.spans)}
          ${t.itn_error ? `<p class="itn-error">${esc(t.itn_error)}</p>` : ''}
        </div>
      </div>
    </div>
  `;
}

function biasingSummary(b) {
  if (!b) return 'off';
  if (b.applied) return `on · ${b.dynamic_phrases} phrases`;
  return `off · ${b.reason || b.error || 'n/a'}`;
}

function metricHtml(k, v) {
  if (k === 'biasing') {
    const title = v && v.top_phrases ? ` title="${esc(v.top_phrases.join(', '))}"` : '';
    return `<div${title} class="health-row"><span>biasing</span><strong>${esc(biasingSummary(v))}</strong></div>`;
  }
  return `<div class="health-row"><span>${esc(k.replaceAll('_', ' '))}</span><strong>${esc(v)}</strong></div>`;
}

function show(r) {
  $('onboarding-panel').classList.add('hidden');
  $('result').classList.remove('hidden');
  $('original').src = r.original_url;
  $('denoised').src = r.denoised_url;
  
  // Show audio players for batch/file modes
  if ($('stream-mode').value === 'batch') {
    $('audio-players-container').classList.remove('hidden');
  } else {
    $('audio-players-container').classList.add('hidden');
  }
  
  // Diagnostics update
  if (r.metrics) {
    if (r.metrics.duration_seconds !== undefined) {
      $('diag-duration').textContent = Number(r.metrics.duration_seconds).toFixed(1) + 's';
    }
    if (r.metrics.diarization_seconds !== undefined) {
      // Show total processing chunks / diarization stats as mock indicator
      $('diag-chunks').textContent = 'Batch processed';
    }
    if (r.metrics.denoise_seconds !== undefined) {
      $('diag-latency').textContent = Number(r.metrics.denoise_seconds * 1000).toFixed(0) + 'ms';
    }
    $('diag-language').textContent = $('language').value.split('-')[0].toUpperCase();
  }

  $('metrics').innerHTML = `
    <div class="pipeline-health-card" style="margin-top:10px;">
      <div class="health-header"><h4>Detailed Metrics</h4></div>
      <div class="health-details">
        ${Object.entries(r.metrics).map(([k, v]) => metricHtml(k, v)).join('')}
      </div>
    </div>
  `;
  
  $('turns').innerHTML = r.turns.filter(t => (t.canonical_text ?? t.text ?? '').trim()).map(t => turnHtml(t, r.itn_backend || 'custom')).join('');
}

let evaluationData = null;
function hitBadge(label, baseline, biased) {
  const state = biased ? "hit" : baseline ? "lost" : "miss";
  return "<span class=\"hit " + state + "\">" + esc(label) + ": " + (baseline ? "base ✓" : "base –") + " / " + (biased ? "biased ✓" : "biased –") + "</span>";
}
function dialogueHtml(turns) {
  if (!turns || !turns.length) return "<p class=\"muted\">No speech detected.</p>";
  return "<div class=\"dialogue\">" + turns.map(t => {
    const customer = t.speaker === "customer";
    return "<div class=\"msg " + (customer ? "customer" : "agent") + "\"><span class=\"who\">" + (customer ? "Customer" : "Agent") + "</span><p>" + esc(t.text) + "</p></div>";
  }).join("") + "</div>";
}
function showEvaluationRow(row) {
  document.querySelectorAll(".eval-row").forEach(x => x.classList.toggle("active", x.dataset.idx === String(row.idx)));
  document.getElementById("eval-detail").innerHTML = "<div class=\"detail-title\"><div><small>ROW " + esc(row.idx) + " · " + esc(row.institute) + "</small><h3>" + esc(row.name) + "</h3></div><span class=\"bias-state " + (row.biasing_applied ? "on" : "off") + "\">" + (row.biasing_applied ? "Biasing applied" : "Biasing unavailable") + "</span></div><div class=\"hit-line\">" + hitBadge("Name", row.name_hit_baseline, row.name_hit_biased) + hitBadge("Brand", row.brand_hit_baseline, row.brand_hit_biased) + "</div>" + (row.recording_url ? "<div class=\"recording\"><span class=\"who\">Call recording</span><audio controls preload=\"none\" src=\"" + esc(row.recording_url) + "\"></audio></div>" : "") + "<div class=\"transcript-compare\"><section><h4>Baseline</h4>" + dialogueHtml(row.baseline_turns) + "</section><section><h4>Context biased</h4>" + dialogueHtml(row.biased_turns) + "</section></div>";
}
function renderEvaluation() {
  if (!evaluationData) return;
  const filter = document.getElementById("eval-filter").value;
  const rows = evaluationData.rows.filter(r => filter === "all" || (filter === "improved" && ((!r.name_hit_baseline && r.name_hit_biased) || (!r.brand_hit_baseline && r.brand_hit_biased))) || (filter === "missed" && !r.name_hit_biased && !r.brand_hit_biased) || (filter === "applied" && r.biasing_applied));
  const box = document.getElementById("eval-rows");
  box.innerHTML = rows.map(r => "<button class=\"eval-row\" data-idx=\"" + esc(r.idx) + "\"><span><strong>" + esc(r.name) + "</strong><small>" + esc(r.brand) + " · ₹" + esc(r.total_due || "0") + "</small></span><span class=\"row-hits\">" + (r.name_hit_biased ? "N" : "–") + (r.brand_hit_biased ? "B" : "–") + "</span></button>").join("") || "<p class=\"muted\">No rows match this filter.</p>";
  box.querySelectorAll(".eval-row").forEach(button => button.onclick = () => showEvaluationRow(evaluationData.rows.find(r => String(r.idx) === button.dataset.idx)));
  if (rows.length) showEvaluationRow(rows[0]);
}
async function loadEvaluation() {
  const res = await fetch("/api/evaluations/Result_8_hindi_biasing_normalized.csv");
  if (!res.ok) {
    document.getElementById("eval-summary").innerHTML = "<p class=\"itn-error\">Evaluation CSV is not available yet.</p>";
    return;
  }
  evaluationData = await res.json();
  const s = evaluationData.summary;
  document.getElementById("eval-summary").innerHTML = [["Rows", s.total], ["Biasing applied", s.biasing_applied + " / " + s.total], ["Name recall", s.name_hit_baseline + " → " + s.name_hit_biased], ["Brand recall", s.brand_hit_baseline + " → " + s.brand_hit_biased]].map(x => "<div><span>" + x[0] + "</span><strong>" + x[1] + "</strong></div>").join("");
  renderEvaluation();
}
document.getElementById("eval-filter").onchange = renderEvaluation;
document.getElementById("eval-refresh").onclick = loadEvaluation;
loadSamples();
loadEvaluation();

async function submitBatchFiles(files) {
  $('onboarding-panel').classList.add('hidden');
  $('status').classList.add('hidden');
  $('result').classList.add('hidden');
  $('batch-jobs-panel').classList.remove('hidden');
  
  $('download-csv-btn').disabled = true;
  $('batch-stage').textContent = 'Submitting...';
  $('batch-percent').textContent = '0%';
  $('batch-progress').value = 0;
  
  const tbody = $('batch-files-tbody');
  tbody.innerHTML = '';
  
  const fileTracks = files.map((file, idx) => {
    const rowId = `batch-row-${idx}`;
    const tr = document.createElement('tr');
    tr.id = rowId;
    tr.style.borderBottom = '1px solid var(--border-color)';
    tr.innerHTML = `
      <td style="padding: 12px 15px; font-weight: 600; color: var(--text-dark);">${esc(file.name)}</td>
      <td class="batch-file-stage" style="padding: 12px 15px; color: var(--text-light);">Queued</td>
      <td class="batch-file-progress" style="padding: 12px 15px; font-weight: 700;">0%</td>
      <td class="batch-file-preview" style="padding: 12px 15px; color: var(--text-light); font-style: italic;">Waiting...</td>
    `;
    tbody.appendChild(tr);
    
    return {
      file: file,
      rowId: rowId,
      jobId: null,
      status: 'queued',
      stage: 'queued',
      progress: 0,
      transcript: '',
      error: ''
    };
  });
  
  for (let track of fileTracks) {
    const fd = options(new FormData());
    fd.append('file', track.file);
    try {
      const job = await fetch('/api/jobs', { method: 'POST', body: fd }).then(r => r.json());
      track.jobId = job.id;
      track.status = job.status;
      track.stage = job.stage;
      track.progress = job.progress;
      updateBatchRow(track);
    } catch (err) {
      track.status = 'failed';
      track.stage = 'failed';
      track.error = 'Submission failed: ' + err.message;
      updateBatchRow(track);
    }
  }
  
  pollBatch(fileTracks);
}

function updateBatchRow(track) {
  const row = $(track.rowId);
  if (!row) return;
  
  const stageCell = row.querySelector('.batch-file-stage');
  const progressCell = row.querySelector('.batch-file-progress');
  const previewCell = row.querySelector('.batch-file-preview');
  
  stageCell.textContent = track.stage;
  progressCell.textContent = track.progress + '%';
  
  if (track.status === 'complete') {
    stageCell.innerHTML = `<span style="color: #10b981; font-weight: 600;">Complete</span>`;
    const cleanUrl = `/api/jobs/${track.jobId}/audio/normalized`;
    const previewText = track.transcript ? track.transcript.substring(0, 80) + '...' : 'Empty transcript';
    previewCell.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 4px;">
        <span style="color: var(--text-dark);">${esc(previewText)}</span>
        <a href="${cleanUrl}" target="_blank" style="font-size: 11px; color: var(--color-primary, #6366f1); font-weight: 600; text-decoration: none;">🔊 Open Audio</a>
      </div>
    `;
  } else if (track.status === 'failed') {
    stageCell.innerHTML = `<span style="color: #ef4444; font-weight: 600;">Failed</span>`;
    previewCell.innerHTML = `<span style="color: #ef4444;">${esc(track.error)}</span>`;
  } else {
    previewCell.textContent = 'Processing...';
  }
}

async function pollBatch(fileTracks) {
  let allDone = false;
  
  while (!allDone) {
    allDone = true;
    let completedCount = 0;
    let totalProgress = 0;
    
    for (let track of fileTracks) {
      if (track.status === 'complete' || track.status === 'failed') {
        completedCount++;
        totalProgress += 100;
        continue;
      }
      
      if (!track.jobId) {
        allDone = false;
        continue;
      }
      
      try {
        const job = await fetch('/api/jobs/' + track.jobId).then(r => r.json());
        track.status = job.status;
        track.stage = job.stage;
        track.progress = job.progress;
        
        if (job.status === 'complete') {
          track.transcript = job.result ? (job.result.transcript || '') : '';
        } else if (job.status === 'failed') {
          track.error = job.error || 'Unknown ASR error';
        }
        
        updateBatchRow(track);
        
        if (track.status !== 'complete' && track.status !== 'failed') {
          allDone = false;
        } else {
          completedCount++;
        }
        totalProgress += track.progress;
      } catch (err) {
        track.status = 'failed';
        track.stage = 'failed';
        track.error = 'Polling failed: ' + err.message;
        updateBatchRow(track);
        completedCount++;
        totalProgress += 100;
      }
    }
    
    const overallProgress = Math.round(totalProgress / fileTracks.length);
    $('batch-progress').value = overallProgress;
    $('batch-percent').textContent = overallProgress + '%';
    $('batch-stage').textContent = allDone ? 'Complete' : `Processing (${completedCount}/${fileTracks.length} done)`;
    
    if (!allDone) {
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
  }
  
  const jobIds = fileTracks.filter(t => t.jobId).map(t => t.jobId).join(',');
  if (jobIds) {
    $('download-csv-btn').disabled = false;
    $('download-csv-btn').onclick = () => {
      window.location.href = `/api/jobs/batch/csv?job_ids=${encodeURIComponent(jobIds)}`;
    };
  }
}
