// WebRTC client for text-only realtime
let pc, dc, localStream;
// Track transcription completion so we can wait before closing
let transcriptionWaiter = null;
// Track generation completion as well when auto-generation is enabled
let generationWaiter = null;
// Flag to avoid double stop; continue accepting messages until done
let isShuttingDown = false;

function resetTranscriptionWaiter() {
  let resolveFn;
  const promise = new Promise((resolve) => { resolveFn = resolve; });
  transcriptionWaiter = { promise, resolve: resolveFn, done: false };
}

function resolveTranscriptionIfPending() {
  if (transcriptionWaiter && !transcriptionWaiter.done) {
    transcriptionWaiter.done = true;
    try { transcriptionWaiter.resolve(); } catch (_) {}
  }
}

function resetGenerationWaiter() {
  let resolveFn;
  const promise = new Promise((resolve) => { resolveFn = resolve; });
  generationWaiter = { promise, resolve: resolveFn, done: false };
}

function resolveGenerationIfPending() {
  if (generationWaiter && !generationWaiter.done) {
    generationWaiter.done = true;
    try { generationWaiter.resolve(); } catch (_) {}
  }
}
let cachedSession = null;
let cachedSessionFetchedAt = 0;
let isRecording = false;
let startTime, timerInterval;
// During shutdown, skip further transcription deltas but still accept response.*
let suppressTranscriptionDuringShutdown = false;

// Debug logging support
const urlParams = new URLSearchParams(window.location.search);
const DEBUG_RTC = urlParams.get('debug') === '1';
const __rtcLogs = [];
function logRTC(...args) {
  if (!DEBUG_RTC) return;
  try {
    const ts = new Date().toISOString();
    __rtcLogs.push([ts, ...args]);
    // Also mirror to console for live inspection
    console.debug('[RTC]', ts, ...args);
  } catch (_) {}
}
// Expose helpers for export and inspection
window.dumpRealtimeLogs = function() {
  try {
    const blob = new Blob([JSON.stringify(__rtcLogs)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `realtime_logs_${Date.now()}.json`;
    a.click();
  } catch (e) {
    console.error('Failed to dump logs', e);
  }
};

const transcript = document.getElementById('transcript');
const enhancedTranscript = document.getElementById('enhancedTranscript');
const copyButton = document.getElementById('copyButton');
const copyEnhancedButton = document.getElementById('copyEnhancedButton');
const recordButton = document.getElementById('recordButton');
const readabilityButton = document.getElementById('readabilityButton');
const correctnessButton = document.getElementById('correctnessButton');
const promptInput = document.getElementById('promptInput');

function updateConnectionStatus(status) {
  const statusDot = document.getElementById('connectionStatus');
  statusDot.classList.remove('connected', 'connecting', 'idle');
  switch (status) {
    case 'connected':
      statusDot.classList.add('connected');
      statusDot.style.backgroundColor = '#34C759';
      break;
    case 'connecting':
      statusDot.classList.add('connecting');
      statusDot.style.backgroundColor = '#FF9500';
      break;
    case 'idle':
      statusDot.classList.add('idle');
      statusDot.style.backgroundColor = '#007AFF';
      break;
    default:
      statusDot.style.backgroundColor = '#FF3B30';
  }
}

function startTimer() {
  clearInterval(timerInterval);
  document.getElementById('timer').textContent = '00:00';
  startTime = Date.now();
  timerInterval = setInterval(() => {
    const elapsed = Date.now() - startTime;
    const minutes = Math.floor(elapsed / 60000);
    const seconds = Math.floor((elapsed % 60000) / 1000);
    document.getElementById('timer').textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
}

async function fetchEphemeralSession() {
  const modelSelect = document.getElementById('modelSelect');
  const selectedModel = modelSelect ? modelSelect.value : 'gpt-realtime';
  const resp = await fetch('/api/v1/realtime/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: selectedModel })
  });
  if (!resp.ok) throw new Error('Failed to create realtime session');
  const json = await resp.json();
  cachedSession = json?.data || null;
  cachedSessionFetchedAt = Date.now();
  return cachedSession;
}

async function prefetchSessionAndIndicateReady() {
  try {
    updateConnectionStatus('connecting');
    await fetchEphemeralSession();
    updateConnectionStatus('idle');
  } catch (_) {
    // stay red if prefetch fails
  }
}

async function ensureLocalStream() {
  // Reuse only if an active audio track exists
  if (localStream) {
    const tracks = localStream.getAudioTracks();
    const hasLiveTrack = tracks.length > 0 && tracks[0].readyState === 'live' && tracks[0].enabled;
    logRTC('ensureLocalStream: existing stream check', { tracks: tracks.length, hasLiveTrack, trackState: tracks[0] && tracks[0].readyState });
    if (hasLiveTrack) return localStream;
    // Clean up any ended tracks
    try { tracks.forEach(t => { try { t.stop(); } catch(_) {} }); } catch(_) {}
    localStream = null;
  }
  localStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
  });
  logRTC('ensureLocalStream: acquired new stream', { tracks: localStream.getAudioTracks().length });
  return localStream;
}

async function startRecording() {
  if (isRecording) return;
  try {
    transcript.value = '';
    enhancedTranscript.value = '';
    logRTC('startRecording: begin');
    const data = await fetchEphemeralSession();
    logRTC('startRecording: fetched session', { fetchedAt: Date.now(), hasClientSecret: !!(data && (data.client_secret || data.clientSecret)) });
    // Token obtained → show connecting (yellow)
    updateConnectionStatus('connecting');

    // Build RTCPeerConnection
    pc = new RTCPeerConnection({ iceServers: [] });
    logRTC('RTCPeerConnection created');
    dc = pc.createDataChannel('oai-events');
    logRTC('DataChannel created');

    dc.onopen = () => {
      logRTC('DataChannel onopen');
      updateConnectionStatus('connected');
      // Session-level instructions are set by the server. No response.create here.
      // We rely on input_audio_transcription.* deltas for live text.
    };

    dc.onmessage = (event) => {
      // During shutdown we still accept response events to flush tail
      logRTC('DataChannel onmessage', { size: (event.data && event.data.length) || 0, preview: String(event.data).slice(0, 120) });
      try {
        const msg = JSON.parse(event.data);
        logRTC('Parsed message', { type: msg.type });
        // Ignore transcription messages if we're shutting down, but keep response.*
        if (suppressTranscriptionDuringShutdown && typeof msg.type === 'string' && msg.type.startsWith('conversation.item.input_audio_transcription')) {
          logRTC('Ignoring transcription message during shutdown');
          return;
        }
        switch (msg.type) {
          case 'input_audio_buffer.speech_stopped': {
            logRTC('speech_stopped received');
            break;
          }
          case 'conversation.item.input_audio_transcription.delta': {
            // Response-only mode: ignore live transcription deltas
            logRTC('Ignoring transcription delta (response-only mode)');
            break;
          }
          case 'response.created': {
            logRTC('Response created');
            // Start fresh for generated output in the upper box
            transcript.value = '';
            break;
          }
          case 'response.text.delta': {
            const text = msg.delta || msg.text || '';
            if (text) {
              transcript.value += text;
              transcript.scrollTop = transcript.scrollHeight;
              logRTC('Appended response.text.delta to transcript', { len: text.length, tailPreview: text.slice(-20) });
            }
            break;
          }
          case 'response.text.done': {
            logRTC('response.text.done received');
            break;
          }
          case 'response.done': {
            logRTC('response.done received');
            resolveGenerationIfPending();
            break;
          }
          case 'conversation.item.input_audio_transcription.completed': {
            logRTC('Transcription completed message received');
          resolveTranscriptionIfPending();
            // Ignore completed append to avoid double printing; deltas already appended
            break;
          }
          case 'error':
            logRTC('Error message received', { error: msg.error });
            alert(msg.error?.message || 'Realtime error');
            break;
        }
      } catch (_) {}
    };

    const stream = await ensureLocalStream();
    logRTC('Got local stream, adding tracks');
    for (const track of stream.getTracks()) {
      logRTC('Adding track', { kind: track.kind, readyState: track.readyState });
      pc.addTrack(track, stream);
    }

    const offer = await pc.createOffer();
    logRTC('Created offer', { sdpLen: (offer.sdp && offer.sdp.length) || 0 });
    await pc.setLocalDescription(offer);

    // Send SDP to OpenAI; use ephemeral client_secret
    const clientSecret = data?.client_secret?.value || data?.client_secret || data?.clientSecret || '';
    if (!clientSecret) throw new Error('No ephemeral client secret returned');

    const sdpResp = await fetch(`https://api.openai.com/v1/realtime?model=${encodeURIComponent(data?.model || 'gpt-realtime')}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${clientSecret}`,
        'Content-Type': 'application/sdp',
        'Accept': 'application/sdp'
      },
      body: offer.sdp
    });
    logRTC('Sent offer to OpenAI, awaiting answer', { status: sdpResp.status });
    if (!sdpResp.ok) throw new Error('Failed to exchange SDP with OpenAI');
    const answerSdp = await sdpResp.text();
    logRTC('Received answer', { sdpLen: answerSdp.length });
    await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });

    isRecording = true;
    isShuttingDown = false;
    suppressTranscriptionDuringShutdown = false;
    startTimer();
    recordButton.textContent = 'Stop';
    recordButton.classList.add('recording');
    // Prepare waiter for completion for this session
    resetTranscriptionWaiter();
    resetGenerationWaiter();
    logRTC('startRecording: success');
  } catch (err) {
    console.error('Start error', err);
    logRTC('startRecording: error', { message: err && err.message });
    alert(err.message || 'Failed to start WebRTC session');
    updateConnectionStatus('idle');
  }
}

async function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  try {
    logRTC('stopRecording: begin');
    isShuttingDown = true;
    suppressTranscriptionDuringShutdown = true;
    // Short drain so the last frames reach the server
    await new Promise(r => setTimeout(r, 300));
    // Commit buffered audio to avoid truncating tail
    if (dc && dc.readyState === 'open') {
      try {
        dc.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
        logRTC('Sent input_audio_buffer.commit');
      } catch(_) { logRTC('Failed to send commit'); }
    }
    // Give the commit a brief moment to propagate
    await new Promise(r => setTimeout(r, 150));
    // Explicitly trigger a response to get generated text immediately
    if (dc && dc.readyState === 'open') {
      try {
        dc.send(JSON.stringify({ type: 'response.create' }));
        logRTC('Sent response.create');
      } catch (_) { logRTC('Failed to send response.create'); }
    }
    // Stop sending audio after commit
    if (pc) {
      pc.getSenders().forEach(s => { try { s.track && s.track.stop(); logRTC('Stopped sender track', { kind: s.track && s.track.kind }); } catch(_){} });
    }
    // Wait for transcription and generation to finish with a bounded timeout
    try {
      const timeoutMs = 5000;
      const timeout = new Promise(r => setTimeout(r, timeoutMs));
      const waits = [];
      if (transcriptionWaiter && !transcriptionWaiter.done) waits.push(transcriptionWaiter.promise);
      if (generationWaiter && !generationWaiter.done) waits.push(generationWaiter.promise);
      if (waits.length > 0) {
        await Promise.race([Promise.allSettled(waits), timeout]);
        logRTC('stopRecording: waited for completion or timeout', { waitedMs: timeoutMs, transcriptionDone: transcriptionWaiter?.done, generationDone: generationWaiter?.done });
      }
    } catch (_) {}
    if (pc) { try { pc.close(); } catch(_){} }
    logRTC('stopRecording: closed pc');
  } catch(_) {}
  stopTimer();
  recordButton.textContent = 'Start';
  recordButton.classList.remove('recording');
  updateConnectionStatus('idle');
  // Ensure a fresh stream will be obtained next start
  localStream = null;
  logRTC('stopRecording: done');
}

recordButton.onclick = () => isRecording ? stopRecording() : startRecording();
copyButton.onclick = async () => {
  try { await navigator.clipboard.writeText(transcript.value || ''); } catch(_) {}
};
copyEnhancedButton.onclick = async () => {
  try { await navigator.clipboard.writeText(enhancedTranscript.value || ''); } catch(_) {}
};

// Reuse existing REST actions
readabilityButton.onclick = async () => {
  const inputText = transcript.value.trim();
  if (!inputText) return alert('No text to enhance.');
  try {
    const response = await fetch('/api/v1/readability', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: inputText }) });
    if (!response.ok) throw new Error('Readability failed');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      fullText += decoder.decode(value, { stream: true });
      enhancedTranscript.value = fullText;
      enhancedTranscript.scrollTop = enhancedTranscript.scrollHeight;
    }
  } catch (e) {
    alert(e.message || 'Readability error');
  }
};

correctnessButton.onclick = async () => {
  const inputText = transcript.value.trim();
  if (!inputText) return alert('No text to check.');
  try {
    const response = await fetch('/api/v1/correctness', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: inputText }) });
    if (!response.ok) throw new Error('Correctness failed');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      fullText += decoder.decode(value, { stream: true });
      enhancedTranscript.value = fullText;
      enhancedTranscript.scrollTop = enhancedTranscript.scrollHeight;
    }
  } catch (e) {
    alert(e.message || 'Correctness error');
  }
};

// Theme toggle reuse
function toggleTheme() {
  const body = document.body;
  const themeToggle = document.getElementById('themeToggle');
  const isDarkTheme = body.classList.toggle('dark-theme');
  themeToggle.textContent = isDarkTheme ? '☀️' : '🌙';
  localStorage.setItem('darkTheme', isDarkTheme);
}
function initializeTheme() {
  const darkTheme = localStorage.getItem('darkTheme') === 'true';
  const themeToggle = document.getElementById('themeToggle');
  if (darkTheme) {
    document.body.classList.add('dark-theme');
    themeToggle.textContent = '☀️';
  }
}
document.getElementById('themeToggle').onclick = toggleTheme;
document.addEventListener('DOMContentLoaded', async () => {
  initializeTheme();
  // Initial state is idle (blue). We no longer prefetch to avoid token waste.
  updateConnectionStatus('idle');
  if (DEBUG_RTC) {
    console.info('Realtime debug logging ENABLED. Use window.dumpRealtimeLogs() to export logs.');
  }
  
  // Clear cached session when model changes
  const modelSelect = document.getElementById('modelSelect');
  if (modelSelect) {
    modelSelect.addEventListener('change', () => {
      cachedSession = null;
      cachedSessionFetchedAt = 0;
      logRTC('Model changed, cleared cached session');
    });
  }
});


// Handle spacebar toggle (avoid inputs/content editable, respect disabled)
document.addEventListener('keydown', (event) => {
  if (event.code === 'Space' && recordButton) {
    const activeElement = document.activeElement;
    const inEditable = (activeElement && (activeElement.tagName && activeElement.tagName.match(/INPUT|TEXTAREA/) || activeElement.isContentEditable));
    if (!inEditable && !recordButton.disabled) {
      event.preventDefault();
      recordButton.click();
      logRTC('Spacebar toggled record button');
    }
  }
});


