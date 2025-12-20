// Global state
let ws, audioContext, processor, source, stream;
let isRecording = false;
let isReplaying = false;
let isStopping = false; // Flag to allow final audio processing during stop
let timerInterval;
let startTime;
let audioBuffer = new Int16Array(0);
let wsConnected = false;
let streamInitialized = false;
let isAutoStarted = false;

// IndexedDB state
let db = null;
let currentSessionId = null;
let sessionStartTime = null;
let chunkSeq = 0;
let storageAvailable = false;

// DOM elements
const recordButton = document.getElementById('recordButton');
const replayButton = document.getElementById('replayButton');
const storageStatus = document.getElementById('storageStatus');
const transcript = document.getElementById('transcript');
const enhancedTranscript = document.getElementById('enhancedTranscript');
const copyButton = document.getElementById('copyButton');
const copyEnhancedButton = document.getElementById('copyEnhancedButton');
const readabilityButton = document.getElementById('readabilityButton');
const askAIButton = document.getElementById('askAIButton');
const correctnessButton = document.getElementById('correctnessButton');

// Configuration
const targetSeconds = 5;
const urlParams = new URLSearchParams(window.location.search);
const autoStart = urlParams.get('start') === '1';

// Utility functions
const isMobileDevice = () => /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

async function copyToClipboard(text, button) {
    if (!text) return;
    try {
        await navigator.clipboard.writeText(text);
        showCopiedFeedback(button, 'Copied!');
    } catch (err) {
        console.error('Clipboard copy failed:', err);
        // alert('Clipboard copy failed: ' + err.message);
        // We don't show this message because it's not accurate. We could still write to the clipboard in this case.
    }
}

function showCopiedFeedback(button, message) {
    if (!button) return;
    const originalText = button.textContent;
    button.textContent = message;
    setTimeout(() => {
        button.textContent = originalText;
    }, 2000);
}

// Timer functions
function startTimer() {
    clearInterval(timerInterval);
    document.getElementById('timer').textContent = '00:00';
    startTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const minutes = Math.floor(elapsed / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        document.getElementById('timer').textContent = 
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }, 1000);
}

function stopTimer() {
    clearInterval(timerInterval);
}

// IndexedDB initialization
async function initIndexedDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('brainwave-replay', 1);
        
        request.onerror = () => {
            console.warn('IndexedDB not available, replay disabled');
            storageAvailable = false;
            if (replayButton) {
                replayButton.disabled = true;
                replayButton.title = 'Local storage not available';
            }
            resolve(false);
        };
        
        request.onsuccess = () => {
            db = request.result;
            
            // Check if required object stores exist
            if (!db.objectStoreNames.contains('sessions') || !db.objectStoreNames.contains('chunks')) {
                // Close and delete the database, then reopen to trigger onupgradeneeded
                db.close();
                const deleteRequest = indexedDB.deleteDatabase('brainwave-replay');
                deleteRequest.onsuccess = () => {
                    // Reopen database - this will trigger onupgradeneeded
                    const reopenRequest = indexedDB.open('brainwave-replay', 1);
                    reopenRequest.onsuccess = () => {
                        db = reopenRequest.result;
                        storageAvailable = true;
                        if (replayButton) {
                            replayButton.disabled = false;
                        }
                        updateReplayButtonState();
                        resolve(true);
                    };
                    reopenRequest.onerror = () => {
                        console.warn('Failed to reopen IndexedDB after cleanup');
                        storageAvailable = false;
                        if (replayButton) {
                            replayButton.disabled = true;
                            replayButton.title = 'Local storage not available';
                        }
                        resolve(false);
                    };
                    reopenRequest.onupgradeneeded = (event) => {
                        const db = event.target.result;
                        // Create sessions store
                        if (!db.objectStoreNames.contains('sessions')) {
                            const sessionsStore = db.createObjectStore('sessions', { keyPath: 'id', autoIncrement: true });
                            sessionsStore.createIndex('status', 'status', { unique: false });
                            sessionsStore.createIndex('createdAt', 'createdAt', { unique: false });
                        }
                        // Create chunks store
                        if (!db.objectStoreNames.contains('chunks')) {
                            const chunksStore = db.createObjectStore('chunks', { keyPath: 'id', autoIncrement: true });
                            chunksStore.createIndex('sessionId', 'sessionId', { unique: false });
                            chunksStore.createIndex('seq', 'seq', { unique: false });
                        }
                    };
                };
                deleteRequest.onerror = () => {
                    console.warn('Failed to delete corrupted IndexedDB');
                    storageAvailable = false;
                    if (replayButton) {
                        replayButton.disabled = true;
                        replayButton.title = 'Local storage not available';
                    }
                    resolve(false);
                };
                return;
            }
            
            storageAvailable = true;
            if (replayButton) {
                replayButton.disabled = false;
            }
            updateReplayButtonState();
            resolve(true);
        };
        
        request.onupgradeneeded = (event) => {
            db = event.target.result;
            
            // Create sessions store
            if (!db.objectStoreNames.contains('sessions')) {
                const sessionsStore = db.createObjectStore('sessions', { keyPath: 'id', autoIncrement: true });
                sessionsStore.createIndex('status', 'status', { unique: false });
                sessionsStore.createIndex('createdAt', 'createdAt', { unique: false });
            }
            
            // Create chunks store
            if (!db.objectStoreNames.contains('chunks')) {
                const chunksStore = db.createObjectStore('chunks', { keyPath: 'id', autoIncrement: true });
                chunksStore.createIndex('sessionId', 'sessionId', { unique: false });
                chunksStore.createIndex('seq', 'seq', { unique: false });
            }
        };
    });
}

// Audio processing
function createAudioProcessor() {
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = async (e) => {
        // Allow processing if recording OR if stopping (to capture final audio)
        if (!isRecording && !isStopping) return;
        
        const inputData = e.inputBuffer.getChannelData(0);
        const pcmData = new Int16Array(inputData.length);
        
        for (let i = 0; i < inputData.length; i++) {
            pcmData[i] = Math.max(-32768, Math.min(32767, Math.floor(inputData[i] * 32767)));
        }
        
        const combinedBuffer = new Int16Array(audioBuffer.length + pcmData.length);
        combinedBuffer.set(audioBuffer);
        combinedBuffer.set(pcmData, audioBuffer.length);
        audioBuffer = combinedBuffer;
        
        if (audioBuffer.length >= 24000) {
            const sendBuffer = audioBuffer.slice(0, 24000);
            audioBuffer = audioBuffer.slice(24000);
            
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(sendBuffer.buffer);
                
                // Store chunk in IndexedDB
                if (storageAvailable && currentSessionId && sessionStartTime) {
                    const deltaMs = performance.now() - sessionStartTime;
                    await appendChunk(currentSessionId, {
                        seq: chunkSeq++,
                        deltaMs: deltaMs,
                        kind: 'audio',
                        payload: sendBuffer.buffer,
                        byteLength: sendBuffer.byteLength
                    });
                }
            }
        }
    };
    return processor;
}

async function initAudio(stream) {
    console.log('Initializing AudioContext...');
    // Clean up existing audio context if any
    if (audioContext && audioContext.state !== 'closed') {
        try {
            await audioContext.close();
        } catch (e) {
            console.warn('Error closing existing audio context:', e);
        }
    }
    
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        source = audioContext.createMediaStreamSource(stream);
        processor = createAudioProcessor();
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
    } catch (e) {
        console.error('initAudio failed:', e);
        throw e;
    }
}

function cleanupAudioResources() {
    // Stop all tracks in the stream
    if (stream) {
        stream.getTracks().forEach(track => {
            track.stop();
        });
        stream = null;
    }
    
    // Close audio context
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close().catch(e => {
            console.warn('Error closing audio context:', e);
        });
        audioContext = null;
    }
    
    source = null;
    processor = null;
    streamInitialized = false;
}

// WebSocket handling
function updateConnectionStatus(status) {
    const statusDot = document.getElementById('connectionStatus');
    statusDot.classList.remove('connected', 'connecting', 'idle');
    
    switch (status) {
        case 'connected':  // OpenAI is connected and ready
            statusDot.classList.add('connected');
            statusDot.style.backgroundColor = '#34C759';  // Green
            break;
        case 'connecting':  // Establishing OpenAI connection
            statusDot.classList.add('connecting');
            statusDot.style.backgroundColor = '#FF9500';  // Orange
            break;
        case 'idle':  // Client connected, OpenAI not connected
            statusDot.classList.add('idle');
            statusDot.style.backgroundColor = '#007AFF';  // Blue
            break;
        default:  // Disconnected
            statusDot.style.backgroundColor = '#FF3B30';  // Red
    }
}

function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${protocol}://${window.location.host}/api/v1/ws`);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        wsConnected = true;
        // Set initial UI state to idle (blue) when socket opens
        updateConnectionStatus('idle');
        if (autoStart && !isRecording && !isAutoStarted) startRecording();
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch (data.type) {
            case 'status':
                updateConnectionStatus(data.status);
                // Stop timer when not actively recording/generating
                if (data.status === 'idle' || data.status === 'generating') {
                    stopTimer();
                }
                if (data.status === 'idle') {
                    copyToClipboard(transcript.value, copyButton);
                    // Update replay button state when status becomes idle
                    updateReplayButtonState();
                }
                break;
            case 'text':
                if (data.isNewResponse) {
                    transcript.value = data.content;
                    stopTimer();
                } else {
                    transcript.value += data.content;
                }
                transcript.scrollTop = transcript.scrollHeight;
                break;
            case 'error':
                alert(data.content);
                updateConnectionStatus('idle');
                updateReplayButtonState();
                break;
        }
    };
    
    ws.onclose = () => {
        wsConnected = false;
        updateConnectionStatus(false);
        // Do NOT clean up audio resources here to allow reuse of microphone
        // only reset recording state
        isRecording = false;
        isStopping = false;
        recordButton.textContent = 'Start';
        recordButton.classList.remove('recording');
        // Reconnect after a short delay
        setTimeout(initializeWebSocket, 1000);
    };
}

// IndexedDB helpers
async function createSession() {
    if (!db) return null;
    
    return new Promise((resolve, reject) => {
        const transaction = db.transaction(['sessions'], 'readwrite');
        const store = transaction.objectStore('sessions');
        const session = {
            createdAt: new Date(),
            status: 'recording',
            sampleRate: 24000,
            channelCount: 1,
            durationMs: 0
        };
        
        const request = store.add(session);
        request.onsuccess = () => {
            currentSessionId = request.result;
            sessionStartTime = performance.now();
            chunkSeq = 0;
            
            // Store start event
            appendChunk(currentSessionId, {
                seq: 0,
                deltaMs: 0,
                kind: 'start',
                payload: null,
                byteLength: 0
            });
            
            resolve(currentSessionId);
        };
        request.onerror = () => reject(request.error);
    });
}

async function appendChunk(sessionId, chunk) {
    if (!db || !sessionId) return;
    
    return new Promise((resolve, reject) => {
        const transaction = db.transaction(['chunks'], 'readwrite');
        const store = transaction.objectStore('chunks');
        const chunkData = {
            sessionId: sessionId,
            seq: chunk.seq,
            deltaMs: chunk.deltaMs,
            kind: chunk.kind,
            payload: chunk.payload,
            byteLength: chunk.byteLength
        };
        
        const request = store.add(chunkData);
        request.onsuccess = () => resolve();
        request.onerror = () => {
            console.warn('Failed to store chunk:', request.error);
            resolve(); // Don't fail recording if storage fails
        };
    });
}

async function completeSession(sessionId, durationMs) {
    if (!db || !sessionId) return;
    
    return new Promise((resolve) => {
        const transaction = db.transaction(['sessions', 'chunks'], 'readwrite');
        const sessionsStore = transaction.objectStore('sessions');
        const chunksStore = transaction.objectStore('chunks');
        
        // Update session status
        const getRequest = sessionsStore.get(sessionId);
        getRequest.onsuccess = () => {
            const session = getRequest.result;
            session.status = 'completed';
            session.durationMs = durationMs;
            sessionsStore.put(session);
            
            // Store stop event
            chunksStore.add({
                sessionId: sessionId,
                seq: chunkSeq++,
                deltaMs: durationMs,
                kind: 'stop',
                payload: null,
                byteLength: 0
            });
            
            // Enforce quota
            enforceQuota({ maxSessions: 5, maxBytes: 100 * 1024 * 1024 });
            
            resolve();
        };
        getRequest.onerror = () => resolve();
    });
}

async function enforceQuota({ maxSessions, maxBytes }) {
    if (!db) return;
    
    const transaction = db.transaction(['sessions', 'chunks'], 'readwrite');
    const sessionsStore = transaction.objectStore('sessions');
    const chunksStore = transaction.objectStore('chunks');
    const index = sessionsStore.index('createdAt');
    
    const sessions = [];
    index.openCursor().onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
            sessions.push(cursor.value);
            cursor.continue();
        } else {
            // Sort by creation time (oldest first)
            sessions.sort((a, b) => a.createdAt - b.createdAt);
            
            // Delete oldest sessions if over limit
            while (sessions.length > maxSessions) {
                const session = sessions.shift();
                deleteSession(session.id);
            }
        }
    };
}

async function deleteSession(sessionId) {
    if (!db) return;
    
    const transaction = db.transaction(['sessions', 'chunks'], 'readwrite');
    const sessionsStore = transaction.objectStore('sessions');
    const chunksStore = transaction.objectStore('chunks');
    const index = chunksStore.index('sessionId');
    
    // Delete all chunks for this session
    index.openKeyCursor(IDBKeyRange.only(sessionId)).onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
            chunksStore.delete(cursor.primaryKey);
            cursor.continue();
        }
    };
    
    // Delete session
    sessionsStore.delete(sessionId);
}

async function getLatestCompletedSession() {
    if (!db) return null;
    
    return new Promise((resolve) => {
        const transaction = db.transaction(['sessions'], 'readonly');
        const store = transaction.objectStore('sessions');
        const index = store.index('status');
        const request = index.getAll('completed');
        
        request.onsuccess = () => {
            const sessions = request.result;
            if (sessions.length === 0) {
                resolve(null);
                return;
            }
            
            // Sort by creation time (newest first)
            sessions.sort((a, b) => b.createdAt - a.createdAt);
            resolve(sessions[0]);
        };
        request.onerror = () => resolve(null);
    });
}

async function getSessionChunks(sessionId) {
    if (!db || !sessionId) return [];
    
    return new Promise((resolve) => {
        const transaction = db.transaction(['chunks'], 'readonly');
        const store = transaction.objectStore('chunks');
        const index = store.index('sessionId');
        const request = index.getAll(sessionId);
        
        request.onsuccess = () => {
            const chunks = request.result;
            chunks.sort((a, b) => a.seq - b.seq);
            resolve(chunks);
        };
        request.onerror = () => resolve([]);
    });
}

// Recording control
async function startRecording() {
    if (isRecording || isReplaying) return;
    
    // Check WebSocket connection
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        alert('WebSocket is not connected. Please wait a moment or refresh the page.');
        return;
    }
    
    try {
        transcript.value = '';
        enhancedTranscript.value = '';

        // Check if stream is still valid, reinitialize ONLY if needed
        let streamActive = false;
        try {
            streamActive = streamInitialized && stream && stream.active && 
                          stream.getTracks().length > 0 && 
                          stream.getTracks().every(track => track.readyState === 'live');
        } catch (e) {
            console.warn('Error checking stream status:', e);
            streamActive = false;
        }
        
        if (!streamActive) {
            console.log('Reinitializing microphone stream...');
            cleanupAudioResources();
            
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                streamInitialized = true;
            } catch (err) {
                console.error('getUserMedia error:', err);
                throw new Error('Could not access microphone. Please check permissions.');
            }
        }

        if (!stream) throw new Error('Failed to initialize audio stream');
        
        // Ensure AudioContext is active
        if (!audioContext || audioContext.state === 'closed') {
            await initAudio(stream);
        } else if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        isRecording = true;
        const modelSelect = document.getElementById('modelSelect');
        const selectedModel = modelSelect ? modelSelect.value : 'gpt-realtime-mini-2025-12-15';
        
        // Determine provider and model/voice based on selection
        let provider = 'openai';
        let model = selectedModel;
        let voice = null;
        
        if (selectedModel === 'xai-grok') {
            provider = 'xai';
            model = null;  // x.ai doesn't use model parameter
            voice = 'Ara';  // Default voice for x.ai
        }
        
        // Create session in IndexedDB
        if (storageAvailable) {
            await createSession();
        }
        
        await ws.send(JSON.stringify({ 
            type: 'start_recording', 
            provider: provider,
            model: model,
            voice: voice
        }));
        
        startTimer();
        recordButton.textContent = 'Stop';
        recordButton.classList.add('recording');
        if (replayButton) replayButton.disabled = true;
        
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Error accessing microphone: ' + error.message);
    }
}

async function stopRecording() {
    if (!isRecording) return;
    
    // Set stopping flag first to allow final audio processing
    isStopping = true;
    isRecording = false;
    const durationMs = performance.now() - sessionStartTime;
    
    // Stop local timer immediately on stop
    stopTimer();
    
    // Wait a bit to allow any in-flight onaudioprocess callbacks to complete
    // This ensures we capture the last bit of audio data
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Send any remaining audio buffer
    if (audioBuffer.length > 0 && ws.readyState === WebSocket.OPEN) {
        const sendBuffer = audioBuffer.slice();
        ws.send(sendBuffer.buffer);
        
        // Store final chunk
        if (storageAvailable && currentSessionId && sessionStartTime) {
            const deltaMs = performance.now() - sessionStartTime;
            await appendChunk(currentSessionId, {
                seq: chunkSeq++,
                deltaMs: deltaMs,
                kind: 'audio',
                payload: sendBuffer.buffer,
                byteLength: sendBuffer.byteLength
            });
        }
        
        audioBuffer = new Int16Array(0);
    }
    
    // Clear stopping flag
    isStopping = false;
    
    // Wait a bit more to ensure all audio is sent before stopping
    await new Promise(resolve => setTimeout(resolve, 500));
    await ws.send(JSON.stringify({ type: 'stop_recording' }));
    
    // Complete session in IndexedDB
    if (storageAvailable && currentSessionId) {
        await completeSession(currentSessionId, durationMs);
        currentSessionId = null;
        sessionStartTime = null;
        chunkSeq = 0;
    }
    
    recordButton.textContent = 'Start';
    recordButton.classList.remove('recording');
    updateReplayButtonState();
}

// Replay functionality
async function replayLastRecording() {
    if (isRecording || isReplaying || !storageAvailable) return;
    
    const session = await getLatestCompletedSession();
    if (!session) {
        alert('No completed recording found to replay.');
        return;
    }
    
    if (isRecording) {
        alert('Please stop recording before replaying.');
        return;
    }
    
    isReplaying = true;
    recordButton.disabled = true;
    if (replayButton) {
        replayButton.disabled = true;
        replayButton.title = 'Replaying...';
        replayButton.style.animation = 'spin 1s linear infinite';
    }
    
    try {
        // Get chunks for this session
        const chunks = await getSessionChunks(session.id);
        if (chunks.length === 0) {
            throw new Error('No chunks found for session');
        }
        
        // Clear transcript
        transcript.value = '';
        enhancedTranscript.value = '';
        
        // Ensure WebSocket is connected
        if (!wsConnected || ws.readyState !== WebSocket.OPEN) {
            await new Promise((resolve) => {
                const checkConnection = setInterval(() => {
                    if (wsConnected && ws.readyState === WebSocket.OPEN) {
                        clearInterval(checkConnection);
                        resolve();
                    }
                }, 100);
            });
        }
        
        // Send start_recording message
        const modelSelect = document.getElementById('modelSelect');
        const selectedModel = modelSelect ? modelSelect.value : 'gpt-realtime-mini-2025-12-15';
        
        // Determine provider and model/voice based on selection
        let provider = 'openai';
        let model = selectedModel;
        let voice = null;
        
        if (selectedModel === 'xai-grok') {
            provider = 'xai';
            model = null;
            voice = 'Ara';
        }
        
        await ws.send(JSON.stringify({ 
            type: 'start_recording', 
            provider: provider,
            model: model,
            voice: voice
        }));
        
        // Wait a bit for backend to initialize
        await new Promise(resolve => setTimeout(resolve, 200));
        
        // Replay audio chunks - send as fast as possible
        const audioChunks = chunks.filter(c => c.kind === 'audio' && c.payload);
        
        // Send all audio chunks immediately
        for (const chunk of audioChunks) {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(chunk.payload);
            } else {
                throw new Error('WebSocket closed during replay');
            }
        }
        
        // Send stop_recording after all audio chunks
        await ws.send(JSON.stringify({ type: 'stop_recording' }));
        
    } catch (error) {
        console.error('Error replaying recording:', error);
        alert('Error replaying recording: ' + error.message);
    } finally {
        isReplaying = false;
        recordButton.disabled = false;
        updateReplayButtonState();
    }
}

function updateReplayButtonState() {
    if (!replayButton) return;
    
    if (!storageAvailable) {
        replayButton.disabled = true;
        replayButton.title = 'Local storage not available';
        return;
    }
    
    if (isRecording || isReplaying) {
        replayButton.disabled = true;
        if (isReplaying) {
            replayButton.title = 'Replaying...';
            replayButton.style.animation = 'spin 1s linear infinite';
        } else {
            replayButton.title = 'Recording in progress';
            replayButton.style.animation = '';
        }
        return;
    }
    
    // Check if there's a completed session
    getLatestCompletedSession().then(session => {
        if (session) {
            replayButton.disabled = false;
            replayButton.title = 'Replay last recording';
            replayButton.style.animation = '';
        } else {
            replayButton.disabled = true;
            replayButton.title = 'No recording to replay';
            replayButton.style.animation = '';
        }
    });
}

// Event listeners
recordButton.onclick = () => isRecording ? stopRecording() : startRecording();
if (replayButton) replayButton.onclick = replayLastRecording;
copyButton.onclick = () => copyToClipboard(transcript.value, copyButton);
copyEnhancedButton.onclick = () => copyToClipboard(enhancedTranscript.value, copyEnhancedButton);

// Handle spacebar toggle
document.addEventListener('keydown', (event) => {
    if (event.code === 'Space') {
        const activeElement = document.activeElement;
        if (!activeElement.tagName.match(/INPUT|TEXTAREA/) && !activeElement.isContentEditable) {
            event.preventDefault();
            recordButton.click();
        }
    }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await initIndexedDB();
    initializeWebSocket();
    initializeTheme();
});
// Readability and AI handlers
if (readabilityButton) readabilityButton.onclick = async () => {
    startTimer();
    const inputText = transcript.value.trim();
    if (!inputText) {
        alert('Please enter text to enhance readability.');
        stopTimer();
        return;
    }

    try {
        const response = await fetch('/api/v1/readability', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: inputText })
        });

        if (!response.ok) throw new Error('Readability enhancement failed');

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

        if (!isMobileDevice()) copyToClipboard(fullText, copyEnhancedButton);
        stopTimer();

    } catch (error) {
        console.error('Error:', error);
        alert('Error enhancing readability');
        stopTimer();
    }
};

if (askAIButton) askAIButton.onclick = async () => {
    startTimer();
    const inputText = transcript.value.trim();
    if (!inputText) {
        alert('Please enter text to ask AI about.');
        stopTimer();
        return;
    }

    try {
        const response = await fetch('/api/v1/ask_ai', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: inputText })
        });

        if (!response.ok) throw new Error('AI request failed');

        const result = await response.json();
        enhancedTranscript.value = result.answer;
        if (!isMobileDevice()) copyToClipboard(result.answer, copyEnhancedButton);
        stopTimer();

    } catch (error) {
        console.error('Error:', error);
        alert('Error asking AI');
        stopTimer();
    }
};

if (correctnessButton) correctnessButton.onclick = async () => {
    startTimer();
    const inputText = transcript.value.trim();
    if (!inputText) {
        alert('Please enter text to check for correctness.');
        stopTimer();
        return;
    }

    try {
        const response = await fetch('/api/v1/correctness', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: inputText })
        });

        if (!response.ok) throw new Error('Correctness check failed');

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

        if (!isMobileDevice()) copyToClipboard(fullText, copyEnhancedButton);
        stopTimer();

    } catch (error) {
        console.error('Error:', error);
        alert('Error checking correctness');
        stopTimer();
    }
};

// Theme handling
function toggleTheme() {
    const body = document.body;
    const themeToggle = document.getElementById('themeToggle');
    const isDarkTheme = body.classList.toggle('dark-theme');
    
    // Update button text
    themeToggle.textContent = isDarkTheme ? '☀️' : '🌙';
    
    // Save preference to localStorage
    localStorage.setItem('darkTheme', isDarkTheme);
}

// Initialize theme from saved preference
function initializeTheme() {
    const darkTheme = localStorage.getItem('darkTheme') === 'true';
    const themeToggle = document.getElementById('themeToggle');
    
    if (darkTheme) {
        document.body.classList.add('dark-theme');
        themeToggle.textContent = '☀️';
    }
}

// Add to your existing event listeners
document.getElementById('themeToggle').onclick = toggleTheme;
