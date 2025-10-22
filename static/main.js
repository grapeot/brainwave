// Global state
let ws, audioContext, processor, source, stream;
let isRecording = false;
let timerInterval;
let startTime;
let audioBuffer = new Int16Array(0);
let wsConnected = false;
let streamInitialized = false;
let isAutoStarted = false;
let spaceKeyHeld = false;
let spaceRecordingActive = false;
let spaceStartInProgress = false;
let currentStatus = 'disconnected';
let currentLanguage = 'zh';

// DOM elements
const recordButton = document.getElementById('recordButton');
const transcript = document.getElementById('transcript');
const copyButton = document.getElementById('copyButton');
const themeToggleButton = document.getElementById('themeToggle');
const languageToggle = document.getElementById('languageToggle');
const hotkeyHint = document.getElementById('hotkeyHelp');
const recordLabel = document.querySelector('.record-label');
const transcriptTitle = document.querySelector('.transcript-title');
const transcriptArea = document.getElementById('transcript');
const brandSubtitle = document.querySelector('.brand-subtitle');

// Configuration
const targetSeconds = 5;
const urlParams = new URLSearchParams(window.location.search);
const autoStart = urlParams.get('start') === '1';
const prefersDarkScheme = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

const LANGUAGE_MAP = {
    zh: {
        copyButton: '复制',
        copySuccess: '已复制',
        hotkeyHint: '按住 Space 开始录音，松开停止；按 Shift 在开始和结束之间切换',
        recordStart: '开始',
        recordStop: '停止',
        brandSubtitle: '实时语音笔记助手',
        transcriptTitle: '转录结果',
        transcriptPlaceholder: '转录内容将显示在此处……',
        statuses: {
            disconnected: '未连接',
            connecting: '连接中…',
            connected: '已连接',
            idle: '空闲',
            error: '连接错误'
        }
    },
    en: {
        copyButton: 'Copy',
        copySuccess: 'Copied!',
        hotkeyHint: 'Hold Space to start recording, release to stop; press Shift to toggle.',
        recordStart: 'Start',
        recordStop: 'Stop',
        brandSubtitle: 'Real-time notes companion',
        transcriptTitle: 'Transcript',
        transcriptPlaceholder: 'Transcription will appear here…',
        statuses: {
            disconnected: 'Disconnected',
            connecting: 'Connecting…',
            connected: 'Connected',
            idle: 'Idle',
            error: 'Connection error'
        }
    }
};

// Utility functions
const isMobileDevice = () => /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

const isTypingContext = () => {
    const activeElement = document.activeElement;
    if (!activeElement) return false;
    const tagName = activeElement.tagName;
    if (activeElement.isContentEditable) return true;
    return tagName === 'INPUT' || tagName === 'TEXTAREA' || activeElement.getAttribute('role') === 'textbox';
};

const applyThemePreference = (isDark) => {
    document.body.classList.toggle('dark-theme', Boolean(isDark));
    if (themeToggleButton) {
        themeToggleButton.textContent = isDark ? '☀️' : '🌙';
    }
};

const handleSystemThemeChange = (event) => {
    if (localStorage.getItem('darkTheme') === null) {
        applyThemePreference(event.matches);
    }
};

const getTranslations = () => {
    return LANGUAGE_MAP[currentLanguage] || LANGUAGE_MAP.en;
};

const updateRecordButtonLabel = () => {
    if (!recordLabel) return;
    const translations = getTranslations();
    recordLabel.textContent = isRecording ? translations.recordStop : translations.recordStart;
};

const applyLanguage = () => {
    const translations = getTranslations();
    if (languageToggle) {
        languageToggle.value = currentLanguage;
        languageToggle.setAttribute('aria-label', currentLanguage === 'zh' ? '选择语言' : 'Select language');
        languageToggle.setAttribute('title', currentLanguage === 'zh' ? '选择语言' : 'Select language');
    }
    updateRecordButtonLabel();
    if (copyButton) {
        copyButton.textContent = translations.copyButton;
    }
    if (hotkeyHint) {
        hotkeyHint.textContent = translations.hotkeyHint;
    }
    if (transcriptTitle) {
        transcriptTitle.textContent = translations.transcriptTitle;
    }
    if (transcriptArea) {
        transcriptArea.setAttribute('placeholder', translations.transcriptPlaceholder);
    }
    if (brandSubtitle) {
        brandSubtitle.textContent = translations.brandSubtitle;
    }
    updateConnectionStatus(currentStatus);
};

const setLanguage = (lang) => {
    if (!LANGUAGE_MAP[lang]) return;
    currentLanguage = lang;
    localStorage.setItem('uiLanguage', lang);
    applyLanguage();
};

const initializeLanguage = () => {
    let lang = localStorage.getItem('uiLanguage');

    if (!lang || !LANGUAGE_MAP[lang]) {
        const navigatorLang = (navigator.language || navigator.userLanguage || '').toLowerCase();
        lang = navigatorLang.startsWith('zh') ? 'zh' : 'en';
    }

    if (!LANGUAGE_MAP[lang]) {
        lang = 'en';
    }

    currentLanguage = lang;

    if (languageToggle) {
        languageToggle.value = currentLanguage;
        languageToggle.addEventListener('change', (event) => {
            setLanguage(event.target.value);
        });
    }

    applyLanguage();
};

async function copyToClipboard(text, button) {
    if (!text) return;
    try {
        await navigator.clipboard.writeText(text);
        const translations = getTranslations();
        showCopiedFeedback(button, translations.copySuccess);
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

// Audio processing
function createAudioProcessor() {
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
        if (!isRecording) return;
        
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
            }
        }
    };
    return processor;
}

async function initAudio(stream) {
    audioContext = new AudioContext();
    source = audioContext.createMediaStreamSource(stream);
    processor = createAudioProcessor();
    source.connect(processor);
    processor.connect(audioContext.destination);
}

// WebSocket handling
function updateConnectionStatus(status) {
    const statusElement = document.getElementById('connectionStatus');
    if (!statusElement) return;

    let normalizedStatus = status;
    if (normalizedStatus === true) normalizedStatus = 'connected';
    if (normalizedStatus === false) normalizedStatus = 'disconnected';
    if (typeof normalizedStatus !== 'string') {
        normalizedStatus = 'disconnected';
    }

    const statusClasses = ['status-connected', 'status-connecting', 'status-idle', 'status-disconnected', 'status-error'];
    statusElement.classList.remove(...statusClasses);

    let className = 'status-disconnected';
    switch (normalizedStatus) {
        case 'connected':
            className = 'status-connected';
            break;
        case 'connecting':
            className = 'status-connecting';
            break;
        case 'idle':
            className = 'status-idle';
            break;
        case 'error':
            className = 'status-error';
            break;
        default:
            normalizedStatus = 'disconnected';
            className = 'status-disconnected';
    }

    const translations = getTranslations();
    const label = translations.statuses[normalizedStatus] || translations.statuses.disconnected;

    statusElement.classList.add(className);
    statusElement.textContent = label;
    currentStatus = normalizedStatus;
}

function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    updateConnectionStatus('connecting');
    ws = new WebSocket(`${protocol}://${window.location.host}/api/v1/ws`);
    
    ws.onopen = () => {
        wsConnected = true;
        updateConnectionStatus('connected');
        if (autoStart && !isRecording && !isAutoStarted) startRecording();
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch (data.type) {
            case 'status':
                updateConnectionStatus(data.status);
                if (data.status === 'idle') {
                    copyToClipboard(transcript.value, copyButton);
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
                updateConnectionStatus('error');
                break;
        }
    };
    
    ws.onclose = () => {
        wsConnected = false;
        updateConnectionStatus('disconnected');
        setTimeout(initializeWebSocket, 1000);
    };
}

// Recording control
async function startRecording() {
    if (isRecording) return;
    
    try {
        transcript.value = '';

        if (!streamInitialized) {
            stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            streamInitialized = true;
        }

        if (!stream) throw new Error('Failed to initialize audio stream');
        if (!audioContext) await initAudio(stream);

        isRecording = true;
        await ws.send(JSON.stringify({ type: 'start_recording' }));
        
        startTimer();
        updateRecordButtonLabel();
        recordButton.classList.add('recording');
        
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Error accessing microphone: ' + error.message);
    }
}

async function stopRecording() {
    if (!isRecording) return;
    
    isRecording = false;
    startTimer();
    
    if (audioBuffer.length > 0 && ws.readyState === WebSocket.OPEN) {
        ws.send(audioBuffer.buffer);
        audioBuffer = new Int16Array(0);
    }
    
    await new Promise(resolve => setTimeout(resolve, 500));
    await ws.send(JSON.stringify({ type: 'stop_recording' }));
    
    updateRecordButtonLabel();
    recordButton.classList.remove('recording');
}

// Event listeners
recordButton.onclick = () => isRecording ? stopRecording() : startRecording();
copyButton.onclick = () => copyToClipboard(transcript.value, copyButton);

const finalizeSpaceRelease = () => {
    if (!spaceKeyHeld && spaceRecordingActive && !spaceStartInProgress) {
        spaceRecordingActive = false;
        stopRecording().catch((error) => {
            console.error('Error stopping recording via spacebar:', error);
        });
    }
};

const handleSpaceKeyDown = (event) => {
    if (event.code !== 'Space') return;
    if (isTypingContext()) return;
    if (event.repeat) {
        event.preventDefault();
        return;
    }

    event.preventDefault();
    spaceKeyHeld = true;

    if (spaceRecordingActive || spaceStartInProgress || isRecording) {
        return;
    }

    spaceStartInProgress = true;

    (async () => {
        try {
            await startRecording();
            if (spaceKeyHeld) {
                spaceRecordingActive = true;
                return;
            }

            spaceRecordingActive = true;
            await stopRecording();
            spaceRecordingActive = false;
        } catch (error) {
            console.error('Error starting recording via spacebar:', error);
            spaceRecordingActive = false;
            spaceKeyHeld = false;
        } finally {
            spaceStartInProgress = false;
            finalizeSpaceRelease();
        }
    })();
};

const handleSpaceKeyUp = (event) => {
    if (event.code !== 'Space') return;
    if (isTypingContext()) return;

    event.preventDefault();
    spaceKeyHeld = false;
    finalizeSpaceRelease();
};

const handleShiftKeyDown = (event) => {
    if (event.code !== 'ShiftLeft' && event.code !== 'ShiftRight') return;
    if (isTypingContext()) return;
    if (event.repeat) {
        event.preventDefault();
        return;
    }

    event.preventDefault();

    if (spaceStartInProgress) {
        return;
    }

    spaceKeyHeld = false;
    spaceRecordingActive = false;

    (async () => {
        try {
            if (isRecording) {
                await stopRecording();
            } else {
                await startRecording();
            }
        } catch (error) {
            console.error('Error toggling recording via Shift:', error);
        }
    })();
};

window.addEventListener('keydown', (event) => {
    handleSpaceKeyDown(event);
    handleShiftKeyDown(event);
}, true);

window.addEventListener('keyup', (event) => {
    handleSpaceKeyUp(event);
}, true);

window.addEventListener('blur', () => {
    if (!spaceKeyHeld && !spaceRecordingActive) return;
    spaceKeyHeld = false;
    finalizeSpaceRelease();
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeLanguage();
    initializeTheme();
    initializeWebSocket();
    if (autoStart) initializeAudioStream();
});

// Theme handling
function toggleTheme() {
    const nextIsDark = !document.body.classList.contains('dark-theme');
    applyThemePreference(nextIsDark);
    localStorage.setItem('darkTheme', nextIsDark ? 'true' : 'false');
}

// Initialize theme from saved preference / system preference
function initializeTheme() {
    const storedPreference = localStorage.getItem('darkTheme');

    if (storedPreference === null) {
        if (prefersDarkScheme) {
            applyThemePreference(prefersDarkScheme.matches);
            if (typeof prefersDarkScheme.addEventListener === 'function') {
                prefersDarkScheme.addEventListener('change', handleSystemThemeChange);
            } else if (typeof prefersDarkScheme.addListener === 'function') {
                prefersDarkScheme.addListener(handleSystemThemeChange);
            }
        } else {
            applyThemePreference(false);
        }
    } else {
        applyThemePreference(storedPreference === 'true');
    }
}

if (themeToggleButton) {
    themeToggleButton.onclick = toggleTheme;
}
