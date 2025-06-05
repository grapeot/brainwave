// Global state for transcription
// let transcribeWs = null; // REMOVED WebSocket
let mediaRecorder = null; // ADDED for MediaRecorder
let recordedChunks = []; // ADDED for audio data
let isClientRecording = false; // ADDED to track recording state

let transcribeAudioContext = null; // May not be needed if MediaRecorder is self-sufficient
// let transcribeProcessor = null; // REMOVED, ScriptProcessor specific
// let transcribeSource = null;    // REMOVED, ScriptProcessor specific
let transcribeStream = null; // Will be used for MediaRecorder's stream
let transcribeTimerInterval = null;
let transcribeStartTime = null;

// DOM elements for transcription page
const transcribeRecordButton = document.getElementById('recordButton');
const transcribeTranscriptTextarea = document.getElementById('transcript');
const transcribeCopyButton = document.getElementById('copyButton');
const transcribeConnectionStatus = document.getElementById('connectionStatus');
const transcribeTimerDisplay = document.getElementById('timer');
const transcribeThemeToggle = document.getElementById('themeToggle');

// Audio format detection for cross-platform compatibility
function detectBestAudioFormat() {
    const testRecorder = document.createElement('canvas');
    const formats = [
        // Priority 1: MP4/AAC - Best iOS/Safari compatibility
        { mimeType: 'audio/mp4; codecs=mp4a.40.2', name: 'MP4/AAC', efficient: true, lossy: true },
        { mimeType: 'audio/mp4', name: 'MP4 (default codec)', efficient: true, lossy: true },
        // Priority 2: WebM/Opus - Great for modern browsers, excellent efficiency
        { mimeType: 'audio/webm; codecs=opus', name: 'WebM/Opus', efficient: true, lossy: true },
        { mimeType: 'audio/webm', name: 'WebM (default codec)', efficient: true, lossy: true },
        // Priority 3: Legacy fallbacks
        { mimeType: 'audio/ogg; codecs=opus', name: 'OGG/Opus', efficient: true, lossy: true },
        { mimeType: 'audio/wav', name: 'WAV', efficient: false, lossy: false }
    ];

    let selectedFormat = null;
    
    // Test each format for MediaRecorder support
    for (const format of formats) {
        if (MediaRecorder.isTypeSupported(format.mimeType)) {
            selectedFormat = format;
            console.log(`✅ Selected audio format: ${format.name} (${format.mimeType})`);
            console.log(`   Compression: ${format.lossy ? 'Lossy' : 'Lossless'}, Efficiency: ${format.efficient ? 'High' : 'Low'}`);
            break;
        } else {
            console.log(`❌ Format not supported: ${format.name} (${format.mimeType})`);
        }
    }
    
    if (!selectedFormat) {
        console.warn('⚠️  No supported audio formats found, using browser default');
        selectedFormat = { mimeType: '', name: 'Browser Default', efficient: true, lossy: true };
    }
    
    return selectedFormat;
}

// Utility functions
function showCopiedFeedback(button, message) {
    if (!button) return;
    const originalText = button.textContent;
    button.textContent = message;
    setTimeout(() => {
        button.textContent = originalText;
    }, 2000);
}

async function copyMainTranscriptToClipboard() {
    if (!transcribeTranscriptTextarea || !transcribeCopyButton) return;
    const textToCopy = transcribeTranscriptTextarea.value;
    if (!textToCopy) return;
    try {
        await navigator.clipboard.writeText(textToCopy);
        showCopiedFeedback(transcribeCopyButton, 'Copied!');
    } catch (err) {
        console.error('Clipboard copy failed:', err);
    }
}

// Timer functions
function startTranscribeTimer() {
    clearInterval(transcribeTimerInterval);
    if (transcribeTimerDisplay) transcribeTimerDisplay.textContent = '00:00';
    transcribeStartTime = Date.now();
    transcribeTimerInterval = setInterval(() => {
        if (!transcribeTimerDisplay) return;
        const elapsed = Date.now() - transcribeStartTime;
        const minutes = Math.floor(elapsed / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        transcribeTimerDisplay.textContent =
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }, 1000);
}

function stopTranscribeTimer() {
    clearInterval(transcribeTimerInterval);
}

// Recording control for transcription (MediaRecorder based)
async function startClientRecording() {
    if (isClientRecording) return;

    transcribeTranscriptTextarea.value = ''; // Clear previous transcript
    recordedChunks = [];

    try {
        if (!transcribeStream || !transcribeStream.active || transcribeStream.getAudioTracks().length === 0 || !transcribeStream.getAudioTracks()[0].enabled) {
            transcribeStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
        }
    } catch (error) {
        console.error("Error getting user media:", error);
        alert("Error accessing microphone: " + error.message + ". Please ensure permission is granted.");
        updateTranscribeConnectionStatus('mic_error');
        transcribeRecordButton.textContent = 'Start Recording';
        transcribeRecordButton.classList.remove('recording');
        isClientRecording = false;
        return;
    }

    // Use intelligent format detection instead of hardcoded OGG
    const selectedFormat = detectBestAudioFormat();
    console.log(`🎵 Using audio format: ${selectedFormat.name}`);

    try {
        if (selectedFormat.mimeType) {
            mediaRecorder = new MediaRecorder(transcribeStream, { mimeType: selectedFormat.mimeType });
        } else {
            mediaRecorder = new MediaRecorder(transcribeStream); // Use browser default
        }
    } catch (error) {
        console.error("Failed to create MediaRecorder with selected format:", error);
        // Ultimate fallback to browser default
        try {
            mediaRecorder = new MediaRecorder(transcribeStream);
            console.log("✅ Using browser default MediaRecorder format");
        } catch (fallbackError) {
            console.error("Failed to create MediaRecorder with default options:", fallbackError);
            alert("Failed to initialize audio recorder. Your browser might not support MediaRecorder.");
            updateTranscribeConnectionStatus('error');
            return;
        }
    }
    
    console.log("MediaRecorder created with mimeType:", mediaRecorder.mimeType);

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = async () => {
        console.log("MediaRecorder stopped. Processing recorded chunks.");
        const audioBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
        recordedChunks = []; 
        await sendAudioAndProcessSSE(audioBlob);
    };
    
    mediaRecorder.onerror = (event) => {
        console.error("MediaRecorder error:", event.error);
        alert("An error occurred with the audio recorder: " + event.error.name);
        stopClientRecordingAndProcess(true); 
        updateTranscribeConnectionStatus('error');
    };

    mediaRecorder.start();
    isClientRecording = true;
    startTranscribeTimer();
    transcribeRecordButton.textContent = 'Stop Recording';
    transcribeRecordButton.classList.add('recording');
    updateTranscribeConnectionStatus('recording');
}

async function stopClientRecordingAndProcess(errorOccurred = false) {
    if (!isClientRecording && !errorOccurred) {
        return;
    }

    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop(); 
    }
    
    isClientRecording = false;
    stopTranscribeTimer();
    transcribeRecordButton.textContent = 'Start Recording'; 
    transcribeRecordButton.classList.remove('recording');
    
    if (errorOccurred) {
        updateTranscribeConnectionStatus('error');
        if (transcribeStream) {
            transcribeStream.getTracks().forEach(track => track.stop());
            transcribeStream = null;
        }
    } else {
        updateTranscribeConnectionStatus('processing'); 
    }
}

async function sendAudioAndProcessSSE(audioBlob) {
    if (!audioBlob || audioBlob.size === 0) {
        console.warn("Audio blob is empty, not sending.");
        updateTranscribeConnectionStatus('idle');
        transcribeRecordButton.textContent = 'Start Recording';
        return;
    }

    updateTranscribeConnectionStatus('processing');
    transcribeRecordButton.textContent = 'Processing...';
    transcribeRecordButton.disabled = true;
    transcribeTranscriptTextarea.value = ""; // Clear for new transcription (no extra whitespace)

    const formData = new FormData();
    const fileName = `recording.${audioBlob.type.split('/')[1] || 'ogg'}`.replace(';codecs=opus','');
    formData.append('file', audioBlob, fileName);

    try {
        const response = await fetch('/api/v1/transcribe_gemini', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error("Server error:", response.status, errorText);
            alert(`Error from server: ${response.status} - ${errorText || 'Failed to start transcription.'}`);
            throw new Error(`Server responded with ${response.status}`);
        }

        if (!response.body) {
            throw new Error("Response body is null");
        }

        updateTranscribeConnectionStatus('transcribing');
        transcribeRecordButton.textContent = 'Transcribing...';

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                console.log("SSE Stream finished.");
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            console.log("Current buffer content:", JSON.stringify(buffer));
            console.log("Looking for \\n\\n separator in buffer...");
            
            // Process all complete events in the current buffer
            let eventSeparatorIndex;
            while ((eventSeparatorIndex = buffer.indexOf('\n\n')) !== -1) {
                const rawEvent = buffer.substring(0, eventSeparatorIndex);
                buffer = buffer.substring(eventSeparatorIndex + 2);

                console.log("Processing SSE event:", JSON.stringify(rawEvent));
                console.log("Remaining buffer after extraction:", JSON.stringify(buffer));

                if (rawEvent.trim() === '') {
                    console.log("Empty event, skipping...");
                    continue;
                }

                let eventType = 'message'; 
                let eventData = '';

                const lines = rawEvent.split('\n');
                console.log("Event lines:", lines);
                
                for (const line of lines) {
                    if (line.startsWith('event:')) {
                        eventType = line.substring('event:'.length).trim();
                        console.log("Found event type:", eventType);
                    } else if (line.startsWith('data:')) {
                        eventData = line.substring('data:'.length).trim();
                        console.log("Found event data:", JSON.stringify(eventData));
                    }
                }

                if (eventData) {
                    try {
                        console.log("Attempting to parse JSON:", eventData);
                        const jsonData = JSON.parse(eventData);
                        console.log("Successfully parsed JSON:", jsonData);
                        
                        if (eventType === 'error') {
                            console.error("SSE Error Event:", jsonData.error);
                            transcribeTranscriptTextarea.value += `[ERROR: ${jsonData.error}]`;
                            updateTranscribeConnectionStatus('error');
                        } else if (jsonData.text_chunk) {
                            console.log("Adding text chunk to textarea:", jsonData.text_chunk);
                            console.log("Text chunk analysis:", {
                                length: jsonData.text_chunk.length,
                                startsWithSpace: jsonData.text_chunk.startsWith(' '),
                                endsWithSpace: jsonData.text_chunk.endsWith(' '),
                                trimmed: jsonData.text_chunk.trim(),
                                repr: JSON.stringify(jsonData.text_chunk)
                            });
                            
                            // Smart text concatenation - avoid duplicate spaces
                            let textToAdd = jsonData.text_chunk;
                            const currentText = transcribeTranscriptTextarea.value;
                            
                            console.log("Current text ending:", {
                                lastChar: currentText.length > 0 ? currentText.charAt(currentText.length - 1) : 'EMPTY',
                                endsWithSpace: currentText.endsWith(' ')
                            });
                            
                            // If current text ends with space and new text starts with space, remove one
                            if (currentText.length > 0 && 
                                currentText.endsWith(' ') && 
                                textToAdd.startsWith(' ')) {
                                textToAdd = textToAdd.substring(1);
                                console.log("Removed duplicate space. New text:", JSON.stringify(textToAdd));
                            }
                            
                            transcribeTranscriptTextarea.value += textToAdd;
                            transcribeTranscriptTextarea.scrollTop = transcribeTranscriptTextarea.scrollHeight;
                        }
                    } catch (e) {
                        console.error("Error parsing SSE JSON data:", e, "Raw data:", eventData);
                        // Only show RAW DATA if the data seems meaningful (not just whitespace or artifacts)
                        if (eventData.trim() && eventData.length > 2) {
                            console.log("Adding RAW DATA to textarea:", eventData);
                            transcribeTranscriptTextarea.value += `[RAW DATA: ${eventData}]`;
                        }
                    }
                } else {
                    console.log("No event data found in event:", rawEvent);
                }
            }
            
            console.log("Finished processing events in this chunk. Remaining buffer:", JSON.stringify(buffer));
        }
        
        // Handle any remaining data in the buffer if the stream ended without \n\n
        if (buffer.trim()) {
            console.warn("SSE stream ended with unprocessed data in buffer:", JSON.stringify(buffer));
            console.warn("This should normally be empty or incomplete data. If you see complete events here, there's still a parsing issue.");
            
            // Try to process as a single event if it looks like one
            if (buffer.includes('data:') && !buffer.includes('\n\n')) {
                console.log("Found incomplete event in final buffer, attempting to process...");
                
                let eventType = 'message';
                let eventData = '';

                const lines = buffer.split('\n');
                console.log("Final buffer lines:", lines);
                
                for (const line of lines) {
                    if (line.startsWith('event:')) {
                        eventType = line.substring('event:'.length).trim();
                        console.log("Found final event type:", eventType);
                    } else if (line.startsWith('data:')) {
                        eventData = line.substring('data:'.length).trim();
                        console.log("Found final event data:", JSON.stringify(eventData));
                    }
                }

                if (eventData) {
                    try {
                        console.log("Attempting to parse final JSON:", eventData);
                        const jsonData = JSON.parse(eventData);
                        console.log("Successfully parsed final JSON:", jsonData);
                        
                        if (eventType === 'error') {
                            console.error("SSE Error Event (final):", jsonData.error);
                            transcribeTranscriptTextarea.value += `[ERROR: ${jsonData.error}]`;
                            updateTranscribeConnectionStatus('error');
                        } else if (jsonData.text_chunk) {
                            console.log("Adding final text chunk to textarea:", jsonData.text_chunk);
                            // Smart text concatenation - avoid duplicate spaces
                            let textToAdd = jsonData.text_chunk;
                            const currentText = transcribeTranscriptTextarea.value;
                            
                            // If current text ends with space and new text starts with space, remove one
                            if (currentText.length > 0 && 
                                currentText.endsWith(' ') && 
                                textToAdd.startsWith(' ')) {
                                textToAdd = textToAdd.substring(1);
                            }
                            
                            transcribeTranscriptTextarea.value += textToAdd;
                            transcribeTranscriptTextarea.scrollTop = transcribeTranscriptTextarea.scrollHeight;
                        }
                    } catch (e) {
                        console.error("Error parsing final SSE JSON data:", e, "Raw data:", eventData);
                        if (eventData.trim() && eventData.length > 2) {
                            console.log("Adding final RAW DATA to textarea:", eventData);
                            transcribeTranscriptTextarea.value += `[RAW DATA: ${eventData}]`;
                        }
                    }
                }
            } else {
                console.log("Final buffer contains complex data or multiple events - this suggests main loop parsing failed:");
                console.log("Buffer content:", JSON.stringify(buffer));
                transcribeTranscriptTextarea.value += `[UNPARSED BUFFER CONTENT]`;
            }
        }

        updateTranscribeConnectionStatus('finished');
    } catch (error) {
        console.error("Error during transcription fetch/SSE processing:", error);
        alert("An error occurred during transcription: " + error.message);
        updateTranscribeConnectionStatus('error');
    } finally {
        transcribeRecordButton.textContent = 'Start Recording';
        transcribeRecordButton.disabled = false;
        if (transcribeStream) {
            transcribeStream.getTracks().forEach(track => track.stop());
            transcribeStream = null;
        }
    }
}

// Update status function
function updateTranscribeConnectionStatus(status) {
    if (!transcribeConnectionStatus) return;
    transcribeConnectionStatus.className = 'connection-status'; 

    switch (status) {
        case 'idle':
            transcribeConnectionStatus.classList.add('idle');
            transcribeConnectionStatus.title = 'Ready to Record';
            break;
        case 'recording':
            transcribeConnectionStatus.classList.add('connected'); 
            transcribeConnectionStatus.title = 'Recording...';
            break;
        case 'processing':
            transcribeConnectionStatus.classList.add('connecting'); 
            transcribeConnectionStatus.title = 'Processing audio...';
            break;
        case 'transcribing':
            transcribeConnectionStatus.classList.add('connected'); 
            transcribeConnectionStatus.title = 'Transcribing...';
            break;
        case 'finished':
            transcribeConnectionStatus.classList.add('idle');
            transcribeConnectionStatus.title = 'Transcription finished. Ready to record again.';
            break;
        case 'mic_error':
            transcribeConnectionStatus.classList.add('error');
            transcribeConnectionStatus.title = 'Microphone Access Error.';
            break;
        case 'error':
        default:
            transcribeConnectionStatus.classList.add('error');
            transcribeConnectionStatus.title = 'Error Occurred';
            break;
    }
}

// Theme handling
function toggleTranscribeTheme() {
    const body = document.body;
    const isDarkTheme = body.classList.toggle('dark-theme');
    if (transcribeThemeToggle) transcribeThemeToggle.textContent = isDarkTheme ? '☀️' : '🌙';
    localStorage.setItem('darkTheme', isDarkTheme ? 'true' : 'false');
}

function initializeTranscribeTheme() {
    if (!transcribeThemeToggle) return;
    const darkTheme = localStorage.getItem('darkTheme') === 'true';
    if (darkTheme) {
        document.body.classList.add('dark-theme');
        transcribeThemeToggle.textContent = '☀️';
    } else {
        document.body.classList.remove('dark-theme');
        transcribeThemeToggle.textContent = '🌙';
    }
}

// Event listeners for transcription page
document.addEventListener('DOMContentLoaded', () => {
    if (transcribeRecordButton) {
        transcribeRecordButton.onclick = () => {
            if (isClientRecording) {
                stopClientRecordingAndProcess();
            } else {
                startClientRecording();
            }
        };
    }

    if (transcribeCopyButton) {
        transcribeCopyButton.onclick = copyMainTranscriptToClipboard;
    }

    if (transcribeThemeToggle) {
        transcribeThemeToggle.onclick = toggleTranscribeTheme;
    }

    initializeTranscribeTheme();
    updateTranscribeConnectionStatus('idle'); 
});

// Optional: Handle spacebar toggle for transcription
document.addEventListener('keydown', (event) => {
    if (event.code === 'Space' && transcribeRecordButton) {
        const activeElement = document.activeElement;
        if ((!activeElement || !activeElement.tagName.match(/INPUT|TEXTAREA/)) &&
            (!activeElement || !activeElement.isContentEditable)) {
            event.preventDefault();
            // Ensure the button click respects disabled state if we add one during processing
            if (!transcribeRecordButton.disabled) {
                transcribeRecordButton.click();
            }
        }
    }
});

// REMOVE THE FOLLOWING OLD FUNCTIONS:
// function createTranscribeAudioProcessor() { ... }
// async function initTranscribeAudio() { ... }
// function initializeTranscribeWebSocket() { ... }
// async function startTranscription() { ... } 
// async function stopTranscription(errorStop = false) { ... }
// The actual removal will be done by not including them in the code_edit above.
// This comment is a note for the LLM. The above code_edit represents the full desired file content. 