# Audio Capture and Volume Control

## Overview

This document explains how Brainwave handles microphone input, audio processing, and volume control for real-time transcription with OpenAI's Realtime API.

## Audio Flow Architecture

```
Browser Microphone
    ↓ (48kHz, Float32, mono)
Web Audio API ScriptProcessor
    ↓ (Convert to PCM16)
WebSocket to Backend
    ↓
FastAPI Server (realtime_server.py)
    ↓ (Resample 48kHz → 24kHz)
Audio Quality Detection
    ↓
OpenAI Realtime API
    ↓ (whisper-1 transcription)
Transcription Results
```

## Browser-Side Audio Capture

### Microphone Constraints

**File**: [`static/main.js:577-585`](static/main.js#L577-L585)

```javascript
stream = await navigator.mediaDevices.getUserMedia({
    audio: {
        channelCount: 1,              // Mono audio
        echoCancellation: true,        // Remove echo
        noiseSuppression: true,        // Reduce background noise
        autoGainControl: false         // CRITICAL: Disabled - see below
    }
});
```

### Why Auto Gain Control (AGC) is Disabled

**IMPORTANT**: Browser AGC is **disabled** (`autoGainControl: false`) for the following reasons:

1. **Preserves System Settings**: Users configure their microphone volume at the OS level (System Settings → Sound → Input). Browser AGC would override these settings.

2. **Prevents Unexpected Volume Reduction**: During debugging, we discovered that `autoGainControl: true` caused audio amplitude to drop from expected values (>1000) to extremely low values (5-57), causing transcription failures.

3. **User Control**: Professional users with proper microphone setups expect their hardware/system settings to be respected, not overridden by browser heuristics.

4. **Predictable Behavior**: With AGC disabled, audio levels are consistent and predictable based on system configuration.

**Trade-off**: Users must manually configure their microphone volume at the system level. The application provides warnings when volume is too low (see Audio Quality Detection below).

### Audio Format Conversion

**File**: [`static/main.js:189-192`](static/main.js#L189-L192)

```javascript
// Convert float32 (-1.0 to 1.0) to PCM16 (-32768 to 32767)
for (let i = 0; i < inputData.length; i++) {
    pcmData[i] = Math.max(-32768, Math.min(32767, Math.floor(inputData[i] * 32767)));
}
```

Browser audio is captured as Float32 values in the range [-1.0, 1.0], which are converted to PCM16 format (16-bit signed integers) before sending to the server.

## Server-Side Audio Processing

### Audio Resampling

**File**: [`realtime_server.py:179-211`](realtime_server.py#L179-L211)

OpenAI's Realtime API requires audio at **24kHz sample rate**, but browsers typically capture at **48kHz**. We downsample using `scipy.signal.resample_poly`:

```python
def process_audio_chunk(self, audio_data):
    # Convert binary PCM16 to numpy array
    pcm_data = np.frombuffer(audio_data, dtype=np.int16)

    # Convert to float32 for precision during resampling
    float_data = pcm_data.astype(np.float32) / 32768.0

    # Resample: 48kHz → 24kHz (reduces data by half)
    resampled_data = scipy.signal.resample_poly(
        float_data,
        self.target_sample_rate,  # 24000
        self.source_sample_rate   # 48000
    )

    # Convert back to PCM16
    resampled_int16 = (resampled_data * 32768.0).clip(-32768, 32767).astype(np.int16)

    return resampled_int16.tobytes(), quality_status, max_amplitude
```

**Why resample_poly?**: This method preserves audio quality better than simple decimation and prevents aliasing artifacts.

## Audio Quality Detection

### Quality Metrics

**File**: [`realtime_server.py:182-197`](realtime_server.py#L182-L197)

The system analyzes each audio chunk to detect quality issues:

```python
# Check audio quality
max_amplitude = np.max(np.abs(pcm_data))
rms = np.sqrt(np.mean(pcm_data.astype(np.float32) ** 2))

# Determine audio quality status
if max_amplitude == 0:
    quality_status = "silent"
    logger.warning("⚠️ Silent audio detected! All samples are zero.")
elif max_amplitude < 100:
    quality_status = "too_quiet"
    logger.warning(f"⚠️ Very quiet audio detected! Max amplitude: {max_amplitude} (expected > 1000)")
else:
    quality_status = "ok"
    logger.debug(f"Audio quality: max_amp={max_amplitude}, rms={rms:.1f}")
```

### Quality Thresholds

| Status | Max Amplitude | Description |
|--------|---------------|-------------|
| `silent` | 0 | No audio signal detected (all zeros) |
| `too_quiet` | 1-99 | Audio too quiet for reliable transcription |
| `ok` | ≥100 | Acceptable audio level (typically >1000 for normal speech) |

**Expected Range**: Normal speech at comfortable microphone distance typically produces amplitudes of **1,000 to 15,000** (PCM16 range).

### Developer-Friendly Audio Monitoring

**Browser Console Logging** - **File**: [`static/main.js:199-202`](static/main.js#L199-L202)

Audio levels are logged to the browser console every ~2 seconds for debugging:

```javascript
// Log audio levels every ~2 seconds (at 48kHz with 4096 buffer = ~12 chunks/sec)
if (Math.random() < 0.08) {
    console.log(`🎤 Audio levels - Float: ${maxFloat.toFixed(3)} (0.0-1.0), PCM16: ${maxPCM} (expected >1000 for speech)`);
}
```

**Server-Side Logging** - **File**: [`realtime_server.py:182-197`](realtime_server.py#L182-L197)

The server logs audio quality warnings when issues are detected:
- `logger.warning()` for silent or too-quiet audio
- `logger.debug()` for normal audio with amplitude details

**Why console logging instead of popups?**: Since the primary users are experienced developers, console logging provides debugging information without interrupting the user experience. Users can monitor audio levels in the browser DevTools console.

## OpenAI Integration

### Transcription Model

**File**: [`openai_realtime_client.py:56-66`](openai_realtime_client.py#L56-L66)

We use **gpt-4o-transcribe** for transcription:

```python
if session_mode == "transcription":
    session_config_payload["input_audio_transcription"] = {
        "model": "gpt-4o-transcribe"
    }
```

**Note**: This model is part of OpenAI's Realtime API and is optimized for real-time transcription. Alternative models like `whisper-1` can be used if needed.

### Error Handling

**File**: [`realtime_server.py:486-499`](realtime_server.py#L486-L499)

The system handles transcription failures from OpenAI:

```python
async def handle_transcription_failed(data):
    """Handle transcription failure events from OpenAI"""
    logger.error(f"⚠️ TRANSCRIPTION FAILED: {json.dumps(data, indent=2)}")

    error_msg = data.get("error", {})
    item_id = data.get("item_id", "unknown")

    # Send error to frontend
    await websocket.send_text(json.dumps({
        "type": "error",
        "content": f"Transcription failed: {error_msg.get('message', 'Unknown error')}",
        "details": error_msg
    }, ensure_ascii=False))
```

This handler captures detailed error information from OpenAI's `conversation.item.input_audio_transcription.failed` event, which would otherwise be silently lost.

## Troubleshooting

### Problem: No transcription appearing

**Possible Causes**:
1. **Low microphone volume** → Increase system microphone volume
2. **Rate limiting** → Wait 10-15 minutes between tests during development
3. **Microphone permissions** → Check browser permissions
4. **Silent audio** → Verify microphone is working in system settings

### Problem: Low audio levels (shown in browser console)

**Check browser console** for messages like: `🎤 Audio levels - Float: 0.003 (0.0-1.0), PCM16: 57 (expected >1000 for speech)`

**Solutions**:
1. **Increase system volume**: macOS → System Settings → Sound → Input → adjust Input volume slider
2. **Select different microphone**: Choose a microphone with higher sensitivity
3. **Move closer to mic**: Ensure proper distance from microphone
4. **Test with Voice Memo**: macOS Voice Memo app can verify microphone is working properly
5. **Check browser permissions**: Ensure the correct microphone is selected in browser settings

### Problem: 429 Rate Limit Error

**Cause**: Too many connection attempts in a short time period.

**Solution**:
- Wait 10-15 minutes for rate limits to reset
- OpenAI uses sliding window rate limits (not fixed intervals)
- Each connection attempt counts against your limit, even if it fails
- Test sparingly during development (2-3 minute gaps between attempts)

## Technical Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample Rate (Browser) | 48,000 Hz | Standard browser capture rate |
| Sample Rate (OpenAI) | 24,000 Hz | Required by OpenAI Realtime API |
| Format | PCM16 (16-bit signed) | Mono channel |
| Encoding | Base64 | For WebSocket transmission |
| Buffer Size | 4,096 samples | ScriptProcessor buffer |
| Auto Gain Control | **Disabled** | User controls volume at OS level |
| Echo Cancellation | Enabled | Removes audio output echo |
| Noise Suppression | Enabled | Reduces background noise |

## Best Practices

### For Users

1. **Configure microphone at system level** before using the app
2. **Test microphone** with native apps (e.g., Voice Memo on macOS)
3. **Position microphone** properly for clear speech capture
4. **Avoid rapid testing** during development to prevent rate limits

### For Developers

1. **Monitor audio quality** using the built-in detection system
2. **Handle transcription failures** explicitly (don't rely on silent failures)
3. **Use debug logging** sparingly in production (`logger.debug()` for verbose output)
4. **Respect rate limits** when testing (use sliding window approach)
5. **Keep AGC disabled** unless users specifically request it

## Related Files

- [`static/main.js`](../static/main.js) - Browser audio capture and WebSocket client
- [`realtime_server.py`](../realtime_server.py) - Server-side audio processing and quality detection
- [`openai_realtime_client.py`](../openai_realtime_client.py) - OpenAI Realtime API integration
- [`audio_processor.py`](../audio_processor.py) - Audio resampling utilities (AudioProcessor class is in realtime_server.py)

## Version History

- **2026-01-05**: Initial documentation
  - Disabled browser Auto Gain Control
  - Implemented audio quality detection with console logging (developer-friendly)
  - Using gpt-4o-transcribe transcription model (can be switched to whisper-1 if needed)
  - Added transcription failure error handling
