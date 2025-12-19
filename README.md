# Brainwave: Real-Time Speech Recognition and Summarization Tool

## Table of Contents

1. [Introduction](#introduction)
2. [Deployment](#deployment)
3. [Code Structure & Architecture](#code-structure--architecture)
4. [Testing](#testing)

---

## Introduction

### Background

In the era of rapid information exchange, capturing and organizing ideas swiftly is paramount. **Brainwave** addresses this need by providing a robust speech recognition input method that allows users to effortlessly input their thoughts, regardless of their initial organization. Leveraging advanced technologies, Brainwave transforms potentially messy and unstructured verbal inputs into coherent and logical summaries, enhancing productivity and idea management.

### Goals

- **Efficient Speech Recognition:** Enable users to quickly input ideas through speech, reducing the friction of manual typing.
- **Organized Summarization:** Automatically process and summarize spoken input into structured and logical formats.
- **Multilingual Support:** Cater to a diverse user base by supporting multiple languages, ensuring accessibility and convenience.

### Technical Advantages

1. **Real-Time Processing:**
   - **Low Latency:** Processes audio streams in real-time, providing immediate transcription and summarization, which is essential for maintaining the flow of thoughts.
   - **Continuous Interaction:** Unlike traditional batch processing systems, Brainwave offers seamless real-time interaction, ensuring that users receive timely response on their inputs.

2. **Multilingual Proficiency:**
   - **Diverse Language Support:** Handles inputs in multiple languages without the need for separate processing pipelines, enhancing versatility and user accessibility.
   - **Automatic Language Detection:** Identifies the language of the input automatically, streamlining the user experience.

3. **Sophisticated Text Processing:**
   - **Error Correction:** Utilizes advanced algorithms to identify and correct errors inherent in speech recognition, ensuring accurate transcriptions.
   - **Readability Enhancement:** Improves punctuation and structure of the transcribed text, making summaries clear and professional.
   - **Intent Recognition:** Understands the context and intent behind the spoken words, enabling the generation of meaningful summaries.

---

## Deployment

Deploying **Brainwave** involves setting up a Python-based environment, installing the necessary dependencies, and launching the server to handle real-time speech recognition and summarization. Follow the steps below to get started:

### Prerequisites

- **Python 3.8+**: Ensure that Python is installed on your system. You can download it from the [official website](https://www.python.org/downloads/).
- **Virtual Environment Tool**: It's recommended to use `venv` or `virtualenv` to manage project dependencies.

### Setup Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/grapeot/brainwave.git
   cd brainwave
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   ```

3. **Activate the Virtual Environment**

   - **On macOS/Linux:**

     ```bash
     source venv/bin/activate
     ```

   - **On Windows:**

     ```bash
     venv\Scripts\activate
     ```

4. **Install Dependencies**

   Ensure that you have `pip` updated, then install the required packages:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables**

   Brainwave requires the OpenAI API key to function. Set the `OPENAI_API_KEY` environment variable:

   - **On macOS/Linux:**

     ```bash
     export OPENAI_API_KEY='your-openai-api-key'
     ```

   - **On Windows (Command Prompt):**

     ```cmd
     set OPENAI_API_KEY=your-openai-api-key
     ```

   - **On Windows (PowerShell):**

     ```powershell
     $env:OPENAI_API_KEY="your-openai-api-key"
     ```

6. **Launch the Server**

   Start the FastAPI server using Uvicorn:

   ```bash
   uvicorn realtime_server:app --host 0.0.0.0 --port 3005
   ```

   The server will be accessible at `http://localhost:3005`.

7. **Access the Application**

   Open your web browser and navigate to `http://localhost:3005` to interact with Brainwave's speech recognition interface.

---

## Code Structure & Architecture

Understanding the architecture of **Brainwave** provides insights into its real-time processing capabilities and multilingual support. The project is organized into several key components, each responsible for distinct functionalities.

### 1. **Backend**

#### a. `realtime_server.py`

- **Framework:** Utilizes **FastAPI** to handle HTTP and WebSocket connections, offering high performance and scalability.
- **WebSocket Endpoint:** Establishes a `/ws` endpoint for real-time audio streaming between the client and server.
- **Audio Processing:**
  - **`AudioProcessor` Class:** Resamples incoming audio data from 48kHz to 24kHz to match OpenAI's requirements.
  - **Buffer Management:** Accumulates audio chunks for efficient processing and transmission.
- **Concurrency:** Employs `asyncio` to manage asynchronous tasks for receiving and sending audio data, ensuring non-blocking operations.
- **Logging:** Implements comprehensive logging to monitor connections, data flow, and potential errors.

#### b. `openai_realtime_client.py`

- **WebSocket Client:** Manages the connection to OpenAI's real-time API, facilitating the transmission of audio data and reception of transcriptions.
- **Session Management:** Handles session creation, updates, and closure, ensuring a stable and persistent connection.
- **Event Handlers:** Registers and manages handlers for various message types from OpenAI, allowing for customizable responses and actions based on incoming data.
- **Error Handling:** Incorporates robust mechanisms to handle and log connection issues or unexpected messages.

#### c. `prompts.py`

- **Prompt Definitions:** Contains a dictionary of prompts in both Chinese and English, tailored for tasks such as paraphrasing, readability enhancement, and generating insightful summaries.
- **Customization:** Allows for easy modification and extension of prompts to cater to different processing requirements or languages.

### 2. **Frontend**

#### a. `static/realtime.html`

- **User Interface:** Provides a clean and responsive UI for users to interact with Brainwave, featuring:
  - **Recording Controls:** A toggle button to start and stop audio recording.
  - **Transcript Display:** A section to display the transcribed and summarized text in real-time.
  - **Copy Functionality:** Enables users to easily copy the summarized text.
  - **Timer:** Visual feedback to indicate recording duration.

- **Styling:** Utilizes CSS to ensure a modern and user-friendly appearance, optimized for both desktop and mobile devices.

- **Audio Handling:**
  - **Web Audio API:** Captures audio streams from the user's microphone, processes them into the required format, and handles chunking for transmission.
  - **WebSocket Integration:** Establishes and manages the WebSocket connection to the backend server, ensuring seamless data flow.

### 3. **Configuration**

#### a. `requirements.txt`

Lists all Python dependencies required to run Brainwave, ensuring that the environment is set up with compatible packages:

### 4. **Prompts & Text Processing**

Brainwave leverages a suite of predefined prompts to enhance text processing capabilities:

- **Paraphrasing:** Corrects speech-to-text errors and improves punctuation without altering the original meaning.
- **Readability Enhancement:** Improves the readability of transcribed text by adding appropriate punctuation and formatting.
- **Summary Generation:** Creates concise and logical summaries from the user's spoken input, making ideas easier to review and manage.

These prompts are meticulously crafted to ensure that the transcribed text is not only accurate but also contextually rich and user-friendly.

### 5. **Logging & Monitoring**

Comprehensive logging is integrated throughout the backend components to monitor:

- **Connection Status:** Tracks WebSocket connections and disconnections.
- **Data Transmission:** Logs the size and status of audio chunks being processed and sent.
- **Error Reporting:** Captures and logs any errors or exceptions, facilitating easier debugging and maintenance.

### 6. **Local Storage Replay Feature (IndexedDB)**

Brainwave includes a browser-based replay feature that allows users to replay their most recent recording session if recognition fails, network is interrupted, or backend errors occur.

#### Overview

- **Purpose:** Persist audio chunks and control events (start/stop) to browser local storage during recording, enabling replay without re-speaking.
- **Storage:** Uses IndexedDB (not localStorage) for binary data support and larger capacity (typically GB-level).
- **Scope:** Only keeps the most recent session by default to avoid unlimited storage usage. If local storage is unavailable, real-time recording still works (but replay is disabled).

#### Technical Details

**Storage Model:**
- **Database:** `brainwave-replay`, version `1`
- **Object Stores:**
  - `sessions` (keyPath: `id`, autoIncrement: true)
    - Fields: `id`, `createdAt`, `status` ("recording"|"completed"|"failed"), `sampleRate` (default 24000), `channelCount` (1), `durationMs`, `error`
  - `chunks` (keyPath: `id`, autoIncrement: true, index `sessionId`)
    - Fields: `sessionId`, `seq`, `deltaMs` (time offset from start for replay throttling), `kind` ("start"|"audio"|"stop"|"meta"), `payload` (Blob or ArrayBuffer), `byteLength`

**Storage Size Estimation:**
- 24000 sample rate, 16-bit mono => 48KB/second
- Only keeps the most recent N sessions (e.g., 5 sessions or total < 100MB)
- Oldest sessions and their chunks are deleted when quota is exceeded

**Recording Workflow:**
1. On `startRecording` success: Create session, write a `kind: "start"` record with `deltaMs = 0`
2. During `onaudioprocess`: When sending each 24k-sample chunk to backend, also call `appendChunk(sessionId, { seq, deltaMs, kind: "audio", payload: sendBuffer })`. `deltaMs` is calculated using `performance.now()` relative to session start to maintain original timing.
3. On `stopRecording`: After sending the last chunk, write a `kind: "stop"` record, update `sessions.status = "completed"`, and record `durationMs`
4. Cleanup: After recording ends, trigger `enforceQuota({ maxSessions: 5, maxBytes: 100 * 1024 * 1024 })`
5. Fallback: If IndexedDB initialization fails (no permission, incognito mode quota restrictions, etc.), set an in-memory fallback flag, only send in real-time without prompts, and disable the replay button

**Replay Workflow:**
1. **Session Selection:** Default to replaying the most recent `status = "completed"` session
2. **State Check:** If currently recording/generating, prompt user to stop before replaying
3. **Connection:** Create a new WebSocket connection (or reuse existing, ensuring backend accepts new start/stop sequences; prefer new connection)
4. **Replay Steps:**
   - Send `kind: "start"` control message (consistent with current protocol, e.g., `{type: "start_recording"}`)
   - Send audio chunks in recorded order
   - Audio chunk sending uses `setTimeout` based on `deltaMs` timing; if timing is too dense, set minimum interval of 5-10ms to prevent blocking
   - Send `kind: "stop"` control message at the end
5. **Error Handling:** On errors (WebSocket closed, backend 4xx/5xx, etc.), log and prompt but keep session data for retry

**Backend Compatibility:**
- WebSocket protocol remains consistent: `start_recording` control message -> binary PCM chunks -> `stop_recording`. Replay follows the same protocol.
- Backend allows multiple new connections from the same browser in a short time; if not allowed, frontend should wait for existing connection to return to idle before replaying

**UI/UX:**
- A circular refresh icon button (🔄) is displayed next to the Start button
- Button states: available/disabled/replaying
- During replay, the refresh icon rotates with animation
- Disable record button during replay
- If local storage is unavailable or quota insufficient, the replay button is disabled with a tooltip explaining the reason

---

## Testing

Brainwave includes a comprehensive test suite to ensure reliability and maintainability. The tests cover various components:

- **Audio Processing Tests:** Verify the correct handling of audio data, including resampling and buffer management.
- **LLM Integration Tests:** Test the integration with language models (GPT and Gemini) for text processing.
- **API Endpoint Tests:** Ensure the FastAPI endpoints work correctly, including streaming responses.
- **WebSocket Tests:** Verify real-time communication for audio streaming.

To run the tests:

1. **Install Test Dependencies**

   The test dependencies are included in `requirements.txt`. Make sure you have them installed:
   ```bash
   pip install pytest pytest-asyncio pytest-mock httpx
   ```

2. **Run Tests**

   ```bash
   # Run all tests
   pytest tests/

   # Run tests with verbose output
   pytest -v tests/

   # Run tests for a specific component
   pytest tests/test_audio_processor.py
   ```

3. **Test Environment**

   Tests use mocked API clients to avoid actual API calls. Set up the test environment variables:
   ```bash
   export OPENAI_API_KEY='test_key'  # For local testing
   export GOOGLE_API_KEY='test_key'  # For local testing
   ```

The test suite is designed to run without making actual API calls, making it suitable for CI/CD pipelines.

---

## Conclusion

**Brainwave** revolutionizes the way users capture and organize their ideas by providing a seamless speech recognition and summarization tool. Its real-time processing capabilities, combined with multilingual support and sophisticated text enhancement, make it an invaluable asset for anyone looking to efficiently manage their thoughts and ideas. Whether you're brainstorming, taking notes, or organizing project ideas, Brainwave ensures that your spoken words are transformed into clear, organized, and actionable summaries.

For any questions, contributions, or feedback, feel free to [open an issue](https://github.com/grapeot/brainwave/issues) or submit a pull request on the repository.

---

*Empower Your Ideas with Brainwave!*