import asyncio
import json
import os
import numpy as np
from fastapi import FastAPI, WebSocket, Request, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import uvicorn
import logging
from prompts import PROMPTS
from openai_realtime_client import OpenAIRealtimeAudioTextClient
from starlette.websockets import WebSocketState
import wave
import datetime
import scipy.signal
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Generator
from llm_processor import get_llm_processor
from datetime import datetime, timedelta
import time

# Import the refactored Gemini client function
from gemini_client import generate_transcription_stream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for request and response schemas
class ReadabilityRequest(BaseModel):
    text: str = Field(..., description="The text to improve readability for.")

class ReadabilityResponse(BaseModel):
    enhanced_text: str = Field(..., description="The text with improved readability.")

class CorrectnessRequest(BaseModel):
    text: str = Field(..., description="The text to check for factual correctness.")

class CorrectnessResponse(BaseModel):
    analysis: str = Field(..., description="The factual correctness analysis.")

class AskAIRequest(BaseModel):
    text: str = Field(..., description="The question to ask AI.")

class AskAIResponse(BaseModel):
    answer: str = Field(..., description="AI's answer to the question.")

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in environment variables.")
    raise EnvironmentError("OPENAI_API_KEY is not set.")

# Initialize with a default model
llm_processor = get_llm_processor("gpt-4o")  # Default processor

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_realtime_page(request: Request):
    return FileResponse("static/realtime.html")

@app.get("/transcribe", response_class=HTMLResponse)
async def get_transcribe_page(request: Request):
    return FileResponse("static/transcribe.html")

class AudioProcessor:
    def __init__(self, target_sample_rate=24000):
        self.target_sample_rate = target_sample_rate
        self.source_sample_rate = 48000  # Most common sample rate for microphones
        
    def process_audio_chunk(self, audio_data):
        # Convert binary audio data to Int16 array
        pcm_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # Convert to float32 for better precision during resampling
        float_data = pcm_data.astype(np.float32) / 32768.0
        
        # Resample from 48kHz to 24kHz
        resampled_data = scipy.signal.resample_poly(
            float_data, 
            self.target_sample_rate, 
            self.source_sample_rate
        )
        
        # Convert back to int16 while preserving amplitude
        resampled_int16 = (resampled_data * 32768.0).clip(-32768, 32767).astype(np.int16)
        return resampled_int16.tobytes()

    def save_audio_buffer(self, audio_buffer, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wf.setframerate(self.target_sample_rate)
            wf.writeframes(b''.join(audio_buffer))
        logger.info(f"Saved audio buffer to {filename}")

@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("New WebSocket connection attempt")
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    # Add initial status update here
    await websocket.send_text(json.dumps({
        "type": "status",
        "status": "idle"  # Set initial status to idle (blue)
    }))
    
    client = None
    audio_processor = AudioProcessor()
    audio_buffer = []
    recording_stopped = asyncio.Event()
    openai_ready = asyncio.Event()
    pending_audio_chunks = []
    
    async def initialize_openai():
        nonlocal client
        try:
            # Clear the ready flag while initializing
            openai_ready.clear()
            
            client = OpenAIRealtimeAudioTextClient(os.getenv("OPENAI_API_KEY"))
            await client.connect()
            logger.info("Successfully connected to OpenAI client")
            
            # Register handlers after client is initialized
            client.register_handler("session.updated", lambda data: handle_generic_event("session.updated", data))
            client.register_handler("input_audio_buffer.cleared", lambda data: handle_generic_event("input_audio_buffer.cleared", data))
            client.register_handler("input_audio_buffer.speech_started", lambda data: handle_generic_event("input_audio_buffer.speech_started", data))
            client.register_handler("rate_limits.updated", lambda data: handle_generic_event("rate_limits.updated", data))
            client.register_handler("response.output_item.added", lambda data: handle_generic_event("response.output_item.added", data))
            client.register_handler("conversation.item.created", lambda data: handle_generic_event("conversation.item.created", data))
            client.register_handler("response.content_part.added", lambda data: handle_generic_event("response.content_part.added", data))
            client.register_handler("response.text.done", lambda data: handle_generic_event("response.text.done", data))
            client.register_handler("response.content_part.done", lambda data: handle_generic_event("response.content_part.done", data))
            client.register_handler("response.output_item.done", lambda data: handle_generic_event("response.output_item.done", data))
            client.register_handler("response.done", lambda data: handle_response_done(data))
            client.register_handler("error", lambda data: handle_error(data))
            client.register_handler("response.text.delta", lambda data: handle_text_delta(data))
            client.register_handler("response.created", lambda data: handle_response_created(data))
            
            openai_ready.set()  # Set ready flag after successful initialization
            await websocket.send_text(json.dumps({
                "type": "status",
                "status": "connected"
            }))
            return True
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            openai_ready.clear()  # Ensure flag is cleared on failure
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": "Failed to initialize OpenAI connection"
            }))
            return False

    # Move the handler definitions here (before initialize_openai)
    async def handle_text_delta(data):
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({
                    "type": "text",
                    "content": data.get("delta", ""),
                    "isNewResponse": False
                }))
        except Exception as e:
            logger.error(f"Error in handle_text_delta: {str(e)}", exc_info=True)

    async def handle_response_created(data):
        await websocket.send_text(json.dumps({
            "type": "text",
            "content": "",
            "isNewResponse": True
        }))
        logger.info("Handled response.created")

    async def handle_error(data):
        error_msg = data.get("error", {}).get("message", "Unknown error")
        logger.error(f"OpenAI error: {error_msg}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "content": error_msg
        }))
        logger.info("Handled error message from OpenAI")

    async def handle_response_done(data):
        nonlocal client
        logger.info("Handled response.done")
        recording_stopped.set()
        
        if client:
            try:
                await client.close()
                client = None
                openai_ready.clear()
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "status": "idle"
                }))
                logger.info("Connection closed after response completion")
            except Exception as e:
                logger.error(f"Error closing client after response done: {str(e)}")

    async def handle_generic_event(event_type, data):
        logger.info(f"Handled {event_type} with data: {json.dumps(data, ensure_ascii=False)}")

    # Create a queue to handle incoming audio chunks
    audio_queue = asyncio.Queue()

    async def receive_messages():
        nonlocal client
        
        try:
            while True:
                if websocket.client_state == WebSocketState.DISCONNECTED:
                    logger.info("WebSocket client disconnected")
                    openai_ready.clear()
                    break
                    
                try:
                    # Add timeout to prevent infinite waiting
                    data = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                    
                    if "bytes" in data:
                        processed_audio = audio_processor.process_audio_chunk(data["bytes"])
                        if not openai_ready.is_set():
                            logger.debug("OpenAI not ready, buffering audio chunk")
                            pending_audio_chunks.append(processed_audio)
                        elif client:
                            await client.send_audio(processed_audio)
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "status": "connected"
                            }))
                            logger.debug(f"Sent audio chunk, size: {len(processed_audio)} bytes")
                        else:
                            logger.warning("Received audio but client is not initialized")
                            
                    elif "text" in data:
                        msg = json.loads(data["text"])
                        
                        if msg.get("type") == "start_recording":
                            # Update status to connecting while initializing OpenAI
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "status": "connecting"
                            }))
                            if not await initialize_openai():
                                continue
                            recording_stopped.clear()
                            pending_audio_chunks.clear()
                            
                            # Send any buffered chunks
                            if pending_audio_chunks and client:
                                logger.info(f"Sending {len(pending_audio_chunks)} buffered chunks")
                                for chunk in pending_audio_chunks:
                                    await client.send_audio(chunk)
                                pending_audio_chunks.clear()
                            
                        elif msg.get("type") == "stop_recording":
                            if client:
                                await client.commit_audio()
                                await client.start_response(PROMPTS['paraphrase-gpt-realtime'])
                                await recording_stopped.wait()
                                # Don't close the client here, let the disconnect timer handle it
                                # Update client status to connected (waiting for response)
                                await websocket.send_text(json.dumps({
                                    "type": "status",
                                    "status": "connected"
                                }))

                except asyncio.TimeoutError:
                    logger.debug("No message received for 30 seconds")
                    continue
                except Exception as e:
                    logger.error(f"Error in receive_messages loop: {str(e)}", exc_info=True)
                    break
                
        finally:
            # Cleanup when the loop exits
            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.error(f"Error closing client in receive_messages: {str(e)}")
            logger.info("Receive messages loop ended")

    async def send_audio_messages():
        while True:
            try:
                processed_audio = await audio_queue.get()
                if processed_audio is None:
                    break
                
                # Add validation
                if len(processed_audio) == 0:
                    logger.warning("Empty audio chunk received, skipping")
                    continue
                
                # Append the processed audio to the buffer
                audio_buffer.append(processed_audio)

                await client.send_audio(processed_audio)
                logger.info(f"Audio chunk sent to OpenAI client, size: {len(processed_audio)} bytes")
                
            except Exception as e:
                logger.error(f"Error in send_audio_messages: {str(e)}", exc_info=True)
                break

        # After processing all audio, set the event
        recording_stopped.set()

    # Start concurrent tasks for receiving and sending
    receive_task = asyncio.create_task(receive_messages())
    send_task = asyncio.create_task(send_audio_messages())

    try:
        # Wait for both tasks to complete
        await asyncio.gather(receive_task, send_task)
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}", exc_info=True)
    finally:
        if client:
            await client.close()
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except RuntimeError as e:
                logger.warning(f"Ignoring error during websocket close: {e}")
        logger.info("WebSocket connection closed for /api/v1/ws")

@app.websocket("/api/v1/ws/transcribe")
async def websocket_transcribe_endpoint(websocket: WebSocket):
    logger.info("New Transcription WebSocket connection attempt")
    await websocket.accept()
    logger.info("Transcription WebSocket connection accepted")

    await websocket.send_text(json.dumps({
        "type": "status",
        "status": "idle"
    }))

    openai_client: OpenAIRealtimeAudioTextClient | None = None
    audio_processor = AudioProcessor() # Assuming AudioProcessor is suitable or OpenAI handles 24kHz PCM16
    chunk_count = 0  # Counter for audio chunks
    
    # Audio buffering for 5-second intervals
    audio_buffer = []
    last_send_time = time.time()
    send_interval = 5.0  # Send every 5 seconds
    
    # Handlers for OpenAI transcription messages
    async def handle_partial_transcript(data: dict):
        transcript_text = data.get("text", "") # Assuming 'text' field based on typical OpenAI responses
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({
                "type": "transcription_update",
                "is_partial": True,
                "text": transcript_text
            }))
            logger.debug(f"Sent partial transcript: {transcript_text}")

    async def handle_final_transcript(data: dict):
        transcript_text = data.get("text", "") # Assuming 'text' field
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({
                "type": "transcription_update",
                "is_partial": False,
                "text": transcript_text
            }))
            logger.info(f"Sent final transcript: {transcript_text}")
    
    async def handle_transcription_error(data: dict):
        error_msg = data.get("error", {}).get("message", "Unknown transcription error")
        logger.error(f"OpenAI transcription error: {error_msg}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": error_msg
            }))

    async def handle_transcription_delta(data: dict):
        # Handle partial transcription from conversation.item.input_audio_transcription.delta
        transcript_text = data.get("delta", "") # Check for 'delta' field
        if not transcript_text:
            transcript_text = data.get("text", "") # Fallback to 'text' field
        if websocket.client_state == WebSocketState.CONNECTED and transcript_text:
            await websocket.send_text(json.dumps({
                "type": "transcription_update",
                "is_partial": True,
                "text": transcript_text
            }))
            logger.debug(f"Sent partial transcript delta: {transcript_text}")

    async def handle_transcription_completed(data: dict):
        # Handle completed transcription from conversation.item.input_audio_transcription.completed
        transcript_text = data.get("transcript", "") # Check for 'transcript' field
        if not transcript_text:
            transcript_text = data.get("text", "") # Fallback to 'text' field
        if websocket.client_state == WebSocketState.CONNECTED and transcript_text:
            await websocket.send_text(json.dumps({
                "type": "transcription_update",
                "is_partial": False,
                "text": transcript_text
            }))
            logger.info(f"Sent completed transcript: {transcript_text}")

    async def handle_generic_transcription_event(event_type: str, data: dict):
        logger.info(f"Handled transcription event {event_type} with data: {json.dumps(data, ensure_ascii=False)}")

    async def send_buffered_audio():
        """Send accumulated audio buffer to OpenAI"""
        nonlocal audio_buffer, last_send_time
        
        if audio_buffer and openai_client:
            # Combine all buffered audio chunks
            combined_audio = b''.join(audio_buffer)
            await openai_client.send_audio(combined_audio)
            await openai_client.commit_audio()
            logger.info(f"Sent and committed {len(audio_buffer)} audio chunks ({len(combined_audio)} bytes) for transcription")
            
            # Clear buffer and update timestamp
            audio_buffer = []
            last_send_time = time.time()

    try:
        openai_client = OpenAIRealtimeAudioTextClient(api_key=OPENAI_API_KEY)
        
        # Connect in transcription mode
        await websocket.send_text(json.dumps({"type": "status", "status": "connecting_openai"}))
        await openai_client.connect(session_mode="transcription")
        logger.info("Successfully connected to OpenAI client for transcription")

        # Register handlers
        # IMPORTANT: Adjust message types if these are not what OpenAI uses for realtime transcription
        openai_client.register_handler("transcript.partial", handle_partial_transcript) # Placeholder
        openai_client.register_handler("transcript.final", handle_final_transcript)     # Placeholder
        openai_client.register_handler("text.delta", handle_partial_transcript) # More likely for streaming text
        openai_client.register_handler("text.final", handle_final_transcript) # More likely for final text
        openai_client.register_handler("error", handle_transcription_error)
        
        # Add handlers for the actual transcription message types we see in logs
        openai_client.register_handler("conversation.item.input_audio_transcription.delta", handle_transcription_delta)
        openai_client.register_handler("conversation.item.input_audio_transcription.completed", handle_transcription_completed)
        openai_client.register_handler("session.updated", lambda data: handle_generic_transcription_event("session.updated", data))
        openai_client.register_handler("input_audio_buffer.speech_started", lambda data: handle_generic_transcription_event("input_audio_buffer.speech_started", data))
        openai_client.register_handler("input_audio_buffer.committed", lambda data: handle_generic_transcription_event("input_audio_buffer.committed", data))
        openai_client.register_handler("conversation.item.created", lambda data: handle_generic_transcription_event("conversation.item.created", data))

        await websocket.send_text(json.dumps({"type": "status", "status": "connected_openai_transcribing"}))

        # Main loop to receive audio from client and send to OpenAI
        while True:
            if websocket.client_state == WebSocketState.DISCONNECTED:
                logger.info("Transcription WebSocket client disconnected by client.")
                break
            
            try:
                # Use a short timeout to check if we need to send buffered audio
                data = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                
                if "bytes" in data:
                    audio_chunk = data["bytes"]
                    processed_audio = audio_processor.process_audio_chunk(audio_chunk)
                    audio_buffer.append(processed_audio)
                    chunk_count += 1
                    
                    # Check if 5 seconds have passed since last send
                    current_time = time.time()
                    if current_time - last_send_time >= send_interval:
                        await send_buffered_audio()
                    
                    logger.debug(f"Buffered audio chunk {chunk_count}, buffer size: {len(audio_buffer)} chunks")
                    
                elif "text" in data:
                    message = json.loads(data["text"])
                    if message.get("type") == "stop_transcription":
                        # Send any remaining buffered audio before stopping
                        if audio_buffer:
                            await send_buffered_audio()
                        logger.info("Received stop_transcription message from client.")
                        break
                    # Handle other text messages if needed
                    logger.info(f"Received text message on transcribe endpoint: {message}")
                    
            except asyncio.TimeoutError:
                # Check if we need to send buffered audio even without new data
                current_time = time.time()
                if audio_buffer and current_time - last_send_time >= send_interval:
                    await send_buffered_audio()
                continue

    except websockets.exceptions.ConnectionClosedOK:
        logger.info("Transcription WebSocket connection closed normally by client.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Transcription WebSocket connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Error in transcription WebSocket endpoint: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))
    finally:
        # Send any remaining buffered audio before closing
        if audio_buffer and openai_client:
            try:
                await send_buffered_audio()
            except Exception as e:
                logger.error(f"Error sending final buffered audio: {e}")
        
        if openai_client:
            logger.info("Closing OpenAI client for transcription.")
            await openai_client.close()
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except RuntimeError as e: # Handle case where connection might already be closing
                logger.warning(f"Ignoring error during transcription websocket close: {e}")
        logger.info("Transcription WebSocket connection logic finished.")

@app.post(
    "/api/v1/readability",
    response_model=ReadabilityResponse,
    summary="Enhance Text Readability",
    description="Improve the readability of the provided text using GPT-4o."
)
async def enhance_readability(request: ReadabilityRequest):
    prompt = PROMPTS.get('readability-enhance')
    if not prompt:
        raise HTTPException(status_code=500, detail="Readability prompt not found.")

    try:
        async def text_generator():
            # Use gpt-4o specifically for readability
            async for part in llm_processor.process_text(request.text, prompt, model="gpt-4o"):
                yield part

        return StreamingResponse(text_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error enhancing readability: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing readability enhancement.")

@app.post(
    "/api/v1/ask_ai",
    response_model=AskAIResponse,
    summary="Ask AI a Question",
    description="Ask AI to provide insights using O1-mini model."
)
def ask_ai(request: AskAIRequest):
    prompt = PROMPTS.get('ask-ai')
    if not prompt:
        raise HTTPException(status_code=500, detail="Ask AI prompt not found.")

    try:
        # Use o3-mini specifically for ask_ai
        answer = llm_processor.process_text_sync(request.text, prompt, model="gpt-4.1")
        return AskAIResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error processing AI question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing AI question.")

@app.post(
    "/api/v1/correctness",
    response_model=CorrectnessResponse,
    summary="Check Factual Correctness",
    description="Analyze the text for factual accuracy using GPT-4o."
)
async def check_correctness(request: CorrectnessRequest):
    prompt = PROMPTS.get('correctness-check')
    if not prompt:
        raise HTTPException(status_code=500, detail="Correctness prompt not found.")

    try:
        async def text_generator():
            async for part in llm_processor.process_text(request.text, prompt, model="gpt-4o-search-preview"):
                yield part

        return StreamingResponse(text_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error checking correctness: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing correctness check.")

# Pydantic model for documenting Gemini transcription SSE data
class GeminiTranscriptionSSEData(BaseModel):
    text_chunk: str | None = None
    error: str | None = None

@app.post(
    "/api/v1/transcribe_gemini",
    summary="Transcribe Audio with Gemini (SSE)",
    description=(
        "Upload an audio file (M4A format assumed by the backend Gemini client) to be transcribed using Google's Gemini model." 
        "The transcription will be streamed back to the client using Server-Sent Events (SSE).\n\n"
        "**SSE Event Format:**\n\n"
        "Each SSE event will typically be a message event (default event type):\n"
        "```\n"
        "data: {\"text_chunk\": \"some transcribed text\"}\n"
        "```\n\n"
        "If an error occurs during processing, an error event will be sent:\n"
        "```\n"
        "event: error\n"
        "data: {\"error\": \"Error message details\"}\n"
        "```\n\n"
        "The stream concludes when the connection is closed by the server after transcription is complete or an unrecoverable error occurs.\n"
        "The `GOOGLE_API_KEY` environment variable must be set on the server.\n"
        "The audio file is sent as part of a multipart/form-data request."
    )
)
async def transcribe_gemini_sse(request: Request, file: UploadFile = File(...)):
    logger.info(f"Received file for Gemini transcription: {file.filename}, content type: {file.content_type}")
    
    request_processed_successfully = False # Initialize here

    # Read the file content immediately to avoid I/O on closed file issues later
    # especially with how StreamingResponse might handle the file object lifecycle.

    try:
        # Read the entire file into memory immediately.
        audio_bytes = await file.read()
        # It's good practice to close the upload file explicitly after reading, 
        # though FastAPI might do this upon request completion.
        await file.close()
    except Exception as e:
        logger.error(f"Error reading or closing uploaded file {file.filename}: {e}", exc_info=True)
        # This error happens before SSE stream starts, so an HTTP error is more appropriate if we weren't committed to SSE for all comms.
        # For now, we'll let it fall through to the generator, which will yield an SSE error.
        # To make it more robust, we could return an HTTPException here.
        # However, to keep SSE error reporting consistent:
        async def error_sse_generator():
            error_payload = json.dumps({'error': f'Failed to read uploaded file: {str(e)}'})
            yield f"event: error\ndata: {error_payload}\n\n"
        return StreamingResponse(error_sse_generator(), media_type="text/event-stream")

    # This inner generator now takes the bytes and filename, not the UploadFile object.
    async def sse_event_generator_for_bytes(audio_data: bytes, filename_for_logging: str):
        nonlocal request_processed_successfully
        logger.info(f"Starting Gemini SSE generation for pre-read audio: {filename_for_logging}")
        try:
            prompt = PROMPTS.get("gemini-transcription")
            if not prompt:
                error_detail = "Gemini transcription prompt not found in prompts.py."
                logger.error(error_detail)
                yield f"event: error\ndata: {json.dumps({'error': error_detail}, ensure_ascii=False)}\n\n"
                return

            async for chunk in generate_transcription_stream(audio_data, prompt):
                json_payload = json.dumps({"text_chunk": chunk}, ensure_ascii=False)
                yield f"data: {json_payload}\n\n"
                request_processed_successfully = True # Mark as successful if at least one chunk is sent
        except ValueError as ve:
            logger.error(f"ValueError during Gemini transcription for {filename_for_logging}: {ve}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(ve)}, ensure_ascii=False)}\n\n"
        except RuntimeError as re:
            logger.error(f"RuntimeError during Gemini transcription for {filename_for_logging}: {re}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(re)}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Unexpected error during Gemini transcription stream for {filename_for_logging}: {e}", exc_info=True)
            # Send a generic error to the client
            yield f"event: error\ndata: {json.dumps({'error': 'An unexpected error occurred during transcription.'}, ensure_ascii=False)}\n\n"
        finally:
            logger.info(f"Closing SSE event generator for {filename_for_logging}. Success: {request_processed_successfully}")

    return StreamingResponse(sse_event_generator_for_bytes(audio_bytes, file.filename), media_type="text/event-stream")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=3005)
