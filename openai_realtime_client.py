import websockets
import json
import base64
import logging
import time
from typing import Optional, Callable, Dict, List
import asyncio
import os
from prompts import PROMPTS
from config import OPENAI_REALTIME_MODEL
from realtime_client_base import RealtimeClientBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OpenAIRealtimeAudioTextClient(RealtimeClientBase):
    def __init__(self, api_key: str, model: str = OPENAI_REALTIME_MODEL):
        super().__init__(api_key)
        self.model = model
        self.base_url = "wss://api.openai.com/v1/realtime"
        self.last_audio_time = None 
        self.auto_commit_interval = 5
        
    async def connect(self, modalities: List[str] = ["text"], session_mode: str = "conversation"):
        """Connect to OpenAI's realtime API and configure the session"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        # Support both websockets param names across versions: extra_headers (older) and additional_headers (newer)
        try:
            self.ws = await websockets.connect(
                f"{self.base_url}?model={self.model}",
                extra_headers=headers,
            )
        except TypeError:
            # Fallback for newer versions where the kwarg is 'additional_headers'
            self.ws = await websockets.connect(
                f"{self.base_url}?model={self.model}",
                additional_headers=headers,
            )
        
        # Wait for session creation
        response = await self.ws.recv()
        response_data = json.loads(response)
        if response_data["type"] == "session.created":
            self.session_id = response_data["session"]["id"]
            logger.info(f"Session created with ID: {self.session_id}")
            
            session_config_payload = {
                "modalities": modalities,
                "input_audio_format": "pcm16",
            }

            if session_mode == "transcription":
                session_config_payload["input_audio_transcription"] = {
                    "model": "gpt-4o-transcribe"
                }
                session_config_payload["turn_detection"] = None
                # No instructions for transcription mode
                logger.info("Configuring session for transcription mode.")
            else:  # Default to conversation mode
                # Disable server-side VAD; rely on manual buffering/commits
                session_config_payload["input_audio_transcription"] = {
                    "model": "gpt-4o-transcribe"
                }
                session_config_payload["turn_detection"] = None
                session_config_payload["instructions"] = PROMPTS['paraphrase-gpt-realtime-enhanced']
                logger.info("Configuring session for conversation mode with transcription and no turn detection.")

            # Configure session
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": session_config_payload
            }, ensure_ascii=False))
        
        # Register the default handler
        self.register_handler("default", self.default_handler)
        
        # Start the receiver coroutine
        self.receive_task = asyncio.create_task(self.receive_messages())

    
    async def send_instructions_audio(self):
        """Send the instructions.wav file as audio input to be appended to current buffer"""
        instructions_path = "instructions.wav"
        if not os.path.exists(instructions_path):
            logger.warning(f"Instructions audio file not found: {instructions_path}")
            return
            
        try:
            with open(instructions_path, "rb") as f:
                audio_data = f.read()
            
            # Send the instructions audio to the buffer (appends to existing user audio)
            await self.send_audio(audio_data)
            logger.info("Sent instructions audio to OpenAI buffer (appended to user audio)")
            
        except Exception as e:
            logger.error(f"Error sending instructions audio: {e}")
    
    async def receive_messages(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                message_type = data.get("type", "default")
                handler = self.handlers.get(message_type, self.handlers.get("default"))
                if handler:
                    await handler(data)
                else:
                    logger.warning(f"No handler for message type: {message_type}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"OpenAI WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error in receive_messages: {e}", exc_info=True)
    
    async def default_handler(self, data: dict):
        """Override default handler for OpenAI-specific logging"""
        message_type = data.get("type", "unknown")
        logger.warning(f"Unhandled message type received from OpenAI: {message_type}")
    
    async def send_audio(self, audio_data: bytes):
        if self._is_ws_open():
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_data).decode('utf-8')
            }))
        else:
            logger.error("WebSocket is not open. Cannot send audio.")
    
    async def commit_audio(self):
        """Commit the audio buffer and notify OpenAI"""
        if self._is_ws_open():
            commit_message = json.dumps({"type": "input_audio_buffer.commit"})
            await self.ws.send(commit_message)
            logger.info("Sent input_audio_buffer.commit message to OpenAI")
            # No recv call here. The receive_messages coroutine handles incoming messages.
        else:
            logger.error("WebSocket is not open. Cannot commit audio.")
    
    async def clear_audio_buffer(self):
        """Clear the audio buffer"""
        if self._is_ws_open():
            clear_message = json.dumps({"type": "input_audio_buffer.clear"})
            await self.ws.send(clear_message)
            logger.info("Sent input_audio_buffer.clear message to OpenAI")
        else:
            logger.error("WebSocket is not open. Cannot clear audio buffer.")
    
    async def start_response(self, instructions: str):
        """Start a new response with given instructions"""
        if self._is_ws_open():
            await self.ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "instructions": instructions
                }
            }))
            logger.info(f"Started response with instructions: {instructions}")
        else:
            logger.error("WebSocket is not open. Cannot start response.")
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.ws:
            await self.ws.close()
            logger.info("Closed OpenAI WebSocket connection")
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
