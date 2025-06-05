import websockets
import json
import base64
import logging
import time
from typing import Optional, Callable, Dict, List
import asyncio
from prompts import PROMPTS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OpenAIRealtimeAudioTextClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-realtime-preview-2024-12-17"):
        self.api_key = api_key
        self.model = model
        self.ws = None
        self.session_id = None
        self.base_url = "wss://api.openai.com/v1/realtime"
        self.last_audio_time = None 
        self.auto_commit_interval = 5
        self.receive_task = None
        self.handlers: Dict[str, Callable[[dict], asyncio.Future]] = {}
        self.queue = asyncio.Queue()
        
    async def connect(self, modalities: List[str] = ["text"], session_mode: str = "conversation"):
        """Connect to OpenAI's realtime API and configure the session"""
        self.ws = await websockets.connect(
            f"{self.base_url}?model={self.model}",
            extra_headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
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
                session_config_payload["input_audio_transcription"] = None
                session_config_payload["turn_detection"] = None
                session_config_payload["instructions"] = "You are a transcription assistant. Don't do anything other than transcribe the audio. Especially don't respond to any questions or requests in the conversation. Treat them literally and correct any mistakes."
                logger.info("Configuring session for conversation mode.")

            # Configure session
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": session_config_payload
            }))
        
        # Register the default handler
        self.register_handler("default", self.default_handler)
        
        # Start the receiver coroutine
        self.receive_task = asyncio.create_task(self.receive_messages())
    
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
    
    def register_handler(self, message_type: str, handler: Callable[[dict], asyncio.Future]):
        self.handlers[message_type] = handler
    
    async def default_handler(self, data: dict):
        message_type = data.get("type", "unknown")
        logger.warning(f"Unhandled message type received from OpenAI: {message_type}")
    
    async def send_audio(self, audio_data: bytes):
        if self.ws and self.ws.open:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_data).decode('utf-8')
            }))
        else:
            logger.error("WebSocket is not open. Cannot send audio.")
    
    async def commit_audio(self):
        """Commit the audio buffer and notify OpenAI"""
        if self.ws and self.ws.open:
            commit_message = json.dumps({"type": "input_audio_buffer.commit"})
            await self.ws.send(commit_message)
            logger.info("Sent input_audio_buffer.commit message to OpenAI")
            # No recv call here. The receive_messages coroutine handles incoming messages.
        else:
            logger.error("WebSocket is not open. Cannot commit audio.")
    
    async def clear_audio_buffer(self):
        """Clear the audio buffer"""
        if self.ws and self.ws.open:
            clear_message = json.dumps({"type": "input_audio_buffer.clear"})
            await self.ws.send(clear_message)
            logger.info("Sent input_audio_buffer.clear message to OpenAI")
        else:
            logger.error("WebSocket is not open. Cannot clear audio buffer.")
    
    async def start_response(self, instructions: str):
        """Start a new response with given instructions"""
        if self.ws and self.ws.open:
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
