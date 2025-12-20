"""
x.ai Voice Agent API client implementation.
Implements the RealtimeClientBase interface for x.ai's Grok Voice Agent API.
"""
import websockets
import json
import base64
import logging
import asyncio
from typing import Optional, Callable, Dict, List
from prompts import PROMPTS
from config import XAI_REALTIME_MODALITIES
from realtime_client_base import RealtimeClientBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class XAIRealtimeAudioTextClient(RealtimeClientBase):
    """x.ai Voice Agent API client"""
    
    def __init__(self, api_key: str, voice: str = "Ara"):
        """
        Initialize x.ai realtime client.
        
        Args:
            api_key: x.ai API key
            voice: Voice to use (Ara, Rex, Sal, Eve, Leo). Default: Ara
        """
        super().__init__(api_key)
        self.voice = voice
        self.base_url = "wss://api.x.ai/v1/realtime"
        self.last_audio_time = None
        self.auto_commit_interval = 5
        
    async def connect(self, modalities: List[str] = None, session_mode: str = "conversation"):
        """
        Connect to x.ai's realtime API and configure the session.
        
        Args:
            modalities: List of modalities (not used by x.ai, kept for compatibility)
            session_mode: Mode for the session ("conversation" or "transcription")
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Support both websockets param names across versions
        try:
            self.ws = await websockets.connect(
                self.base_url,
                ssl=True,
                extra_headers=headers,
            )
        except TypeError:
            # Fallback for newer versions where the kwarg is 'additional_headers'
            self.ws = await websockets.connect(
                self.base_url,
                ssl=True,
                additional_headers=headers,
            )
        
        # Wait for initial message (conversation.created)
        response = await self.ws.recv()
        response_data = json.loads(response)
        
        if response_data.get("type") == "conversation.created":
            conversation_id = response_data.get("conversation", {}).get("id")
            logger.info(f"Conversation created with ID: {conversation_id}")
        elif response_data.get("type") == "ping":
            # x.ai may send ping first, wait for conversation.created
            logger.debug("Received ping, waiting for conversation.created...")
            response = await self.ws.recv()
            response_data = json.loads(response)
            if response_data.get("type") == "conversation.created":
                conversation_id = response_data.get("conversation", {}).get("id")
                logger.info(f"Conversation created with ID: {conversation_id}")
        
        # Configure session
        session_config_payload = {
            "voice": self.voice,
            "turn_detection": None,  # Manual turn detection (like OpenAI)
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000  # 24kHz default
                    }
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    }
                }
            }
        }
        
        if session_mode == "transcription":
            # No instructions for transcription mode
            logger.info("Configuring session for transcription mode.")
        else:  # Default to conversation mode
            session_config_payload["instructions"] = PROMPTS.get('paraphrase-gpt-realtime-enhanced', '')
            logger.info("Configuring session for conversation mode with transcription and no turn detection.")
        
        # Send session configuration
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": session_config_payload
        }, ensure_ascii=False))
        
        # Wait for session.updated confirmation (or ping)
        try:
            confirmation = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            confirmation_data = json.loads(confirmation)
            if confirmation_data.get("type") in ["session.updated", "ping"]:
                logger.info(f"Session updated: {confirmation_data.get('type')}")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for session.updated confirmation")
        
        # Register the default handler
        self.register_handler("default", self.default_handler)
        
        # Start the receiver coroutine
        self.receive_task = asyncio.create_task(self.receive_messages())
    
    async def receive_messages(self):
        """Receive and process messages from x.ai WebSocket"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                message_type = data.get("type", "default")
                
                # Map x.ai message types to handlers
                handler = self.handlers.get(message_type, self.handlers.get("default"))
                if handler:
                    await handler(data)
                else:
                    logger.warning(f"No handler for message type: {message_type}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"x.ai WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error in receive_messages: {e}", exc_info=True)
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to x.ai API"""
        if self._is_ws_open():
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_data).decode('utf-8')
            }))
        else:
            logger.error("WebSocket is not open. Cannot send audio.")
    
    async def commit_audio(self):
        """Commit the audio buffer and notify x.ai"""
        if self._is_ws_open():
            commit_message = json.dumps({"type": "input_audio_buffer.commit"})
            await self.ws.send(commit_message)
            logger.info("Sent input_audio_buffer.commit message to x.ai")
        else:
            logger.error("WebSocket is not open. Cannot commit audio.")
    
    async def clear_audio_buffer(self):
        """Clear the audio buffer"""
        if self._is_ws_open():
            clear_message = json.dumps({"type": "input_audio_buffer.clear"})
            await self.ws.send(clear_message)
            logger.info("Sent input_audio_buffer.clear message to x.ai")
        else:
            logger.error("WebSocket is not open. Cannot clear audio buffer.")
    
    async def start_response(self, instructions: str, modalities: List[str] = None):
        """
        Start a new response with given instructions
        
        Args:
            instructions: Instructions for the response
            modalities: List of output modalities. Defaults to XAI_REALTIME_MODALITIES config.
                       Use ["text"] for text-only output, ["text", "audio"] for both.
        """
        if self._is_ws_open():
            # Use provided modalities or fall back to config
            output_modalities = modalities or XAI_REALTIME_MODALITIES
            await self.ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": output_modalities
                }
            }))
            logger.info(f"Started response with modalities: {output_modalities}, instructions: {instructions[:50]}...")
        else:
            logger.error("WebSocket is not open. Cannot start response.")
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.ws:
            await self.ws.close()
            logger.info("Closed x.ai WebSocket connection")
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
    
    async def default_handler(self, data: dict):
        """Default handler for unhandled message types"""
        message_type = data.get("type", "unknown")
        logger.warning(f"Unhandled message type received from x.ai: {message_type}")

