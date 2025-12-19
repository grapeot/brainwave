import pytest
import asyncio
import json
import websockets
from unittest.mock import AsyncMock, patch, MagicMock
from openai_realtime_client import OpenAIRealtimeAudioTextClient
from prompts import PROMPTS

@pytest.fixture
def api_key():
    return "test_api_key"

@pytest.fixture
def client(api_key):
    return OpenAIRealtimeAudioTextClient(api_key)

@pytest.mark.asyncio
async def test_connect_success(client):
    mock_ws = AsyncMock()
    mock_ws.recv.return_value = json.dumps({
        "type": "session.created",
        "session": {"id": "test_session_id"}
    })
    
    with patch('websockets.connect', AsyncMock(return_value=mock_ws)):
        await client.connect()
        
        assert client.session_id == "test_session_id"
        assert client.ws == mock_ws
        assert len(client.handlers) > 0
        assert "default" in client.handlers
        
        # Verify the session update message was sent for conversation mode with server_vad and transcription
        expected_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "input_audio_format": "pcm16",
                "input_audio_transcription": {"model": "gpt-4o-transcribe"},
                "turn_detection": None,
                "instructions": PROMPTS['paraphrase-gpt-realtime-enhanced']
            }
        }
        mock_ws.send.assert_awaited_with(json.dumps(expected_update, ensure_ascii=False))

@pytest.mark.asyncio
async def test_send_audio(client):
    mock_ws = AsyncMock()
    mock_ws.open = True
    mock_ws.closed = False
    client.ws = mock_ws
    
    test_audio = b"test_audio_data"
    await client.send_audio(test_audio)
    
    expected_message = {
        "type": "input_audio_buffer.append",
        "audio": "dGVzdF9hdWRpb19kYXRh"  # base64 encoded test_audio_data
    }
    mock_ws.send.assert_awaited_with(json.dumps(expected_message))

@pytest.mark.asyncio
async def test_commit_audio(client):
    mock_ws = AsyncMock()
    mock_ws.open = True
    mock_ws.closed = False
    client.ws = mock_ws
    
    await client.commit_audio()
    
    expected_message = {"type": "input_audio_buffer.commit"}
    mock_ws.send.assert_awaited_with(json.dumps(expected_message))

@pytest.mark.asyncio
async def test_clear_audio_buffer(client):
    mock_ws = AsyncMock()
    mock_ws.open = True
    mock_ws.closed = False
    client.ws = mock_ws
    
    await client.clear_audio_buffer()
    
    expected_message = {"type": "input_audio_buffer.clear"}
    mock_ws.send.assert_awaited_with(json.dumps(expected_message))

@pytest.mark.asyncio
async def test_start_response(client):
    mock_ws = AsyncMock()
    mock_ws.open = True
    mock_ws.closed = False
    client.ws = mock_ws
    
    test_instructions = "test instructions"
    await client.start_response(test_instructions)
    
    expected_message = {
        "type": "response.create",
        "response": {
            "modalities": ["text"],
            "instructions": test_instructions
        }
    }
    mock_ws.send.assert_awaited_with(json.dumps(expected_message))

@pytest.mark.asyncio
async def test_close(client):
    mock_ws = AsyncMock()
    client.ws = mock_ws
    client.receive_task = asyncio.create_task(asyncio.sleep(0))
    
    await client.close()
    
    mock_ws.close.assert_awaited_once()
    assert client.receive_task.cancelled()

@pytest.mark.asyncio
async def test_receive_messages(client):
    mock_ws = AsyncMock()
    test_message = {"type": "test_type", "data": "test_data"}
    mock_ws.__aiter__.return_value = [json.dumps(test_message)]
    client.ws = mock_ws
    
    # Create a mock handler and register it
    mock_handler = AsyncMock()
    client.register_handler("test_type", mock_handler)
    
    # Start receive_messages
    receive_task = asyncio.create_task(client.receive_messages())
    await asyncio.sleep(0.1)  # Give some time for the message to be processed
    
    # Verify the handler was called with the correct message
    mock_handler.assert_awaited_once_with(test_message)
    
    # Clean up
    receive_task.cancel()
    try:
        await receive_task
    except asyncio.CancelledError:
        pass
