import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global configuration for OpenAI realtime model
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-mini-2025-12-15")

# Modalities for WebRTC sessions (text-only output by default)
OPENAI_REALTIME_MODALITIES = os.getenv("OPENAI_REALTIME_MODALITIES", "text").split(",")

# Ephemeral session token TTL (seconds)
OPENAI_REALTIME_SESSION_TTL_SEC = int(os.getenv("OPENAI_REALTIME_SESSION_TTL_SEC", "60"))

# x.ai configuration
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_REALTIME_VOICE = os.getenv("XAI_REALTIME_VOICE", "Ara")  # Ara, Rex, Sal, Eve, Leo
XAI_REALTIME_SAMPLE_RATE = int(os.getenv("XAI_REALTIME_SAMPLE_RATE", "24000"))

# Provider selection
REALTIME_PROVIDER = os.getenv("REALTIME_PROVIDER", "openai")  # "openai" or "xai"

