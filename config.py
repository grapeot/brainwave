import os

# Global configuration for OpenAI realtime model
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")


# Modalities for WebRTC sessions (text-only output by default)
OPENAI_REALTIME_MODALITIES = os.getenv("OPENAI_REALTIME_MODALITIES", "text").split(",")

# Ephemeral session token TTL (seconds)
OPENAI_REALTIME_SESSION_TTL_SEC = int(os.getenv("OPENAI_REALTIME_SESSION_TTL_SEC", "60"))


