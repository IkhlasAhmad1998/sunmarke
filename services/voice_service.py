"""Voice transcription helpers using Deepgram.

This module exposes a single async helper `transcribe_audio` that
converts a local audio file path to text. It uses the Deepgram async
client and returns an empty string on failure to keep the UI flow
simple and non-blocking.
"""

import os
from deepgram import AsyncDeepgramClient
from config import settings

# Initialize the ASYNC client
DEEPGRAM_API_KEY = settings.DEEPGRAM_API_KEY
dg_client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)


async def transcribe_audio(audio_path):
    """
    Transcribes audio using Deepgram Nova-3 (Asynchronous).
    Uses the modern v1.media.transcribe_file path.
    """
    try:
        if not audio_path or not os.path.exists(audio_path):
            return ""

        with open(audio_path, "rb") as audio:
            audio_data = audio.read()

        # In Async v5+, use 'request=' keyword argument for the bytes buffer
        response = await dg_client.listen.v1.media.transcribe_file(
            request=audio_data,
            model="nova-3",
            smart_format=True,
            language="en-US"
        )

        return response.results.channels[0].alternatives[0].transcript

    except Exception as e:
        print(f"STT Error: {e}")
        return f"Error: {str(e)}"
