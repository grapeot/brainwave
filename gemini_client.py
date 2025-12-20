# To run this code you need to install the following dependencies:
# pip install google-genai
#
# You also need to set the GOOGLE_API_KEY environment variable.
# export GOOGLE_API_KEY="YOUR_API_KEY"

import os
import asyncio # Added for the main test block
# Imports based on the user's original example
from google import genai
from google.genai import types
from prompts import PROMPTS


async def generate_transcription_stream(audio_bytes: bytes, prompt_text: str):
    """
    Generates a transcription for the given audio bytes using Gemini,
    strictly following the user's initial example structure.
    Yields text chunks as they are received.
    """
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            # Instead of printing, raise an error to be handled by the caller
            raise ValueError("Error: GOOGLE_API_KEY environment variable not set.")

        # Client initialization as per user's example
        client = genai.Client(api_key=api_key)

    except Exception as e:
        # Raise a more specific error or re-raise to be handled by the server
        print(f"An unexpected error occurred during client setup: {e}")
        print("Please ensure you have the 'google-genai' library installed (as per your example's comment)")
        print("and that GOOGLE_API_KEY is set. Also verify the SDK supports 'genai.Client'.")
        raise RuntimeError(f"Gemini client setup failed: {e}") from e

    # Model name as per user's example
    model_name = "gemini-2.5-flash-preview-05-20"

    try:
        # Constructing contents as per user's example structure for a single request turn
        contents_for_request = [
            types.Content(
                role="user",
                parts=[
                    # Using Part.from_bytes as in user's example for byte data
                    types.Part.from_bytes(mime_type="audio/ogg", data=audio_bytes), # Changed from audio/aac to audio/ogg
                    types.Part.from_text(text=prompt_text),
                ],
            )
        ]
        
        # GenerateContentConfig as per user's example
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

    except AttributeError as e:
        # Error in constructing parts, likely SDK issue
        print(f"Error constructing content parts or config: {e}.")
        print("This might indicate an issue with 'types.Part', 'types.Content', or 'types.GenerateContentConfig'.")
        print("Ensure your 'google-genai' library version matches the one implied by your example snippet.")
        raise RuntimeError(f"Gemini content preparation failed (AttributeError): {e}") from e
    except Exception as e:
        print(f"An unexpected error occurred while preparing content parts or config: {e}")
        raise RuntimeError(f"Gemini content preparation failed: {e}") from e

    try:
        # print(f"Sending request to Gemini model: {model_name} for audio transcription...") # Server will log this
        
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=contents_for_request,
            config=generate_content_config, # Using 'config' as per user's exact example
        )

        # Keep track of previous chunk to handle spacing intelligently
        last_chunk_ended_with_space = False
        
        # The user's example implies client.models.generate_content_stream returns a synchronous generator
        for chunk in response_stream: # Iterate synchronously
            print(f"[GEMINI DEBUG] Received chunk from Gemini API: {chunk}")
            if hasattr(chunk, 'text'):
                print(f"[GEMINI DEBUG] Chunk has text attribute: {repr(chunk.text)}")
                if chunk.text: 
                    processed_text = chunk.text
                    
                    # Handle spacing between chunks intelligently
                    if last_chunk_ended_with_space and processed_text.startswith(' '):
                        processed_text = processed_text.lstrip(' ')
                        print(f"[GEMINI DEBUG] Removed leading space from chunk")
                    
                    # Track if this chunk ends with space for next iteration
                    last_chunk_ended_with_space = processed_text.endswith(' ')
                    
                    print(f"[GEMINI DEBUG] Yielding processed text chunk: {repr(processed_text)}")
                    yield processed_text
                else:
                    print(f"[GEMINI DEBUG] Chunk text is empty or None")
            else:
                print(f"[GEMINI DEBUG] Chunk does not have text attribute")
                print(f"[GEMINI DEBUG] Chunk attributes: {dir(chunk)}")
        # print("\\n--- End of transcription ---") # Caller will know when stream ends
        print(f"[GEMINI DEBUG] Finished iterating over response_stream")

    except AttributeError as e:
        print(f"\\nAn error occurred during content generation (AttributeError): {e}")
        print("This could be due to an incorrect method name (e.g., 'generate_content_stream')")
        print("or the response chunk not having a '.text' attribute in your SDK version.")
        # Decide if this should yield an error or raise
        raise RuntimeError(f"Gemini content generation failed (AttributeError): {e}") from e
    except Exception as e:
        print(f"\\nAn error occurred during content generation: {e}")
        # Decide if this should yield an error or raise
        raise RuntimeError(f"Gemini content generation failed: {e}") from e


if __name__ == "__main__":
    # Example of how to run the async generator
    async def main_test():
        # This is a test block. In real use, audio_path and prompt would come from elsewhere.
        audio_path_test = "test.m4a" # Make sure this file exists for testing
        prompt_key_test = "gemini-transcription"
        
        if not os.path.exists(audio_path_test):
            print(f"Test audio file '{audio_path_test}' not found. Skipping test run.")
            return

        if prompt_key_test not in PROMPTS:
            print(f"Test prompt key '{prompt_key_test}' not found in prompts.py. Skipping test run.")
            return
            
        print(f"Testing gemini_client.py with {audio_path_test} and prompt '{prompt_key_test}'...")
        
        try:
            with open(audio_path_test, "rb") as f:
                test_audio_bytes = f.read()
            
            test_prompt_text = PROMPTS[prompt_key_test]

            print("Streaming response from Gemini:")
            async for transcript_chunk in generate_transcription_stream(test_audio_bytes, test_prompt_text):
                print(transcript_chunk, end="", flush=True)
            print("\\n--- End of test transcription ---")
        except ValueError as ve: # Catch specific error for API key
            print(ve)
        except RuntimeError as re:
            print(f"Test run failed: {re}")
        except Exception as e:
            print(f"An unexpected error occurred during test: {e}")

    asyncio.run(main_test()) 