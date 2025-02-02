import asyncio
import websockets
import pyaudio as pa
import time
import threading
import json

# WebSocket URL
WS_URL = "ws://localhost:8766"

# Constants
SAMPLE_RATE = 16000
ENCODER_STEP_LENGTH = 80  # Example value
lookahead_size = 80  # Example value
chunk_size = lookahead_size + ENCODER_STEP_LENGTH

# Create an asyncio queue for sending audio data
audio_queue = asyncio.Queue()


async def send_audio(websocket):
    """Handles sending audio data over WebSocket."""
    while True:
        audio_chunk = await audio_queue.get()
        await websocket.send(audio_chunk)


async def receive_transcription(websocket):
    """Handles receiving ASR transcription results from the server."""
    while True:
        try:
            # Receive and decode the transcription message
            response = await websocket.recv()
            transcription_data = json.loads(response)
            transcription = transcription_data.get("transcription", "")

            # Print the transcription
            print(f"Transcription: {transcription}")
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
            break


def audio_callback(in_data, frame_count, time_info, status):
    if loop.is_running():
        asyncio.run_coroutine_threadsafe(audio_queue.put(in_data), loop)
    else:
        print("Event loop is not running!")
    return (in_data, pa.paContinue)


def start_audio_stream():
    """Handles PyAudio streaming."""
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    p = pa.PyAudio()
    input_devices = [i for i in range(p.get_device_count()) if p.get_device_info_by_index(i).get('maxInputChannels')]

    if not input_devices:
        print("ERROR: No audio input device found.")
        return

    print('Available audio input devices:')
    for i in input_devices:
        print(i, p.get_device_info_by_index(i).get('name'))

    dev_idx = int(input("Please type input device ID: "))

    stream = p.open(format=pa.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=dev_idx,
                    stream_callback=audio_callback,
                    frames_per_buffer=int(SAMPLE_RATE * chunk_size / 1000)
                    )

    print('Listening...')
    stream.start_stream()

    try:
        while stream.is_active():
            time.sleep(0.1)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("PyAudio stopped")


async def main():
    """Handles the WebSocket communication for both sending audio and receiving transcriptions."""
    while True:
        try:
            async with websockets.connect(WS_URL) as websocket:
                # Start sending audio data
                send_task = asyncio.create_task(send_audio(websocket))

                # Start receiving transcriptions from the server
                receive_task = asyncio.create_task(receive_transcription(websocket))

                # Run both tasks concurrently
                await asyncio.gather(send_task, receive_task)
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed. Reconnecting...")
            await asyncio.sleep(1)  # Wait before reconnecting

# Start the event loop for WebSocket communication
loop = asyncio.get_event_loop()
time.sleep(0.1)
threading.Thread(target=start_audio_stream, daemon=True).start()
time.sleep(0.1)
loop.run_until_complete(main())

