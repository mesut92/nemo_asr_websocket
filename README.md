# Real-Time Speech Recognition WebSocket with NeMo toolkit

This project is a real-time Automatic Speech Recognition (ASR) client using WebSockets to stream audio to a server for transcription. It captures audio from a microphone, sends it to a WebSocket server, and prints the received transcriptions.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.10**
- **Conda (optional)**
- **Docker image (optional, recommended for environment management)**
- **NeMo (NVIDIA's conversational AI toolkit)**

### Preview

[![Watch the video](http://img.youtube.com/vi/1oNuXB4Yptg/0.jpg)](https://youtu.be/1oNuXB4Yptg)

## Installation

### Step 1: Set Up a Conda Environment (Recommended)

To avoid dependency conflicts, create a new conda environment:

```bash
conda create --name rt_asr python=3.10 -y
conda activate rt_asr
```

### Step 2: Install NeMo

NeMo is required for speech recognition processing. You can install it using the following command:

```bash
pip install nemo_toolkit[all]
```

If you encounter issues, follow the official NeMo installation guide: [NVIDIA NeMo GitHub](https://github.com/NVIDIA/NeMo).

Run the following command to install NeMo:

```bash
pip install git+https://github.com/NVIDIA/NeMo.git
```

### Step 3: Install Project Dependencies

- PyAudio may require additional system dependencies; install them using:
  ```bash
  sudo apt-get install sox libsndfile1 ffmpeg portaudio19-dev # (For Linux)
  ```
  
After activating the environment, install the required Python packages:

```bash
pip install -r requirements.txt
```

Or you can use docker container for asr server.

## Docker Setup

To build and run the Docker container for this project, follow these steps:

### Build the Docker Image

```bash
docker build . -t ws_asr
```

### Run the Docker Container
```bash
docker run --gpus all -p 8766:8766 --rm ws_asr
```

## Usage

### 1. Start the WebSocket Server

Ensure your WebSocket ASR server is running at `ws://localhost:8766` before starting the client.

```bash
python server.py
```

### 2. Run the Client

```bash
python client.py
```

### 3. Select the Audio Input Device

After running the client, it will list available audio input devices. Choose the appropriate device by entering its corresponding ID.

### 4. Start Speaking

Once the connection is established, the system will capture audio and send it to the WebSocket server for transcription in real-time.

## Cache-Aware Streaming FastConformer

This project utilizes NeMo models trained for streaming applications, as described in the paper: [*Noroozi et al.* "Stateful FastConformer with Cache-based Inference for Streaming Automatic Speech Recognition"](https://arxiv.org/abs/2312.17279) (accepted to ICASSP 2024).

### Model Features

- Trained with limited left and right-side context to enable low-latency streaming transcription.
- Implements **caching** to avoid recomputation of previous activations, reducing latency further.

### Available Model Checkpoints

1) [`stt_en_fastconformer_hybrid_large_streaming_80ms`](https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_80ms) - 80ms lookahead / 160ms chunk size
2) [`stt_en_fastconformer_hybrid_large_streaming_480ms`](https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_480ms) - 480ms lookahead / 540ms chunk size
3) [`stt_en_fastconformer_hybrid_large_streaming_1040ms`](https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_1040ms) - 1040ms lookahead / 1120ms chunk size
4) [`stt_en_fastconformer_hybrid_large_streaming_multi`](https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_multi) - 0ms, 80ms, 480ms, 1040ms lookahead / 80ms, 160ms, 540ms, 1120ms chunk size

### Model Inference Process

- Audio is continuously recorded in chunks and fed into the ASR model.
- Using `pyaudio`, an audio input stream passes the audio to a `stream_callback` function at set intervals.
- The `transcribe` function processes the audio chunk and returns transcriptions in real-time.
- **Chunk size** determines the duration of audio processed per step.
- **Lookahead size** is calculated as `chunk size - 80 ms` (since FastConformer models have a fixed 80ms output timestep duration).

## Configuration

- **WebSocket Server URL:** You can change the WebSocket server address in `client.py` by modifying the `WS_URL` variable.
- **Audio Parameters:** Adjust `SAMPLE_RATE`, `chunk_size`, and `ENCODER_STEP_LENGTH` in `client.py` to fine-tune the audio streaming behavior.

## Troubleshooting

- If NeMo installation fails, refer to the [NeMo Installation Guide](https://github.com/NVIDIA/NeMo) for specific dependencies and troubleshooting steps.
- Ensure your WebSocket server is running and accessible at the correct URL.
- If no audio input devices are found, check your microphone settings and ensure that `pyaudio` is correctly installed.

## License

This project is open-source. Feel free to modify and improve it!

---

### Notes
- The WebSocket server handling ASR is expected to be running separately.


