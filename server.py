import asyncio
import websockets
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
import json
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
import copy
from omegaconf import open_dict


def init_model():
    lookahead_size = 80  # in milliseconds
    decoder_type = "rnnt"
    device = torch.device('cuda:0')
    ENCODER_STEP_LENGTH = 80  # ms

    model_name = "stt_en_fastconformer_hybrid_large_streaming_multi"
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    if model_name == "stt_en_fastconformer_hybrid_large_streaming_multi":
        if lookahead_size not in [0, 80, 480, 1040]:
            raise ValueError(
                f"Invalid lookahead_size {lookahead_size}. Allowed values: 0, 80, 480, 1040 ms"
            )
        left_context_size = asr_model.encoder.att_context_size[0]
        asr_model.encoder.set_default_att_context_size([left_context_size, int(lookahead_size / ENCODER_STEP_LENGTH)])

    asr_model.change_decoding_strategy(decoder_type=decoder_type)
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = False
        if hasattr(asr_model, 'joint'):
            decoding_cfg.greedy.max_symbols = 10
            decoding_cfg.fused_batch_size = -1
        asr_model.change_decoding_strategy(decoding_cfg)

    asr_model.eval()

    # get parameters to use as the initial cache state
    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=1
    )
    pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
    num_channels = asr_model.cfg.preprocessor.features
    preprocessor = init_preprocessor(asr_model, device)
    cache = {
        'device': device,
        'cache_last_channel': cache_last_channel,
        'cache_last_time': cache_last_time,
        'cache_last_channel_len': cache_last_channel_len,
        'previous_hypotheses': None,
        'pred_out_stream': None,
        'step_num': 0,
        'pre_encode_cache_size': asr_model.encoder.streaming_cfg.pre_encode_cache_size[1],
        'num_channels': asr_model.cfg.preprocessor.features,
        'cache_pre_encode': torch.zeros((1, num_channels, pre_encode_cache_size), device=device)

    }
    return asr_model, device, preprocessor, cache

def init_preprocessor(asr_model, device):
    cfg = copy.deepcopy(asr_model._cfg)
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0
    cfg.preprocessor.normalize = "None"
    preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
    preprocessor.to(device)
    return preprocessor

init_model()

def extract_transcriptions(hyps):
    return [hyp.text if isinstance(hyp, Hypothesis) else hyp for hyp in hyps]


def preprocess_audio(audio, preprocessor, device):
    audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
    audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
    return preprocessor(input_signal=audio_signal, length=audio_signal_len)


def transcribe_chunk(new_chunk, asr_model, preprocessor, cache):
    audio_data = new_chunk.astype(np.float32) / 32768.0
    processed_signal, processed_signal_length = preprocess_audio(audio_data, preprocessor, cache['device'])
    processed_signal = torch.cat([cache['cache_pre_encode'], processed_signal], dim=-1)
    processed_signal_length += cache['cache_pre_encode'].shape[1]
    cache['cache_pre_encode'] = processed_signal[:, :, -cache['pre_encode_cache_size']:]

    with torch.no_grad():
        (
            cache['pred_out_stream'],
            transcribed_texts,
            cache['cache_last_channel'],
            cache['cache_last_time'],
            cache['cache_last_channel_len'],
            cache['previous_hypotheses'],
        ) = asr_model.conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache['cache_last_channel'],
            cache_last_time=cache['cache_last_time'],
            cache_last_channel_len=cache['cache_last_channel_len'],
            keep_all_outputs=False,
            previous_hypotheses=cache['previous_hypotheses'],
            previous_pred_out=cache['pred_out_stream'],
            drop_extra_pre_encoded=None,
            return_transcription=True,
        )
    return extract_transcriptions(transcribed_texts)[0]


async def asr_handler(websocket):
    print("Client connected...")
    asr_model, device, preprocessor, cache = init_model()
    try:
        async for message in websocket:
            if not message:
                print("No audio data received.")
                continue
            signal = np.frombuffer(message, dtype=np.int16)
            transcription = transcribe_chunk(signal, asr_model, preprocessor, cache)
            await websocket.send(json.dumps({"transcription": transcription}))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


async def main():

    async with websockets.serve(lambda ws: asr_handler(ws), "0.0.0.0", 8766):
        print("WebSocket Server started on ws://0.0.0.0:8766")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())