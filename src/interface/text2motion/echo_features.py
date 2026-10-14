# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).
# Adapted for SONIC_unitree_h2: single-utterance inference; see licenses/echo_g/.
"""Word-aligned Qwen/wav2vec2 features for the released audio+text model."""
import hashlib
import json
import math
from pathlib import Path
import re
import wave

import numpy as np
import torch
from torch.nn import functional as F

from interface.playback import sha256, wav_duration

TIME_MAPPING = 'character_offset_union_with_spaces_at_previous_word_end'


def load_manifest(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(records) != 1:
        raise ValueError('Provide exactly one utterance per bundle')
    record = records[0]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', record.get('stem', '')):
        raise ValueError('Unsafe or empty utterance stem')
    words, previous_start = record.get('words'), -1.
    if not isinstance(words, list) or not words:
        raise ValueError('ECHO-G requires word timestamps')
    for word in words:
        text = word.get('text', word.get('word', ''))
        start, end = float(word['start']), float(word['end'])
        if (not isinstance(text, str) or not text or not math.isfinite(start)
                or not math.isfinite(end) or start < max(0., previous_start) or end < start):
            raise ValueError('Invalid word timestamps')
        word.update(text=text, start=start, end=end)
        previous_start = start
    if not record.get('audio_path'):
        raise ValueError('Manifest must name the WAV used for feature extraction')
    audio = Path(record['audio_path'])
    audio = audio if audio.is_absolute() else (Path(path).parent / audio).resolve()
    duration = wav_duration(audio)
    if duration > 20 or max(word['end'] for word in words) > duration + .05:
        raise ValueError('Speech must fit 20 seconds, and word timestamps must fit its duration')
    return dict(stem=record['stem'], audio_path=audio, transcript=' '.join(w['text'] for w in words),
                words=words, duration=duration)


def token_time_spans(transcript, offsets, words):
    if not words or transcript != ' '.join(word['text'] for word in words):
        raise ValueError('Token offsets require the canonical space-joined word transcript')
    character_times = []
    for index, word in enumerate(words):
        start, end = float(word['start']), float(word['end'])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError('Invalid word timestamp')
        character_times.extend((start, end) for _ in word['text'])
        if index != len(words) - 1:
            character_times.append((end, end))
    spans = []
    for low, high in offsets:
        if not 0 <= low <= high <= len(transcript):
            raise ValueError('Invalid tokenizer character offsets')
        covered = character_times[low:high]
        spans.append([min(v[0] for v in covered), max(v[1] for v in covered)] if covered else [0., 0.])
    return torch.tensor(spans, dtype=torch.float32)


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def provenance_hashes(payload):
    return dict(canonical_transcript_sha256=hashlib.sha256(payload['canonical_transcript'].encode()).hexdigest(),
                text_tokens_sha256=tensor_sha256(payload['text_tokens']),
                token_times_sha256=tensor_sha256(payload['token_times']))


def validate_conditions(payload):
    """Accept complete locally extracted or upstream cached conditions; reject truncation."""
    audio, text, times = (payload[key] for key in ('audio_features', 'text_tokens', 'token_times'))
    duration = float(payload['duration'])
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', payload.get('stem', ''))
            or not math.isfinite(duration) or not 0 < duration <= 20 or payload.get('fps') != 30
            or not re.fullmatch(r'[0-9a-f]{64}', payload.get('source_audio_sha256', ''))):
        raise ValueError('Invalid condition stem, duration, frame rate or audio checksum')
    frames, count = max(2, round(duration * 30)), payload.get('n_tokens')
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError('Expected 1–256 complete text tokens')
    for values, shape in ((audio, (frames, 1024)), (text, (count, 2560)), (times, (count, 2))):
        if not isinstance(values, torch.Tensor) or values.shape != shape or not torch.isfinite(values).all():
            raise ValueError(f'Expected finite condition tensor {shape}')
    if (not payload.get('has_word_timing') or (times < 0).any() or (times[:, 1] < times[:, 0]).any()
            or float(times.max()) > duration + .05):
        raise ValueError('Invalid or missing token timestamps')
    provenance = payload.get('text_provenance', {})
    ids, offsets = provenance.get('token_ids', []), provenance.get('token_offsets', [])
    if (provenance.get('schema') != 'echo-g-complete-text-provenance-v1'
            or provenance.get('full_token_count') != count or provenance.get('tokenizer_truncation') is not False
            or len(ids) != count or any(type(value) is not int or value < 0 for value in ids)
            or len(offsets) != count or any(len(pair) != 2 or any(type(v) is not int for v in pair) for pair in offsets)
            or payload.get('hidden_layer') != -2 or payload.get('encoder_dtype') != 'bfloat16'
            or provenance.get('token_time_mapping') != TIME_MAPPING
            or not payload.get('source_text_model')
            or provenance.get('encoder_identity', {}).get('model') != payload['source_text_model']):
        raise ValueError('Missing or truncated text/encoder provenance')
    expected_times = token_time_spans(payload['canonical_transcript'], offsets, payload['word_timestamps'])
    if not torch.equal(times.cpu(), expected_times.to(times.dtype)):
        raise ValueError('Token timestamps differ from the recorded word mapping')
    if any(provenance.get(key) != value for key, value in provenance_hashes(payload).items()):
        raise ValueError('Text condition checksum mismatch')


@torch.no_grad()
def audio_features(utterance, model_path, device):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    # The accepted format is already 16 kHz PCM16 mono; no decoder/resampler dependency.
    with wave.open(str(utterance['audio_path']), 'rb') as stream:
        waveform = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float32) / 32768.
    processor = Wav2Vec2Processor.from_pretrained(model_path, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(model_path, local_files_only=True).to(device).eval()
    values = processor(waveform, sampling_rate=16000, return_tensors='pt').input_values.to(device)
    hidden = model(values, output_hidden_states=True).hidden_states[-1].float()
    frames = max(2, round(utterance['duration'] * 30))
    return F.interpolate(hidden.transpose(1, 2), size=frames, mode='linear', align_corners=True)[0].T.cpu().half()


@torch.no_grad()
def extract_features(utterance, models, device):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.utils import logging
    logging.disable_progress_bar()
    audio_path = str((models / 'wav2vec2-large-xlsr-53-english').resolve())
    text_path = str((models / 'Qwen3.5-4B').resolve())
    tokenizer = AutoTokenizer.from_pretrained(text_path, use_fast=True, local_files_only=True)
    encoded = tokenizer(utterance['transcript'], return_tensors='pt', return_offsets_mapping=True, truncation=False)
    count = encoded['input_ids'].shape[1]
    if not 1 <= count <= 256:
        raise ValueError('Text exceeds 256 tokens; split it into shorter utterances')
    offsets = encoded.pop('offset_mapping')[0].tolist()
    times = token_time_spans(utterance['transcript'], offsets, utterance['words'])
    audio = audio_features(utterance, audio_path, device)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        text_path, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    input_ids = encoded['input_ids'].to(device)
    hidden = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                   output_hidden_states=True).hidden_states[-2][0].float()
    payload = dict(stem=utterance['stem'], fps=30., duration=utterance['duration'],
                   audio_features=audio, source_audio_sha256=sha256(utterance['audio_path']),
                   source_audio_model=audio_path, audio_dim=1024,
                   canonical_transcript=utterance['transcript'], word_timestamps=utterance['words'],
                   text_tokens=hidden.cpu().contiguous().half(), token_times=times.half(),
                   text_pooled=hidden.mean(dim=0).cpu().half(), n_tokens=count, text_dim=2560,
                   source_text_model=text_path, hidden_layer=-2, encoder_dtype='bfloat16',
                   has_word_timing=True, token_time_mapping=TIME_MAPPING)
    payload['text_provenance'] = dict(
        schema='echo-g-complete-text-provenance-v1', full_token_count=count, tokenizer_truncation=False,
        token_ids=encoded['input_ids'][0].tolist(), token_offsets=offsets, token_time_mapping=TIME_MAPPING,
        encoder_identity=dict(model=text_path, model_class=type(model).__name__,
                              tokenizer_class=type(tokenizer).__name__,
                              resolved_revision=getattr(model.config, '_commit_hash', None)),
        **provenance_hashes(payload))
    validate_conditions(payload)
    return payload
