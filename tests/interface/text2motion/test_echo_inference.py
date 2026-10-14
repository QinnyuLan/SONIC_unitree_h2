"""Inference input checks; run in the optional text2motion environment."""
import copy
import json
import wave

import pytest

torch = pytest.importorskip('torch')

from interface.text2motion.echo_features import (
    TIME_MAPPING, load_manifest, provenance_hashes, token_time_spans, validate_conditions,
)


def complete_condition():
    payload = dict(stem='speech', fps=30., duration=.1, source_audio_sha256='a' * 64,
                   audio_features=torch.zeros(3, 1024), text_tokens=torch.zeros(2, 2560),
                   token_times=torch.tensor([[0., .04], [.04, .08]], dtype=torch.float16),
                   n_tokens=2, has_word_timing=True, source_text_model='Qwen3.5-4B',
                   hidden_layer=-2, encoder_dtype='bfloat16', canonical_transcript='Hi all',
                   word_timestamps=[dict(text='Hi', start=0., end=.04),
                                    dict(text='all', start=.04, end=.08)])
    payload['text_provenance'] = dict(
        schema='echo-g-complete-text-provenance-v1', full_token_count=2, tokenizer_truncation=False,
        token_ids=[1, 2], token_offsets=[[0, 2], [2, 6]], token_time_mapping=TIME_MAPPING,
        encoder_identity=dict(model='Qwen3.5-4B'), **provenance_hashes(payload))
    return payload


def test_word_timing_preserves_spaces_and_tokens_spanning_words():
    words = [dict(text='大家', start=.1, end=.4), dict(text='好', start=.5, end=.7)]
    times = token_time_spans('大家 好', [[0, 2], [2, 3], [1, 4], [0, 0]], words)
    torch.testing.assert_close(times, torch.tensor([[.1, .4], [.4, .4], [.1, .7], [0., 0.]]))


def test_complete_cached_condition_is_accepted():
    validate_conditions(complete_condition())


@pytest.mark.parametrize('case', ['truncated', 'nonfinite', 'fps', 'duration', 'audio_hash',
                                 'text_hash', 'time_alignment', 'missing_provenance'])
def test_incomplete_or_misaligned_features_are_rejected(case):
    condition = copy.deepcopy(complete_condition())
    if case == 'truncated':
        condition['text_tokens'] = condition['text_tokens'][:1]
    elif case == 'nonfinite':
        condition['audio_features'][0, 0] = float('nan')
    elif case == 'fps':
        condition['fps'] = 50
    elif case == 'duration':
        condition['duration'] = 20.1
    elif case == 'audio_hash':
        condition['source_audio_sha256'] = 'not-a-checksum'
    elif case == 'text_hash':
        condition['text_tokens'][0, 0] = 1
    elif case == 'time_alignment':
        condition['token_times'][0, 1] = .02
        condition['text_provenance'].update(provenance_hashes(condition))
    else:
        condition.pop('text_provenance')
    with pytest.raises(ValueError):
        validate_conditions(condition)


def test_manifest_uses_words_and_measured_wav_duration(tmp_path):
    audio = tmp_path / 'speech.wav'
    with wave.open(str(audio), 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        stream.writeframes(b'\0\0' * 1600)
    record = dict(stem='speech', audio_path='speech.wav', transcript='Hi, all!',
                  words=complete_condition()['word_timestamps'])
    manifest = tmp_path / 'utterance.jsonl'
    manifest.write_text(json.dumps(record))
    result = load_manifest(manifest)
    assert result['transcript'] == 'Hi all'
    assert result['duration'] == .1
    assert result['audio_path'] == audio
    record['words'][-1]['end'] = .5
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='timestamps'):
        load_manifest(manifest)
