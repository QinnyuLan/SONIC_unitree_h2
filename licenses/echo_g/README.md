# ECHO-G source provenance

The following inference modules are adapted from [NJU-SOARLAB/ECHO-G](https://github.com/NJU-SOARLAB/ECHO-G/tree/60b85fb932b124b1a2ddd0868f4917a73c383901):

| Local file | Upstream source | Changes |
|---|---|---|
| `src/interface/text2motion/echo_model.py` | `model.py`, `word_time_attention.py`, `flow.py`, `inference.py` | Retains the released audio+text network, EMA loading, normalization and eight-step sampler; removes training, other model variants and dataset machinery |
| `src/interface/text2motion/echo_features.py` | `conditions.py`, `condition_provenance.py`, `data.py` | Single-utterance WAV/text feature extraction and validation; standard-library PCM reading replaces librosa |
| `models/text2motion_echo_g/encoders.json` | `manifests/condition_encoders.json` | Retains encoder revisions and runtime file checksums |

The code retains the [PolyForm Noncommercial license](LICENSE) and [required notice](NOTICE) from SOAR-LAB, Nanjing University, and Mondo Robotics. These files are exceptions to the repository's Apache-2.0 source-code license. The optional upstream weights retain their [CC BY-NC 4.0 terms](LICENSE-DATA-WEIGHTS). Model files are downloaded separately.

The complete upstream repository is not included. The adapted computation is checked against the pinned upstream implementation; see [reproducibility](../../docs/REPRODUCIBILITY.md#offline-echo-g-speechgesture-validation).
