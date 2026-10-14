# ECHO-G Offline Generation Models

Optional models for [speech and gesture generation](../../docs/TEXT2MOTION.md). Replay of existing bundles does not load them.

```bash
./install.sh text2motion
.venv_text2motion/bin/python -m interface.text2motion.echo_g download
.venv_text2motion/bin/python -m interface.text2motion.echo_g download --verify-only
```

The downloader installs the following ignored local files here:

| Files | Source |
|---|---|
| `best.pt` | [ECHO-G audio+text EMA checkpoint](https://huggingface.co/gaopusen/ECHO-G/tree/v0.2.0), SHA256 `c84f31fce140cdc8c3be5e5dbc8cb8eb3a82ce1865e2d84026a0791d4735928d` |
| `Qwen3.5-4B/` | Pinned text encoder, hidden layer -2 |
| `wav2vec2-large-xlsr-53-english/` | Pinned speech encoder, final hidden layer |

Encoder revisions, required files and SHA256 values are retained from the upstream release in [encoders.json](encoders.json). The complete set is approximately 11 GB; `--checkpoint-only` downloads only the roughly 625 MiB ECHO-G weight for cached-condition inference. Partial downloads can be resumed, and files replace their destination only after checksum validation.

No model files are committed to Git. ECHO-G is an optional PyTorch generator; the policy and planner use their separate TensorRT models. Source and weight terms remain those of the upstream project; see [licenses](../../licenses/README.md).
