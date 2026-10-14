# Reference Motions

The 36 reference clips and six generated speech/gesture bundles are inputs to the independent Python motion replay process. The policy runtime creates its startup standing reference in memory and does not load motion files.

| Directory | Input type | Clips | Encoder selector |
|---|---|---:|---:|
| `example_g1_replay/` | Native G1 basic demos | 13 | 3 |
| `example_g1_explore_replay/` | Native G1 exploration demos | 7 | 3 |
| `example_h2_replay/` | Native H2 basic demos | 8 | 0 |
| `example_smpl_replay/` | SMPL demos | 8 | 2 |
| `example_text2motion_replay/` | ECHO-G Chinese/English speech and G1 gestures | 6 | 3 |

The exploration set includes kneeling, crawling and jumping. Clips with the same name in different collections contain different reference representations. Each reference collection's `manifest.json` records input type, frame rate, frame count, source collection and file SHA256 hashes. Motion data was copied from the corresponding `SONIC-H2/gear_sonic_deploy/reference/` collections; original clip metadata is retained as provenance. The extreme H2 collection is not included.

## Replay

From the repository root, pass one clip directory:

```bash
# Inspect without connecting to the runtime.
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_g1_replay/walking_quip_360_R_002__A428 --inspect

# Send a native H2 trajectory to the running policy runtime.
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_h2_replay/walking_quip_360_R_002__A428

# Send the corresponding SMPL reference.
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_smpl_replay/walking_quip_360_R_002__A428
```

Replay detects the type from `manifest.json` and clip metadata. For external clips without declarations, it uses distinctive file names. Mixed H2/SMPL files without a declaration require `--type h2` or `--type smpl`; an explicit type must agree with existing metadata. The original SONIC manifest name is also supported when reading external clips.

`--fps` overrides the source frame rate. CSV falls back to 50 Hz when metadata does not specify it. `--speed` changes playback speed, including reference velocity. `--loop` requires a continuous loop boundary. On completion or `Ctrl+C`, replay releases its body channel and the runtime returns to standing.

## CSV inputs

| Type | Files read | Interpretation |
|---|---|---|
| G1 | `joint_pos.csv`, optional `joint_vel.csv`, `body_quat.csv` | Native G1 semantic joints, 29 columns; first body quaternion is the root |
| H2 | `h2_joint_pos.csv`, optional `h2_joint_vel.csv`, `body_quat.csv` | Native H2 MuJoCo joints, 31 columns; first body quaternion is the root |
| SMPL | `smpl_joint.csv`, `smpl_root_quat.csv`, `joint_pos.csv` | 24 XYZ vectors (72 columns), SMPL root quaternion (4), semantic wrist angles from columns 23–28 of the 29D joint file |

All files for a clip must have matching frame counts. CSVs may include a column-name header. Missing G1/H2 velocity is computed by finite differences. Angles are radians, velocities rad/s, positions meters and quaternions **wxyz**.

H2's 31D source order is left leg (6), right leg (6), waist yaw/roll/pitch (3), head pitch/yaw (2), left arm (7), right arm (7). Leg order is hip pitch/roll/yaw, knee, ankle roll/pitch. Replay selects and reorders the 29 body joints into native H2 policy order without a G1 projection. Source head columns are excluded; head commands remain on the independent head channel. The H2 collection's additional 29D `joint_pos.csv` is a derived representation and is not used as a G1 trajectory.

SMPL uses its own root quaternion, not the robot quaternion in `body_quat.csv`. Joint vectors are root-centered in world axes, as in the source exports; the runtime rotates them into the SMPL root frame. Wrist order is left roll, right roll, left pitch, right pitch, left yaw, right yaw. `smpl_pose.csv` is retained as source data but is not needed by replay.

G1 and H2 send ten samples spaced **0.1 s** apart. SMPL sends ten samples spaced **0.02 s** apart, matching the model's SMPL observation contract. Producers send updates at 50 Hz. Terminal G1/H2 velocities are zero.

## NPZ inputs and offline text-to-motion

NPZ files use the following arrays, plus scalar `fps`. An optional scalar string `reference_type` declares `g1`, `h2` or `smpl`.

| Type | Required arrays | Optional arrays |
|---|---|---|
| G1 | `joint_pos[N,29]`, `root_quat[N,4]` | `joint_vel[N,29]` |
| H2 | `h2_joint_pos[N,31]`, `root_quat[N,4]` | `h2_joint_vel[N,31]` |
| SMPL | `joints[N,24,3]`, `root_quat[N,4]`, `wrist_joints[N,6]` | — |

These use the same coordinate and joint conventions as CSV. For offline text-to-motion, export the generated motion to one of these formats. The optional [ECHO-G integration](../docs/TEXT2MOTION.md) generates G1 motion with matching speech.

```bash
.venv_inputs/bin/python -m interface.motion_replay generated_motion.npz --inspect
.venv_inputs/bin/python -m interface.motion_replay generated_motion.npz
```

## Speech and motion bundles

Pass a directory under `motions/example_text2motion_replay/` to the same replay entry point:

| Bundle | Content | Duration |
|---|---|---:|
| `h2_trace_intro_zh` | Chinese H2/TRACE introduction, standing | 73.04 s |
| `h2_trace_intro_en` | English H2/TRACE introduction, standing | 78.48 s |
| `h2_trace_intro_full_body_zh` | Chinese introduction, generated full-body comparison | 73.04 s |
| `h2_trace_intro_full_body_en` | English introduction, generated full-body comparison | 78.48 s |
| `echo_g_zh` | Short Chinese lab greeting | 3.26 s |
| `echo_g_en` | Short English lab greeting | 3.53 s |

Standing and full-body pairs use identical speech and timing. The standing versions preserve generated waist/arm gestures and use a fixed G1 leg/root reference. The full-body comparisons retain generated leg/root motion; compare their behavior in simulation. Neither mode locks the H2 to a world position.

Each contains `motion.npz`, `speech.wav` and a `replay.json` manifest with relative paths and SHA256 hashes. The complete introductions also contain `script.txt` and paragraph subtitles in `subtitles.srt`. Their manifests record the seven source utterances and the silent motion transitions between them. All files needed for playback are included; ECHO-G models are not loaded during replay.

Replay checks both files before sending, plays audio locally by default, and supports an independent Unitree speaker adapter. Use `--inspect` to validate without playback or `--mute` for motion only. Timing overrides and looping require muted playback. See [generation and audio setup](../docs/TEXT2MOTION.md).

See [deployment](../docs/DEPLOYMENT.md#motion-replay-and-offline-text-to-motion) and [the command protocol](../docs/ARCHITECTURE.md). Loading, sampling and observation checks do not establish closed-loop tracking performance; validation scope is recorded in [Reproducibility](../docs/REPRODUCIBILITY.md).
