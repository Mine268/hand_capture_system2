# Repository Guidelines

## Project Structure & Module Organization
`src/` contains implementation code split by pipeline stage: `capture/`, `perception/`, `calibration/`, `fusion/`, `render/`, `visualization/`, `pipeline/`, and `io/`. Public headers live under `include/newnewhand/...` and should mirror the module layout in `src/`. Demo entry points live in `apps/`. Design notes and runtime walkthroughs are in `docs/`. Helper export scripts are in `tools/`. Runtime assets live in `resources/`; generated outputs such as `build/`, `captures/`, `results/`, `debug/`, and `offline_dump/` are ignored.

## Build, Test, and Development Commands
Configure from the repository root:

```bash
cmake -S . -B build -DBUILD_DEMOS=ON -DMVS_ROOT=/opt/MVS -DONNX_RUNTIME_DIR=/path/to/onnxruntime
cmake --build build -j
```

Use demo binaries for development checks:

```bash
./build/stereo_capture_demo --frames 1 --save --no_preview --output_dir captures
./build/single_view_hand_pose_demo --image captures/cam0/000001.bmp --gpu
./build/stereo_fused_hand_pose_demo --calibration resources/stereo_calibration.yaml --gpu
```

## Coding Style & Naming Conventions
This repository uses C++17. Follow the existing style: 4-space indentation, braces on the same line, and short, explicit comments only where needed. Keep filenames snake_case such as `stereo_hand_fuser.cpp`; use PascalCase for types such as `StereoCaptureConfig`; keep code inside the `newnewhand` namespace. Public APIs belong in `include/newnewhand/...`; internal helpers stay in `src/`.

## Testing Guidelines
There is no dedicated `tests/` target in CMake yet. Validate changes by building successfully and running the relevant demo app for the module you touched. For perception, capture, calibration, and rendering changes, record the exact command used and note required hardware, models, or calibration files in your PR. If you add automated tests later, wire them into CMake/CTest and keep test data outside ignored output folders.

## Commit & Pull Request Guidelines
Recent history follows short Conventional Commit prefixes: `feat:`, `fix:`, `docs:`, `ignore:`. Keep commit subjects imperative and scoped to one change. PRs should state the affected module, list the commands run, mention any SDK or model-path assumptions, and include screenshots or output paths for UI, overlay, calibration, or 3D viewer changes.

## Configuration & Assets
Do not commit generated captures, results, or offline dumps. `resources/models/` is gitignored, so document any required local model files and paths when your change depends on them.
