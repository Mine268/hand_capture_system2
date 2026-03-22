# Third-Party Notices

This file describes the intended license boundary for this repository and
summarizes third-party components used by the project.

This file is informational only and is not legal advice.

## Scope of the top-level license

The top-level [LICENSE](/home/renkaiwen/src/newnewhand/LICENSE) applies only to
original source code and documentation in this repository, except where another
license or notice applies.

It does not override:

- any license in `third/`
- any vendor SDK or runtime installed outside this repository
- any model file or dataset obtained separately from this repository

## Third-party source code included in this repository

### `third/stella_vslam`

- Upstream/original fork base license: BSD 2-Clause
- Local fork changes license: BSD 2-Clause
- Files:
  - [third/stella_vslam/LICENSE](/home/renkaiwen/src/newnewhand/third/stella_vslam/LICENSE)
  - [third/stella_vslam/LICENSE.original](/home/renkaiwen/src/newnewhand/third/stella_vslam/LICENSE.original)
  - [third/stella_vslam/LICENSE.fork](/home/renkaiwen/src/newnewhand/third/stella_vslam/LICENSE.fork)

### `third/yaml-cpp`

- License: MIT
- File:
  - [third/yaml-cpp/LICENSE](/home/renkaiwen/src/newnewhand/third/yaml-cpp/LICENSE)

### `third/g2o`

`g2o` is not uniformly licensed under a single permissive license.

From the vendored `g2o` README:

- project core is described as BSD
- `csparse_extension` is LGPL v2.1+
- some viewer/example components are GPL3+
- some SuiteSparse / CHOLMOD configurations may introduce additional GPL-sensitive constraints

Relevant files:

- [third/g2o/README.md](/home/renkaiwen/src/newnewhand/third/g2o/README.md)
- [third/g2o/doc/license-bsd.txt](/home/renkaiwen/src/newnewhand/third/g2o/doc/license-bsd.txt)
- [third/g2o/doc/license-lgpl.txt](/home/renkaiwen/src/newnewhand/third/g2o/doc/license-lgpl.txt)
- [third/g2o/doc/license-gpl.txt](/home/renkaiwen/src/newnewhand/third/g2o/doc/license-gpl.txt)

Important build note:

- this repository links `g2o::solver_csparse` and `g2o::csparse_extension` in the
  `newnewhand_slam_optimization` target when `NEWHAND_ENABLE_STELLA_VSLAM=ON`
  is enabled, see [CMakeLists.txt](/home/renkaiwen/src/newnewhand/CMakeLists.txt)
- if you distribute binaries from the `build_stella/` configuration, you should
  separately review LGPL / GPL implications of the exact `g2o` and SuiteSparse
  components present on the build machine

## External dependencies not shipped in this repository

### Hikrobot / Hikvision MVS SDK

This project expects an MVS SDK installation outside the repository, typically
under `/opt/MVS`, and links against `MvCameraControl`.

The repository license does not apply to the MVS SDK.

Relevant local notice file on the current machine:

- [/opt/MVS/license/CLIENT_MVS_Linux_license_notice.txt](/opt/MVS/license/CLIENT_MVS_Linux_license_notice.txt)

### OpenCV

- External system dependency
- OpenCV 4.x is generally distributed under Apache-2.0
- Check the exact OpenCV package you install on your target system

### ONNX Runtime

- External system dependency
- ONNX Runtime is generally distributed under MIT
- Check the exact package and version you deploy

### GLFW / GLEW / OpenGL runtime packages

- External system dependencies
- Follow the license terms of the exact packages installed on your target system

## Model files

The project expects model files under `resources/models/`, but that directory is
gitignored and model files are not committed by default.

Relevant paths in the codebase include:

- `resources/models/detector.onnx`
- `resources/models/wilor_backbone_opset16.onnx`
- `resources/models/mano_cpu_opset16.onnx`

Model files may have licenses separate from the repository source code. The
top-level repository license does not grant rights to third-party model weights
obtained from elsewhere.

## Practical guidance

If you distribute source code for this repository:

- include the top-level [LICENSE](/home/renkaiwen/src/newnewhand/LICENSE)
- preserve licenses and notices inside `third/`
- do not imply that the repository license changes the terms of external SDKs or models

If you distribute binaries:

- include third-party notices for bundled components
- review whether your build includes `build_stella/` targets that link `g2o`
  components with LGPL / GPL implications
- review the redistribution terms of the MVS SDK and separately obtained model files
