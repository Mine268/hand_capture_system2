# newnewhand 运行文档

本文档面向当前仓库的主要使用流程，聚焦以下三条主线：

1. 双视图标定功能
2. 实时 demo
3. 离线 demo

文档内容基于当前源码实现整理，命令行参数、流程和输出目录均按现有程序行为描述。除正文三条主线外，文末还补充了其他相关程序的用途说明。

## 1. 环境准备

### 1.1 适用对象

- 已完成工程编译，能够运行 `./build/...` 或 `./build_stella/...` 下的可执行文件
- 已正确安装海康 MVS SDK，并设置 `MVS_ROOT`
- 已准备好 ONNX Runtime
- 已准备好检测器与手部模型文件

### 1.2 推荐编译命令

基础构建：

```bash
cmake -S . -B build \
  -DBUILD_DEMOS=ON \
  -DMVS_ROOT=/opt/MVS \
  -DONNX_RUNTIME_DIR=/path/to/onnxruntime

cmake --build build -j
```

如果需要离线 demo 的 `stella` 后端，单独构建：

```bash
cmake -S . -B build_stella \
  -DBUILD_DEMOS=ON \
  -DMVS_ROOT=/opt/MVS \
  -DONNX_RUNTIME_DIR=/path/to/onnxruntime \
  -DNEWHAND_ENABLE_STELLA_VSLAM=ON

cmake --build build_stella -j
```

说明：

- `build/` 用于常规实时功能和不依赖 `stella_vslam` 的程序
- `build_stella/` 用于 `stereo_fused_hand_pose_offline_demo --slam_backend stella` 和 `stereo_fused_hand_pose_replay_app`

### 1.3 模型与资源

默认模型路径如下：

- 检测器：`resources/models/detector.onnx`
- WiLoR backbone：`resources/models/wilor_backbone_opset16.onnx`
- MANO：`resources/models/mano_cpu_opset16.onnx`

默认标定文件路径如下：

- `resources/stereo_calibration.yaml`

如果这些文件不在默认位置，需要在命令行中显式传入对应参数。

## 2. 双视图标定功能

### 2.1 功能目的

主入口程序：`stereo_guided_calibration_app`

该程序用于引导完成双目标定的完整闭环，包括：

1. 交互确认左右相机顺序
2. 采集左目单目标定图
3. 采集右目单目标定图
4. 采集双目外参图
5. 自动求解双目标定结果并保存 YAML

这也是当前推荐的双目标定主流程。

### 2.2 适用场景

- 已连接两台相机，但不确定物理左/右顺序
- 需要现场采集标定图并立即完成求解
- 需要把最终左右相机序列号一并写入标定文件，供后续实时/离线流程直接复用

### 2.3 输入与输出

输入：

- 当前连接的两台相机
- 运行时在终端交互输入的棋盘参数：
  - 内角点列数
  - 内角点行数
  - 方格边长（米）

输出：

- 标定结果文件：默认 `results/stereo_calibration.yaml`
- 采集图像目录：自动生成在输出 YAML 同目录下，目录名格式为  
  `标定文件名_capture_时间戳/`

该采集目录内部包含：

- `mono_left/`
- `mono_right/`
- `stereo_left/`
- `stereo_right/`

### 2.4 调用全参数

程序：

```bash
./build/stereo_guided_calibration_app [options]
```

参数说明：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--output <path>` | `results/stereo_calibration.yaml` | 最终标定结果输出路径 |
| `--exposure_us <f>` | `10000` | 采集曝光时间，单位微秒 |
| `--gain <f>` | `-1` | 增益；`-1` 表示自动增益 |
| `--fps <int>` | `3` | 引导采集时的抓帧频率 |
| `--frames <int>` | `30` | 每个阶段需要采集的有效图像数；左目、右目、双目阶段各自达到该数量才结束 |
| `--use_sb` | 启用 | 使用 `findChessboardCornersSB` |
| `--no_use_sb` | 关闭 `SB` | 改为经典 `findChessboardCorners + cornerSubPix` |

### 2.5 推荐流程

#### 步骤 1：启动程序

```bash
./build/stereo_guided_calibration_app \
  --output results/stereo_calibration.yaml \
  --exposure_us 10000 \
  --gain -1 \
  --fps 3 \
  --frames 30
```

#### 步骤 2：确认左右相机

程序会先枚举已连接相机，然后打开两个预览窗口：

- `select_cam0`
- `select_cam1`

终端会提示：

- 按 `1`：表示当前 `cam0` 是左相机，`cam1` 是右相机
- 按 `2`：表示当前 `cam1` 是左相机，`cam0` 是右相机
- 按 `q` 或 `Esc`：退出

这一阶段的目的是把最终的左右序列号固定下来，后续求解结果也会把这两个序列号写入标定 YAML。

#### 步骤 3：输入棋盘参数

程序会在终端依次询问：

1. `checkerboard inner corners cols`
2. `checkerboard inner corners rows`
3. `checkerboard square size in meters`

例如：

- 列数：`9`
- 行数：`6`
- 方格边长：`0.025`

注意这里输入的是内角点数量，不是格子数量。

#### 步骤 4：采集左目单目标定图

程序进入 `LEFT_MONO` 阶段。

行为：

- 只检查左目标定板检测结果
- 每检测到一张有效棋盘图，就保存一张原始图到 `mono_left/`
- 窗口中会显示当前保存进度和是否检测到棋盘

建议：

- 让棋盘覆盖不同位置、不同尺度、不同倾角
- 避免所有图都只在中心位置

#### 步骤 5：采集右目单目标定图

程序进入 `RIGHT_MONO` 阶段。

行为与左目阶段相同，只是目标变为右目，图像保存到 `mono_right/`。

#### 步骤 6：采集双目外参图

程序进入 `STEREO_EXTRINSICS` 阶段。

行为：

- 同时要求左右图像都能检测到棋盘
- 只有双目同时有效时才保存一对图
- 左右图分别保存到：
  - `stereo_left/`
  - `stereo_right/`

#### 步骤 7：自动求解并保存

全部采集完成后，程序会自动执行：

1. 左目单目标定
2. 右目单目标定
3. 双目外参标定
4. 写入 `stereo_calibration.yaml`

终端会输出：

- 左目 RMS
- 右目 RMS
- 双目 RMS
- 图像尺寸
- 有效图像数量
- 最终保存路径
- 左右序列号

### 2.6 结果目录结构

假设输出路径为：

```bash
--output results/stereo_calibration.yaml
```

则典型输出为：

```text
results/
├── stereo_calibration.yaml
└── stereo_calibration_capture_20260322_153000/
    ├── mono_left/
    ├── mono_right/
    ├── stereo_left/
    └── stereo_right/
```

### 2.7 常见问题

#### 只连接了一台或多于两台相机

该程序当前要求“恰好两台已连接相机”，否则会直接报错。

#### 左右顺序不确定

程序已内置相机角色确认阶段，按预览窗口内容确认即可。

#### 棋盘检测不稳定

建议优先保持 `--use_sb` 默认启用，同时：

- 提升照明
- 缩短曝光防止拖影
- 确保棋盘完整出现在视野内

#### 已经有左右图像目录，不想重新采集

可使用附加程序 `stereo_calibration_app` 做纯离线求解，见文末附加说明。

## 3. 实时 demo

### 3.1 功能目的

主入口程序：`stereo_fused_hand_pose_demo`

这是当前仓库的实时主 demo，完成以下实时链路：

1. 双目采集
2. 按标定进行去畸变
3. 单视图手部检测与手部姿态估计
4. 双目融合
5. ChArUco 板定位
6. 2D 预览与 3D OpenGL 可视化
7. 可选保存结果与离线导出

从线程结构看，它由以下模块并行组成：

- 采集线程
- 跟踪线程
- 姿态估计与融合线程
- 渲染线程

### 3.2 适用场景

- 在线双目手部姿态估计与融合演示
- 需要同时看 2D overlay 和 3D OpenGL 视图
- 需要实时保存叠加结果或额外导出离线包

### 3.3 输入与输出

输入：

- 双目标定文件
- 检测器模型
- WiLoR 模型
- MANO 模型
- 在线双目相机
- ChArUco 板参数

默认输出：

- `results/stereo_fused_hand_pose/`

当 `--save` 开启时，程序会保存：

- `cam0/*.png`
- `cam1/*.png`
- `yaml/*.yaml`

当 `--offline_dump_dir <dir>` 指定时，还会额外导出完整离线包，供后续离线分析使用。

### 3.4 调用全参数

程序：

```bash
./build/stereo_fused_hand_pose_demo [options]
```

参数说明：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--calibration <path>` | `resources/stereo_calibration.yaml` | 双目标定文件 |
| `--detector_model <path>` | `resources/models/detector.onnx` | 手检测模型 |
| `--wilor_model <path>` | `resources/models/wilor_backbone_opset16.onnx` | WiLoR backbone 模型 |
| `--mano_model <path>` | `resources/models/mano_cpu_opset16.onnx` | MANO 模型 |
| `--output_dir <dir>` | `results/stereo_fused_hand_pose` | 叠加图和 YAML 输出目录 |
| `--offline_dump_dir <dir>` | 关闭 | 导出完整离线包 |
| `--save` | 开启 | 保存实时结果 |
| `--no_save` | 关闭保存 | 不写磁盘 |
| `--preview` | 开启 | 打开 2D 预览窗口 |
| `--no_preview` | 关闭 | 不显示 2D 窗口 |
| `--glfw_view` | 开启 | 打开 OpenGL 3D 视图 |
| `--no_glfw_view` | 关闭 | 不显示 OpenGL 视图 |
| `--dictionary <name>` | `DICT_APRILTAG_36h11` | ChArUco / AprilTag 字典 |
| `--squares_x <int>` | `5` | 板面 X 向方格数 |
| `--squares_y <int>` | `7` | 板面 Y 向方格数 |
| `--square_length_m <f>` | `0.028` | 方格边长（米） |
| `--marker_length_m <f>` | `0.021` | marker 边长（米） |
| `--verbose` | 开启 | 打印详细日志 |
| `--quiet` | 关闭详细日志 | 精简日志输出 |
| `--gpu` | 开启 | ONNX Runtime 使用 GPU |
| `--cpu` | 关闭 GPU | ONNX Runtime 使用 CPU |

说明：

- 该实时 demo 中，板定位始终启用，没有单独的 `--slam/--no_slam` 开关
- 相机序列号来自标定文件；如果标定文件中保存了左右序列号，程序会强制按该顺序使用相机

### 3.5 推荐命令

最小推荐命令：

```bash
./build/stereo_fused_hand_pose_demo \
  --calibration resources/stereo_calibration.yaml \
  --gpu
```

完整推荐命令：

```bash
./build/stereo_fused_hand_pose_demo \
  --calibration resources/stereo_calibration.yaml \
  --detector_model resources/models/detector.onnx \
  --wilor_model resources/models/wilor_backbone_opset16.onnx \
  --mano_model resources/models/mano_cpu_opset16.onnx \
  --output_dir results/stereo_fused_hand_pose \
  --offline_dump_dir offline_dump/session_001 \
  --save \
  --preview \
  --glfw_view \
  --dictionary DICT_APRILTAG_36h11 \
  --squares_x 5 \
  --squares_y 7 \
  --square_length_m 0.028 \
  --marker_length_m 0.021 \
  --verbose \
  --gpu
```

### 3.6 完整流程

#### 步骤 1：加载标定与模型

程序启动后会先：

1. 读取标定文件
2. 校验 ChArUco 板参数是否合法
3. 组装双目采集配置
4. 配置检测模型和手部模型

#### 步骤 2：采集线程读取双目图像

采集线程会：

1. 启动双目采集
2. 根据标定对左右图做去畸变
3. 把原始帧与去畸变帧分别分发到后续线程

#### 步骤 3：跟踪线程做板定位

跟踪线程基于左目去畸变图执行：

1. ChArUco 检测
2. 位姿估计
3. 更新相机位姿和轨迹

该结果用于：

- 2D 窗口上叠加 tracking 信息
- OpenGL 中显示相机位姿和轨迹

#### 步骤 4：姿态线程做检测、估计与融合

姿态线程会：

1. 对双目图像分别做单视图手部估计
2. 生成每路 overlay
3. 融合为双目手部结果
4. 可选保存结果
5. 可选导出离线包

#### 步骤 5：渲染线程显示结果

渲染线程会：

- 持续刷新 OpenGL viewer
- 刷新 2D overlay 预览窗口
- 响应 `q` / `Esc` 退出

### 3.7 输出目录结构

当 `--save` 开启时，`--output_dir` 目录下典型结构如下：

```text
results/stereo_fused_hand_pose/
├── cam0/
├── cam1/
└── yaml/
```

其中：

- `cam0/*.png` 与 `cam1/*.png` 为叠加后的 2D 结果
- `yaml/*.yaml` 为融合结果 YAML

当设置 `--offline_dump_dir` 时，会额外生成完整离线包，目录结构见离线 demo 章节。

### 3.8 常见问题

#### 标定文件中的左右序列号与实际相机不一致

程序会在启动后直接报错，不会继续运行。

#### 模型文件缺失

请显式传入：

- `--detector_model`
- `--wilor_model`
- `--mano_model`

#### OpenGL 窗口关闭后程序退出

这是当前设计行为；关闭 3D 窗口会触发整体停止。

## 4. 离线 demo

离线主线分为四步：

1. 离线采集视频包
2. 视频包解码为图像序列
3. 离线估计并保存完整 replay 包
4. 从 replay 包重放

### 4.1 第一步：离线采集视频包

主入口程序：`stereo_offline_capture_app --mode capture`

#### 功能目的

把实时双目采集结果写成一个适合后处理的视频包。

#### 输入与输出

输入：

- 双目相机
- 可选标定文件

输出：

- `videos/cam0.avi`
- `videos/cam1.avi`
- `manifest.yaml`
- 可选 `calibration/stereo_calibration.yaml`

#### 调用全参数

程序：

```bash
./build/stereo_offline_capture_app [options]
```

参数说明：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--mode <capture|decode>` | `capture` | capture 为采集视频包；decode 为解码 |
| `--input_dir <dir>` | 无 | decode 模式必填 |
| `--output_dir <dir>` | `offline_capture/session_001` | 输出目录 |
| `--calibration <path>` | 关闭 | 可选标定文件；若其中有左右序列号，会强制按该顺序采集 |
| `--cam0_serial <serial>` | 自动选择 | 指定采集相机序列号 |
| `--cam1_serial <serial>` | 自动选择 | 指定采集相机序列号 |
| `--image_format <fmt>` | `png` | decode 时输出图像格式，支持 `png|bmp|jpg` |
| `--video_codec <fourcc>` | `MJPG` | capture 时视频编码 |
| `--exposure_us <float>` | `10000` | 曝光时间 |
| `--gain <float>` | `-1` | 增益；`-1` 为自动 |
| `--fps <int>` | `30` | capture 模式采集频率；decode 模式也用于参数校验 |
| `--frames <int>` | `-1` | 采集或解码的帧数上限 |
| `--writer_queue <int>` | `256` | 后台视频写线程队列长度 |
| `--preview` | 开启 | 显示预览 |
| `--no_preview` | 关闭预览 | 不显示预览 |

#### 推荐命令

```bash
./build/stereo_offline_capture_app \
  --mode capture \
  --output_dir offline_capture/session_001 \
  --calibration resources/stereo_calibration.yaml \
  --fps 30 \
  --frames 300 \
  --video_codec MJPG \
  --gain -1 \
  --no_preview
```

#### 流程说明

程序内部会：

1. 启动双目采集
2. 把每帧 BGR 图像送入后台视频写线程
3. 同时记录每帧元数据到 `manifest.yaml`

视频写线程异步工作，因此采集线程不直接阻塞在磁盘写入上。

### 4.2 第二步：解码为图像序列

主入口程序：`stereo_offline_capture_app --mode decode`

#### 功能目的

把视频包解码为图像序列，供离线估计程序读取。

#### 输出

输出目录结构如下：

```text
offline_capture/session_001_images/
├── images/cam0/
├── images/cam1/
├── calibration/
└── manifest.yaml
```

其中新的 `manifest.yaml` 会标记为 `storage_mode=image_sequence`。

#### 推荐命令

```bash
./build/stereo_offline_capture_app \
  --mode decode \
  --input_dir offline_capture/session_001 \
  --output_dir offline_capture/session_001_images \
  --image_format png \
  --no_preview
```

#### 流程说明

程序会：

1. 读取输入视频包中的 `manifest.yaml`
2. 打开 `videos/cam0.avi` 和 `videos/cam1.avi`
3. 逐帧解码
4. 按 `capture_index` 写出左右图像
5. 重建图像序列版 `manifest.yaml`

### 4.3 第三步：离线估计并保存 replay 包

主入口程序：`stereo_fused_hand_pose_offline_demo`

#### 功能目的

读取离线图像序列，完成：

1. 去畸变
2. ChArUco 检测
3. SLAM / tracking
4. 轨迹优化
5. 单视图手部估计
6. 双目融合
7. 保存完整 replay 包

#### 输入与输出

输入：

- 图像序列目录，要求包含：
  - `images/cam0 + images/cam1`
  - 或 `cam0 + cam1`
- 标定文件
- 模型文件

输出：

- `--output_dir` 下直接保存完整 replay 包
- 可选 `--offline_dump_dir` 再保存一份额外副本

当前 replay 包结构：

```text
results/stereo_fused_hand_pose_offline/
├── manifest.yaml
├── calibration/stereo_calibration.yaml
├── images/cam0/*.png
├── images/cam1/*.png
├── overlays/cam0/*.png
├── overlays/cam1/*.png
└── frames/*.yaml
```

每帧 `frames/*.yaml` 包含：

- 每路图像与推理元数据
- 每路单视图 pose 结果
- 融合手势结果
- 相机 tracking 结果
- 轨迹 `trajectory_world`

#### 调用全参数

程序：

```bash
./build_stella/stereo_fused_hand_pose_offline_demo [options]
```

参数说明：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--input_dir <dir>` | 必填 | 输入图像序列目录 |
| `--calibration <path>` | `<input_dir>/calibration/stereo_calibration.yaml` 或 `resources/stereo_calibration.yaml` | 标定文件 |
| `--detector_model <path>` | `resources/models/detector.onnx` | 检测器模型 |
| `--wilor_model <path>` | `resources/models/wilor_backbone_opset16.onnx` | WiLoR 模型 |
| `--mano_model <path>` | `resources/models/mano_cpu_opset16.onnx` | MANO 模型 |
| `--output_dir <dir>` | `results/stereo_fused_hand_pose_offline` | 主 replay 包输出目录 |
| `--offline_dump_dir <dir>` | 关闭 | 可选，再额外保存一份完整 replay 包 |
| `--slam_backend <name>` | `stella` | 选择 `stella` 或 `legacy` |
| `--stella_vocab <path>` | 关闭 | `stella` 词典路径 |
| `--stella_config_dump <path>` | 关闭 | 导出 `stella` 配置 |
| `--fps <int>` | `0` | 离线处理节奏；`0` 表示尽快处理 |
| `--frames <int>` | `-1` | 处理帧数上限 |
| `--save` | 开启 | 保存完整 replay 包 |
| `--no_save` | 关闭保存 | 不写 replay 包 |
| `--preview` | 开启 | 显示 2D 预览 |
| `--no_preview` | 关闭预览 | 不显示 2D 预览 |
| `--glfw_view` | 开启 | 显示 OpenGL 3D 视图 |
| `--no_glfw_view` | 关闭 | 不显示 OpenGL 视图 |
| `--dictionary <name>` | `DICT_APRILTAG_36h11` | ChArUco / AprilTag 字典 |
| `--squares_x <int>` | `5` | 板面 X 向方格数 |
| `--squares_y <int>` | `7` | 板面 Y 向方格数 |
| `--square_length_m <f>` | `0.028` | 方格边长（米） |
| `--marker_length_m <f>` | `0.021` | marker 边长（米） |
| `--verbose` | 开启 | 打印详细日志 |
| `--quiet` | 关闭详细日志 | 精简输出 |
| `--gpu` | 开启 | ONNX Runtime 使用 GPU |
| `--cpu` | 关闭 GPU | ONNX Runtime 使用 CPU |

#### 推荐命令

```bash
./build_stella/stereo_fused_hand_pose_offline_demo \
  --input_dir offline_capture/session_001_images \
  --calibration resources/stereo_calibration.yaml \
  --output_dir results/stereo_fused_hand_pose_offline \
  --slam_backend stella \
  --stella_config_dump debug/stella_session_001.yaml \
  --cpu \
  --dictionary DICT_APRILTAG_36h11 \
  --squares_x 5 \
  --squares_y 7 \
  --square_length_m 0.028 \
  --marker_length_m 0.021 \
  --verbose \
  --no_preview \
  --no_glfw_view
```

#### 流程说明

该程序实际分为两遍：

第一遍：

1. 读取图像对
2. 去畸变
3. 跑 ChArUco 定位
4. 跑 SLAM / tracking
5. 收集轨迹样本
6. 做轨迹优化

第二遍：

1. 重新读取图像对
2. 去畸变
3. 跑单视图手部估计
4. 双目融合
5. 叠加 tracking 信息
6. 保存 replay 包
7. 可选 2D / 3D 预览

### 4.4 第四步：从 replay 包重放

主入口程序：`stereo_fused_hand_pose_replay_app`

#### 功能目的

在不重新跑模型的情况下，直接重放第三步保存的完整 replay 包。

它会：

- 读取 `frames/*.yaml`
- 读取对应 overlay 或 raw image
- 2D 窗口重放
- OpenGL viewer 重放保存的 fused hand 与 tracking 轨迹

#### 调用全参数

程序：

```bash
./build_stella/stereo_fused_hand_pose_replay_app [options]
```

参数说明：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--input_dir <dir>` | 必填 | replay 包根目录 |
| `--fps <int>` | `30` | 回放帧率 |
| `--frames <int>` | `-1` | 回放帧数上限 |
| `--preview` | 开启 | 显示 2D 回放窗口 |
| `--no_preview` | 关闭 | 不显示 2D 窗口 |
| `--glfw_view` | 开启 | 显示 OpenGL 3D 回放 |
| `--no_glfw_view` | 关闭 | 不显示 OpenGL 视图 |
| `--verbose` | 开启 | 打印回放日志 |
| `--quiet` | 关闭详细日志 | 精简输出 |

#### 推荐命令

```bash
./build_stella/stereo_fused_hand_pose_replay_app \
  --input_dir results/stereo_fused_hand_pose_offline \
  --fps 30 \
  --preview \
  --glfw_view
```

#### 流程说明

当前 replay 已做“读帧与渲染分离”：

1. 后台线程预加载 YAML 和图片
2. 前台按回放时钟切帧
3. OpenGL 持续刷新当前帧，避免每帧 IO 阻塞交互

### 4.5 四步完整闭环命令

```bash
./build/stereo_offline_capture_app \
  --mode capture \
  --output_dir offline_capture/session_001 \
  --calibration resources/stereo_calibration.yaml \
  --fps 30 \
  --frames 300 \
  --video_codec MJPG \
  --gain -1 \
  --no_preview

./build/stereo_offline_capture_app \
  --mode decode \
  --input_dir offline_capture/session_001 \
  --output_dir offline_capture/session_001_images \
  --image_format png \
  --no_preview

./build_stella/stereo_fused_hand_pose_offline_demo \
  --input_dir offline_capture/session_001_images \
  --calibration resources/stereo_calibration.yaml \
  --output_dir results/stereo_fused_hand_pose_offline \
  --slam_backend stella \
  --stella_config_dump debug/stella_session_001.yaml \
  --cpu \
  --dictionary DICT_APRILTAG_36h11 \
  --squares_x 5 \
  --squares_y 7 \
  --square_length_m 0.028 \
  --marker_length_m 0.021 \
  --verbose \
  --no_preview \
  --no_glfw_view

./build_stella/stereo_fused_hand_pose_replay_app \
  --input_dir results/stereo_fused_hand_pose_offline \
  --fps 30 \
  --preview \
  --glfw_view
```

## 5. 附加程序说明

以下程序不作为本文正文主线，但在实际使用中经常会用到。

### 5.1 `stereo_calibration_app`

用途：

- 对已经采集好的左右图像目录做纯离线双目标定

适用场景：

- 已有 `left_dir` / `right_dir`
- 不需要交互式引导采集

帮助命令：

```bash
./build/stereo_calibration_app --help
```

### 5.2 `stereo_capture_demo`

用途：

- 只做双目采集、预览和可选保存

适用场景：

- 检查双目相机是否工作正常
- 快速验证采集链路

帮助命令：

```bash
./build/stereo_capture_demo --help
```

### 5.3 `stereo_single_view_hand_pose_demo`

用途：

- 做双目采集，但每一路独立执行单视图手部估计，不做双目融合

适用场景：

- 检查单视图估计质量
- 排查某一路检测或姿态问题

帮助命令：

```bash
./build/stereo_single_view_hand_pose_demo --help
```

### 5.4 `stereo_camera_tracking_demo`

用途：

- 只做双目采集和相机 tracking，不做人手估计

适用场景：

- 单独验证板定位 / tracking 稳定性
- 排查标定或轨迹问题

帮助命令：

```bash
./build/stereo_camera_tracking_demo --help
```

## 6. 常见问题与建议

### 6.1 标定完成后实时 demo 提示左右序列号不匹配

说明实际在线相机顺序与标定 YAML 中保存的左右序列号不一致。  
建议重新检查物理接线或重新执行引导式标定。

### 6.2 实时或离线手势估计找不到模型文件

请显式指定：

- `--detector_model`
- `--wilor_model`
- `--mano_model`

### 6.3 离线 demo 找不到输入图像

离线估计阶段输入目录必须满足以下二选一：

- `images/cam0 + images/cam1`
- `cam0 + cam1`

### 6.4 replay 很卡

当前 replay 已将帧读取和 OpenGL 渲染分离。若仍然卡顿，优先检查：

- 图片目录所在磁盘速度
- 图片分辨率过高
- OpenGL / 显卡驱动状态

### 6.5 OpenGL 中视角移动方式

当前 `W/A/S/D` 为相对当前视角的前后左右移动，`Q/E` 为世界 `Y` 方向上下移动。
