cmake_minimum_required(VERSION 3.16)
project(newnewhand LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_DEMOS "Build demo executables" ON)
set(MVS_ROOT "/opt/MVS" CACHE PATH "Hikrobot MVS SDK root directory")
set(
    ONNX_RUNTIME_DIR
    "/home/renkaiwen/src/wilor_deploy/wilor_deploy/cpp_module/onnxruntime-linux-x64-gpu-1.17.1"
    CACHE PATH
    "ONNX Runtime SDK root directory")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so.1.17.1"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNX_RUNTIME_DIR}/include"
)

add_library(newnewhand_mvs STATIC
    src/capture/mv_camera.cpp
)
target_include_directories(newnewhand_mvs
    PUBLIC
        ${MVS_ROOT}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/capture
)
target_link_directories(newnewhand_mvs
    PUBLIC
        ${MVS_ROOT}/lib/64
)
target_link_libraries(newnewhand_mvs
    PUBLIC
        MvCameraControl
        Threads::Threads
)

add_library(newnewhand_capture STATIC
    src/capture/stereo_capture.cpp
)
target_include_directories(newnewhand_capture
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/capture
)
target_link_libraries(newnewhand_capture
    PUBLIC
        newnewhand_mvs
        ${OpenCV_LIBS}
)

add_library(newnewhand_perception STATIC
    src/perception/yolo_detector.cpp
    src/perception/wilor_model.cpp
    src/perception/hand_pose_utils.cpp
    src/perception/hand_pose_estimator.cpp
)
target_include_directories(newnewhand_perception
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/perception
)
target_link_libraries(newnewhand_perception
    PUBLIC
        ${OpenCV_LIBS}
        onnxruntime
)
set_target_properties(newnewhand_perception PROPERTIES
    BUILD_RPATH "${ONNX_RUNTIME_DIR}/lib"
)

add_library(newnewhand_visualization STATIC
    src/visualization/hand_pose_overlay.cpp
)
target_include_directories(newnewhand_visualization
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(newnewhand_visualization
    PRIVATE
        NEWNEWHAND_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(newnewhand_visualization
    PUBLIC
        ${OpenCV_LIBS}
        newnewhand_perception
)

add_library(newnewhand_pipeline STATIC
    src/pipeline/stereo_single_view_hand_pose_pipeline.cpp
)
target_include_directories(newnewhand_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(newnewhand_pipeline
    PUBLIC
        newnewhand_capture
        newnewhand_perception
        newnewhand_visualization
)

add_library(newnewhand_calibration STATIC
    src/calibration/stereo_calibrator.cpp
    src/calibration/stereo_calibration_visualizer.cpp
)
target_include_directories(newnewhand_calibration
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(newnewhand_calibration
    PUBLIC
        ${OpenCV_LIBS}
)

if(BUILD_DEMOS)
    add_executable(stereo_capture_demo
        apps/stereo_capture_demo.cpp
    )
    target_link_libraries(stereo_capture_demo
        PRIVATE
            newnewhand_capture
    )
    set_target_properties(stereo_capture_demo PROPERTIES
        BUILD_RPATH "${MVS_ROOT}/lib/64"
    )

    add_executable(single_view_hand_pose_demo
        apps/single_view_hand_pose_demo.cpp
    )
    target_link_libraries(single_view_hand_pose_demo
        PRIVATE
            newnewhand_visualization
    )
    set_target_properties(single_view_hand_pose_demo PROPERTIES
        BUILD_RPATH "${ONNX_RUNTIME_DIR}/lib"
    )

    add_executable(stereo_single_view_hand_pose_demo
        apps/stereo_single_view_hand_pose_demo.cpp
    )
    target_link_libraries(stereo_single_view_hand_pose_demo
        PRIVATE
            newnewhand_pipeline
    )
    set_target_properties(stereo_single_view_hand_pose_demo PROPERTIES
        BUILD_RPATH "${ONNX_RUNTIME_DIR}/lib;${MVS_ROOT}/lib/64"
    )

    add_executable(stereo_calibration_app
        apps/stereo_calibration_app.cpp
    )
    target_link_libraries(stereo_calibration_app
        PRIVATE
            newnewhand_calibration
    )

    add_executable(stereo_calibration_visualize_app
        apps/stereo_calibration_visualize_app.cpp
    )
    target_link_libraries(stereo_calibration_visualize_app
        PRIVATE
            newnewhand_calibration
    )
endif()
