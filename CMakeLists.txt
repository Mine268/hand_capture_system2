cmake_minimum_required(VERSION 3.16)
project(newnewhand LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_definitions(NEWNEWHAND_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

option(BUILD_DEMOS "Build demo executables" ON)
option(NEWHAND_ENABLE_STELLA_VSLAM "Build vendored stella_vslam backend from third/" OFF)
set(MVS_ROOT "/opt/MVS" CACHE PATH "Hikrobot MVS SDK root directory")
set(
    ONNX_RUNTIME_DIR
    "/home/renkaiwen/src/wilor_deploy/wilor_deploy/cpp_module/onnxruntime-linux-x64-gpu-1.17.1"
    CACHE PATH
    "ONNX Runtime SDK root directory")

if(NOT OpenCV_DIR AND EXISTS "/usr/local/lib/cmake/opencv4/OpenCVConfig.cmake")
    set(OpenCV_DIR "/usr/local/lib/cmake/opencv4" CACHE PATH "Preferred OpenCV config directory")
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)
pkg_check_modules(GLEW REQUIRED glew)

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

add_library(newnewhand_fusion STATIC
    src/fusion/stereo_hand_fuser.cpp
)
target_include_directories(newnewhand_fusion
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/perception
)
target_link_libraries(newnewhand_fusion
    PUBLIC
        newnewhand_pipeline
        newnewhand_calibration
        newnewhand_visualization
        ${OpenCV_LIBS}
)

add_library(newnewhand_slam STATIC
    src/slam/stereo_visual_odometry.cpp
)
target_include_directories(newnewhand_slam
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(newnewhand_slam
    PUBLIC
        newnewhand_pipeline
        newnewhand_calibration
        ${OpenCV_LIBS}
)

if(NEWHAND_ENABLE_STELLA_VSLAM)
    add_library(newnewhand_slam_stella STATIC
        src/slam/stella_vslam_tracker.cpp
    )
    target_include_directories(newnewhand_slam_stella
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third/stella_vslam/src
    )
    target_link_libraries(newnewhand_slam_stella
        PUBLIC
            newnewhand_calibration
            newnewhand_capture
            newnewhand_slam
            stella_vslam
            yaml-cpp
            ${OpenCV_LIBS}
    )
    target_compile_definitions(newnewhand_slam_stella PUBLIC NEWNEWHAND_HAVE_STELLA_VSLAM=1)
endif()

add_library(newnewhand_io STATIC
    src/io/offline_sequence_writer.cpp
)
target_include_directories(newnewhand_io
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(newnewhand_io
    PUBLIC
        newnewhand_fusion
        newnewhand_pipeline
        newnewhand_calibration
        ${OpenCV_LIBS}
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

add_library(newnewhand_render_glfw STATIC
    src/render/glfw_scene_viewer.cpp
)
target_include_directories(newnewhand_render_glfw
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${GLFW_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
)
target_link_libraries(newnewhand_render_glfw
    PUBLIC
        newnewhand_fusion
        OpenGL::GL
        ${GLFW_LIBRARIES}
        ${GLEW_LIBRARIES}
)

if(NEWHAND_ENABLE_STELLA_VSLAM)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third/stella_vslam/CMakeLists.txt")
        message(FATAL_ERROR "NEWHAND_ENABLE_STELLA_VSLAM=ON but third/stella_vslam is missing")
    endif()
    set(USE_ARUCO OFF CACHE BOOL "" FORCE)
    set(USE_ARUCO_NANO OFF CACHE BOOL "" FORCE)
    add_subdirectory(third/stella_vslam EXCLUDE_FROM_ALL)
endif()

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

    add_executable(stereo_offline_capture_app
        apps/stereo_offline_capture_app.cpp
    )
    target_link_libraries(stereo_offline_capture_app
        PRIVATE
            newnewhand_capture
            newnewhand_calibration
    )
    set_target_properties(stereo_offline_capture_app PROPERTIES
        BUILD_RPATH "${MVS_ROOT}/lib/64"
    )

    add_executable(stereo_camera_tracking_demo
        apps/stereo_camera_tracking_demo.cpp
    )
    target_link_libraries(stereo_camera_tracking_demo
        PRIVATE
            newnewhand_capture
            newnewhand_calibration
            newnewhand_slam
            newnewhand_render_glfw
    )
    set_target_properties(stereo_camera_tracking_demo PROPERTIES
        BUILD_RPATH "${MVS_ROOT}/lib/64"
    )

    add_executable(stereo_aruco_localization_demo
        apps/stereo_aruco_localization_demo.cpp
    )
    target_link_libraries(stereo_aruco_localization_demo
        PRIVATE
            newnewhand_capture
            newnewhand_calibration
            newnewhand_render_glfw
            ${OpenCV_LIBS}
    )
    set_target_properties(stereo_aruco_localization_demo PROPERTIES
        BUILD_RPATH "${MVS_ROOT}/lib/64"
    )

    add_executable(stereo_aruco_marker_generator
        apps/stereo_aruco_marker_generator.cpp
    )
    target_link_libraries(stereo_aruco_marker_generator
        PRIVATE
            ${OpenCV_LIBS}
    )

    add_executable(stereo_charuco_board_generator
        apps/stereo_charuco_board_generator.cpp
    )
    target_link_libraries(stereo_charuco_board_generator
        PRIVATE
            ${OpenCV_LIBS}
    )

    add_executable(stereo_charuco_localization_demo
        apps/stereo_charuco_localization_demo.cpp
    )
    target_link_libraries(stereo_charuco_localization_demo
        PRIVATE
            newnewhand_capture
            newnewhand_calibration
            newnewhand_render_glfw
            ${OpenCV_LIBS}
    )
    set_target_properties(stereo_charuco_localization_demo PROPERTIES
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

    add_executable(stereo_guided_calibration_app
        apps/stereo_guided_calibration_app.cpp
    )
    target_link_libraries(stereo_guided_calibration_app
        PRIVATE
            newnewhand_capture
            newnewhand_calibration
            ${OpenCV_LIBS}
    )
    set_target_properties(stereo_guided_calibration_app PROPERTIES
        BUILD_RPATH "${MVS_ROOT}/lib/64"
    )

    add_executable(stereo_calibration_visualize_app
        apps/stereo_calibration_visualize_app.cpp
    )
    target_link_libraries(stereo_calibration_visualize_app
        PRIVATE
            newnewhand_calibration
    )

    add_executable(stereo_fused_hand_pose_demo
        apps/stereo_fused_hand_pose_demo.cpp
    )
    target_link_libraries(stereo_fused_hand_pose_demo
        PRIVATE
            newnewhand_fusion
            newnewhand_render_glfw
            newnewhand_io
    )
    set_target_properties(stereo_fused_hand_pose_demo PROPERTIES
        BUILD_RPATH "${ONNX_RUNTIME_DIR}/lib;${MVS_ROOT}/lib/64"
    )

    add_executable(stereo_fused_hand_pose_offline_demo
        apps/stereo_fused_hand_pose_offline_demo.cpp
    )
    target_link_libraries(stereo_fused_hand_pose_offline_demo
        PRIVATE
            newnewhand_fusion
            newnewhand_slam
            newnewhand_render_glfw
            newnewhand_io
            ${OpenCV_LIBS}
    )
    if(NEWHAND_ENABLE_STELLA_VSLAM)
        target_link_libraries(stereo_fused_hand_pose_offline_demo
            PRIVATE
                newnewhand_slam_stella
        )
    endif()
    set_target_properties(stereo_fused_hand_pose_offline_demo PROPERTIES
        BUILD_RPATH "${ONNX_RUNTIME_DIR}/lib"
    )
endif()
