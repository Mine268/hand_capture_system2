#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "newnewhand/calibration/stereo_calibrator.h"
#include "newnewhand/slam/stereo_visual_odometry.h"
#include "newnewhand/types/stereo_frame.h"

namespace newnewhand {

struct StellaStereoSlamTrackerConfig {
    StereoCalibrationResult calibration;
    std::string vocab_path;
    double nominal_fps = 30.0;
    bool verbose_logging = true;
    std::filesystem::path generated_config_dump_path;
};

class StellaStereoSlamTracker {
public:
    explicit StellaStereoSlamTracker(StellaStereoSlamTrackerConfig config);
    ~StellaStereoSlamTracker();

    StellaStereoSlamTracker(const StellaStereoSlamTracker&) = delete;
    StellaStereoSlamTracker& operator=(const StellaStereoSlamTracker&) = delete;

    StereoCameraTrackingResult Track(const StereoFrame& raw_stereo_frame);
    void Reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace newnewhand
