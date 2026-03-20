#pragma once

#include <filesystem>
#include <string>

#include "newnewhand/calibration/stereo_calibrator.h"
#include "newnewhand/fusion/stereo_hand_fuser.h"
#include "newnewhand/pipeline/stereo_single_view_hand_pose_pipeline.h"

namespace newnewhand {

struct OfflineSequenceWriterConfig {
    std::filesystem::path output_root;
    std::string calibration_source_path;
    bool save_raw_images = true;
    bool save_overlay_images = true;
};

class OfflineSequenceWriter {
public:
    OfflineSequenceWriter(OfflineSequenceWriterConfig config, StereoCalibrationResult calibration);

    void Initialize();
    void SaveFrame(
        const StereoFrame& raw_stereo_frame,
        const StereoSingleViewPoseFrame& stereo_frame,
        const StereoFusedHandPoseFrame& fused_frame);

private:
    void WriteManifest() const;
    void SaveCalibrationFile() const;
    void SaveViewImage(
        const std::filesystem::path& subdir,
        std::uint64_t capture_index,
        const cv::Mat& image,
        std::size_t camera_index) const;
    void WriteFrameYaml(
        const StereoSingleViewPoseFrame& stereo_frame,
        const StereoFusedHandPoseFrame& fused_frame,
        const std::filesystem::path& output_path) const;
    void WriteHandPoseArray(
        cv::FileStorage& fs,
        const std::string& name,
        const std::vector<HandPoseResult>& hands) const;
    void WriteFusedHandArray(
        cv::FileStorage& fs,
        const std::string& name,
        const std::vector<FusedHandPose>& hands) const;

    OfflineSequenceWriterConfig config_;
    StereoCalibrationResult calibration_;
    bool initialized_ = false;
};

}  // namespace newnewhand
