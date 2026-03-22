#include "newnewhand/io/offline_sequence_writer.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>

namespace newnewhand {

namespace {

cv::Mat ToColumnMat(const cv::Vec3f& v) {
    return (cv::Mat_<float>(3, 1) << v[0], v[1], v[2]);
}

cv::Mat ToColumnMat(const float* values, int count) {
    cv::Mat mat(count, 1, CV_32F);
    for (int i = 0; i < count; ++i) {
        mat.at<float>(i, 0) = values[i];
    }
    return mat;
}

std::string CaptureStem(std::uint64_t capture_index) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << capture_index;
    return oss.str();
}

}  // namespace

OfflineSequenceWriter::OfflineSequenceWriter(OfflineSequenceWriterConfig config, StereoCalibrationResult calibration)
    : config_(std::move(config)),
      calibration_(std::move(calibration)) {
    if (config_.output_root.empty()) {
        throw std::invalid_argument("offline output root must not be empty");
    }
}

void OfflineSequenceWriter::Initialize() {
    if (initialized_) {
        return;
    }

    std::filesystem::create_directories(config_.output_root);
    std::filesystem::create_directories(config_.output_root / "frames");
    if (config_.save_raw_images) {
        std::filesystem::create_directories(config_.output_root / "images" / "cam0");
        std::filesystem::create_directories(config_.output_root / "images" / "cam1");
    }
    if (config_.save_overlay_images) {
        std::filesystem::create_directories(config_.output_root / "overlays" / "cam0");
        std::filesystem::create_directories(config_.output_root / "overlays" / "cam1");
    }

    SaveCalibrationFile();
    WriteManifest();
    initialized_ = true;
}

void OfflineSequenceWriter::SaveFrame(
    const StereoFrame& raw_stereo_frame,
    const StereoSingleViewPoseFrame& stereo_frame,
    const StereoFusedHandPoseFrame& fused_frame) {
    if (!initialized_) {
        Initialize();
    }

    if (config_.save_raw_images) {
        for (std::size_t camera_index = 0; camera_index < raw_stereo_frame.views.size(); ++camera_index) {
            if (!raw_stereo_frame.views[camera_index].bgr_image.empty()) {
                SaveViewImage("images", fused_frame.capture_index, raw_stereo_frame.views[camera_index].bgr_image, camera_index);
            }
        }
    }

    if (config_.save_overlay_images) {
        for (std::size_t camera_index = 0; camera_index < stereo_frame.views.size(); ++camera_index) {
            if (!stereo_frame.views[camera_index].overlay_image.empty()) {
                SaveViewImage("overlays", fused_frame.capture_index, stereo_frame.views[camera_index].overlay_image, camera_index);
            }
        }
    }

    const std::filesystem::path yaml_path =
        config_.output_root / "frames" / (CaptureStem(fused_frame.capture_index) + ".yaml");
    WriteFrameYaml(stereo_frame, fused_frame, yaml_path);
}

void OfflineSequenceWriter::WriteManifest() const {
    cv::FileStorage fs((config_.output_root / "manifest.yaml").string(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw std::runtime_error("failed to open offline manifest for writing");
    }
    fs << "format_version" << 1;
    fs << "calibration_file" << "calibration/stereo_calibration.yaml";
    fs << "has_raw_images" << static_cast<int>(config_.save_raw_images);
    fs << "has_overlay_images" << static_cast<int>(config_.save_overlay_images);
    fs << "frames_dir" << "frames";
}

void OfflineSequenceWriter::SaveCalibrationFile() const {
    const std::filesystem::path calibration_dir = config_.output_root / "calibration";
    std::filesystem::create_directories(calibration_dir);
    const std::filesystem::path output_path = calibration_dir / "stereo_calibration.yaml";
    StereoCalibrator::SaveLoadedResult(calibration_, output_path);
}

void OfflineSequenceWriter::SaveViewImage(
    const std::filesystem::path& subdir,
    std::uint64_t capture_index,
    const cv::Mat& image,
    std::size_t camera_index) const {
    const std::filesystem::path output_dir = config_.output_root / subdir / ("cam" + std::to_string(camera_index));
    std::filesystem::create_directories(output_dir);
    const std::filesystem::path output_path = output_dir / (CaptureStem(capture_index) + ".png");
    cv::imwrite(output_path.string(), image);
}

void OfflineSequenceWriter::WriteFrameYaml(
    const StereoSingleViewPoseFrame& stereo_frame,
    const StereoFusedHandPoseFrame& fused_frame,
    const std::filesystem::path& output_path) const {
    cv::FileStorage fs(output_path.string(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw std::runtime_error("failed to open offline frame yaml: " + output_path.string());
    }

    fs << "capture_index" << static_cast<double>(fused_frame.capture_index);
    fs << "calibration_file" << "../calibration/stereo_calibration.yaml";

    fs << "views" << "[";
    for (std::size_t camera_index = 0; camera_index < stereo_frame.views.size(); ++camera_index) {
        const auto& view = stereo_frame.views[camera_index];
        fs << "{";
        fs << "camera_index" << static_cast<int>(camera_index);
        fs << "serial_number" << view.camera_frame.serial_number;
        fs << "frame_index" << static_cast<double>(view.camera_frame.frame_index);
        fs << "device_timestamp" << static_cast<double>(view.camera_frame.device_timestamp);
        fs << "sdk_host_timestamp" << static_cast<double>(view.camera_frame.sdk_host_timestamp);
        fs << "image_width" << static_cast<int>(view.camera_frame.width);
        fs << "image_height" << static_cast<int>(view.camera_frame.height);
        fs << "valid" << static_cast<int>(view.camera_frame.valid);
        fs << "inference_error" << view.inference_error;
        if (config_.save_raw_images) {
            fs << "image_path" << ("../images/cam" + std::to_string(camera_index) + "/" + CaptureStem(fused_frame.capture_index) + ".png");
        }
        if (config_.save_overlay_images) {
            fs << "overlay_path" << ("../overlays/cam" + std::to_string(camera_index) + "/" + CaptureStem(fused_frame.capture_index) + ".png");
        }
        WriteHandPoseArray(fs, "hands", view.hand_poses);
        fs << "}";
    }
    fs << "]";

    WriteFusedHandArray(fs, "fused_hands", fused_frame.hands);
}

void OfflineSequenceWriter::WriteHandPoseArray(
    cv::FileStorage& fs,
    const std::string& name,
    const std::vector<HandPoseResult>& hands) const {
    fs << name << "[";
    for (const auto& hand : hands) {
        fs << "{";
        fs << "is_right" << static_cast<int>(hand.detection.is_right);
        fs << "confidence" << hand.detection.confidence;
        fs << "bbox" << cv::Mat(4, 1, CV_32F, const_cast<float*>(hand.detection.bbox)).clone();
        fs << "crop_center" << cv::Mat(2, 1, CV_32F, const_cast<float*>(hand.crop_center)).clone();
        fs << "crop_size" << hand.crop_size;
        fs << "focal_length_px" << hand.focal_length_px;
        fs << "pred_cam" << ToColumnMat(hand.pred_cam, 3);
        fs << "camera_translation" << ToColumnMat(hand.camera_translation, 3);
        fs << "global_orient" << ToColumnMat(hand.global_orient, 3);
        fs << "hand_pose" << cv::Mat(15, 3, CV_32F, const_cast<float*>(&hand.hand_pose[0][0])).clone();
        fs << "betas" << ToColumnMat(hand.betas, 10);
        fs << "keypoints_2d" << cv::Mat(21, 2, CV_32F, const_cast<float*>(&hand.keypoints_2d[0][0])).clone();
        fs << "keypoints_3d" << cv::Mat(21, 3, CV_32F, const_cast<float*>(&hand.keypoints_3d[0][0])).clone();
        fs << "vertices" << cv::Mat(778, 3, CV_32F, const_cast<float*>(&hand.vertices[0][0])).clone();
        fs << "}";
    }
    fs << "]";
}

void OfflineSequenceWriter::WriteFusedHandArray(
    cv::FileStorage& fs,
    const std::string& name,
    const std::vector<FusedHandPose>& hands) const {
    fs << name << "[";
    for (const auto& hand : hands) {
        fs << "{";
        fs << "is_right" << static_cast<int>(hand.is_right);
        fs << "fused_from_stereo" << static_cast<int>(hand.fused_from_stereo);
        fs << "has_view0" << static_cast<int>(hand.has_view0);
        fs << "has_view1" << static_cast<int>(hand.has_view1);
        fs << "root_joint_cam0" << ToColumnMat(hand.root_joint_cam0);
        fs << "pose_cam0" << "{";
        fs << "confidence" << hand.pose_cam0.detection.confidence;
        fs << "bbox" << cv::Mat(4, 1, CV_32F, const_cast<float*>(hand.pose_cam0.detection.bbox)).clone();
        fs << "pred_cam" << ToColumnMat(hand.pose_cam0.pred_cam, 3);
        fs << "camera_translation" << ToColumnMat(hand.pose_cam0.camera_translation, 3);
        fs << "global_orient" << ToColumnMat(hand.pose_cam0.global_orient, 3);
        fs << "hand_pose" << cv::Mat(15, 3, CV_32F, const_cast<float*>(&hand.pose_cam0.hand_pose[0][0])).clone();
        fs << "betas" << ToColumnMat(hand.pose_cam0.betas, 10);
        fs << "keypoints_2d" << cv::Mat(21, 2, CV_32F, const_cast<float*>(&hand.pose_cam0.keypoints_2d[0][0])).clone();
        fs << "keypoints_3d_cam0" << cv::Mat(21, 3, CV_32F, const_cast<float*>(&hand.pose_cam0.keypoints_3d[0][0])).clone();
        fs << "vertices_cam0" << cv::Mat(778, 3, CV_32F, const_cast<float*>(&hand.pose_cam0.vertices[0][0])).clone();
        fs << "}";
        fs << "}";
    }
    fs << "]";
}

}  // namespace newnewhand
