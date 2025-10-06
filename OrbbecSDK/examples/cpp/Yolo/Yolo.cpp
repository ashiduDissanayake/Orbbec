// SimpleTwoDistanceBallDetector.cpp - FIXED VERSION
// Corrected YOLOv8 output parsing for sports ball detection

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <fstream>
#include <ctime>
#include <sstream>

#include "libobsensor/hpp/Context.hpp"
#include "libobsensor/hpp/Error.hpp"
#include "libobsensor/hpp/Pipeline.hpp"
#include "libobsensor/hpp/StreamProfile.hpp"
#include "window.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// ============================================================================
// CONFIGURATION
// ============================================================================

struct Config {
    std::string modelPath = "./yolov8n.onnx";
    float confThreshold = 0.20f;  // Lowered for better detection
    
    // Two target distances
    float distance1 = 2.0f;
    float distance2 = 1.5f;
    float tolerance = 0.15f;
    
    std::string captureDir = "./ball_captures";
};

// ============================================================================
// UTILITIES
// ============================================================================

OBCameraIntrinsic createDefaultIntrinsics(uint32_t width, uint32_t height) {
    OBCameraIntrinsic intrinsic{};
    intrinsic.width = static_cast<int>(width);
    intrinsic.height = static_cast<int>(height);
    
    const float fovDegrees = 60.0f;
    const float fovRadians = fovDegrees * M_PI / 180.0f;
    const float focal = static_cast<float>(width) / (2.0f * std::tan(fovRadians / 2.0f));
    
    intrinsic.fx = focal;
    intrinsic.fy = focal;
    intrinsic.cx = static_cast<float>(width) / 2.0f;
    intrinsic.cy = static_cast<float>(height) / 2.0f;
    
    return intrinsic;
}

std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
    return ss.str();
}

// ============================================================================
// BALL DETECTION
// ============================================================================

struct BallDetection {
    cv::Rect bbox;
    cv::Point2f center;
    float confidence;
    float depth;
};

class YOLOBallDetector {
public:
    YOLOBallDetector(const std::string &modelPath, float confThreshold) 
        : confThreshold_(confThreshold) {
        
        std::cout << "🤖 Loading YOLO: " << modelPath << std::endl;
        
        std::ifstream test(modelPath);
        if (!test.good()) {
            throw std::runtime_error("Model not found: " + modelPath);
        }
        
        net_ = cv::dnn::readNetFromONNX(modelPath);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        std::cout << "✅ YOLO loaded\n" << std::endl;
    }

    std::vector<BallDetection> detect(const cv::Mat &frame, 
                                  const uint16_t *depthData,
                                  uint32_t depthWidth, 
                                  uint32_t depthHeight) {
    
    std::vector<BallDetection> detections;
    
    if (frame.empty() || !depthData) return detections;

    // Resize to depth resolution
    cv::Mat resized;
    if (frame.cols != static_cast<int>(depthWidth) || 
        frame.rows != static_cast<int>(depthHeight)) {
        cv::resize(frame, resized, cv::Size(depthWidth, depthHeight));
    } else {
        resized = frame;
    }

    // YOLO inference
    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0/255.0, 
                                         cv::Size(640, 640), 
                                         cv::Scalar(), true, false);
    
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    if (outputs.empty()) return detections;

    // Parse YOLOv8 output
    cv::Mat output = outputs[0];
    
    std::cout << "📊 Raw output dims: " << output.dims 
              << " shape: [";
    for (int i = 0; i < output.dims; i++) {
        std::cout << output.size[i];
        if (i < output.dims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Transpose if needed
    if (output.dims == 3 && output.size[0] == 1) {
        output = output.reshape(1, output.size[1]);
        cv::transpose(output, output);
    }

    std::cout << "📊 After reshape: " << output.rows << " x " << output.cols << std::endl;

    float scaleX = static_cast<float>(resized.cols) / 640.0f;
    float scaleY = static_cast<float>(resized.rows) / 640.0f;

    // 🔍 DEBUG: Check what classes have high scores
    std::map<int, int> classCount;
    std::map<int, float> maxClassScore;
    
    for (int i = 0; i < std::min(100, output.rows); i++) { // Check first 100 detections
        float maxScore = 0.0f;
        int maxClass = -1;
        
        // Find max class
        for (int c = 4; c < output.cols; c++) {
            float score = output.at<float>(i, c);
            if (score > maxScore) {
                maxScore = score;
                maxClass = c - 4;
            }
        }
        
        if (maxScore > 0.1f) { // Any detection > 10%
            classCount[maxClass]++;
            if (maxScore > maxClassScore[maxClass]) {
                maxClassScore[maxClass] = maxScore;
            }
        }
    }
    
    // Print what we found
    std::cout << "\n🔍 Top detected classes (first 100 detections):" << std::endl;
    std::vector<std::pair<int, int>> sortedClasses(classCount.begin(), classCount.end());
    std::sort(sortedClasses.begin(), sortedClasses.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });
    
    for (size_t i = 0; i < std::min(size_t(5), sortedClasses.size()); i++) {
        int cls = sortedClasses[i].first;
        std::cout << "  Class " << cls << ": " << sortedClasses[i].second 
                  << " detections (max score: " << maxClassScore[cls] << ")";
        if (cls == 32) std::cout << " ← SPORTS BALL!";
        std::cout << std::endl;
    }
    
    // Now try to find balls with VERY LOW threshold
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    
    for (int i = 0; i < output.rows; i++) {
        float cx = output.at<float>(i, 0) * scaleX;
        float cy = output.at<float>(i, 1) * scaleY;
        float w = output.at<float>(i, 2) * scaleX;
        float h = output.at<float>(i, 3) * scaleY;
        
        // Find max class
        float maxScore = 0.0f;
        int maxClass = -1;
        
        for (int c = 4; c < output.cols; c++) {
            float score = output.at<float>(i, c);
            if (score > maxScore) {
                maxScore = score;
                maxClass = c - 4;
            }
        }
        
        // 🔍 SUPER LOW THRESHOLD for debugging (10%)
        if (maxClass == 32 && maxScore > 0.10f) {
            std::cout << "  🎾 Found ball candidate! Score: " << maxScore 
                      << " at (" << cx << "," << cy << ")" << std::endl;
            
            int x = std::max(0, static_cast<int>(cx - w/2));
            int y = std::max(0, static_cast<int>(cy - h/2));
            int bw = std::min(static_cast<int>(w), resized.cols - x);
            int bh = std::min(static_cast<int>(h), resized.rows - y);
            
            boxes.push_back(cv::Rect(x, y, bw, bh));
            confidences.push_back(maxScore);
        }
    }

    std::cout << "  Total ball candidates: " << boxes.size() << std::endl;

    // NMS
    std::vector<int> indices;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, confidences, 0.10f, 0.45f, indices);
        std::cout << "  After NMS: " << indices.size() << " ball(s)\n" << std::endl;
    } else {
        std::cout << "  ❌ No ball candidates found\n" << std::endl;
    }

    // Create detections
    for (int idx : indices) {
        BallDetection det;
        det.bbox = boxes[idx];
        det.confidence = confidences[idx];
        
        det.center.x = det.bbox.x + det.bbox.width / 2.0f;
        det.center.y = det.bbox.y + det.bbox.height / 2.0f;
        
        det.depth = getDepthAt(det.center.x, det.center.y, 
                               depthData, depthWidth, depthHeight);
        
        detections.push_back(det);
    }

    return detections;
}

private:
    float getDepthAt(float x, float y, const uint16_t *depthData, 
                     uint32_t width, uint32_t height) {
        
        int px = static_cast<int>(x);
        int py = static_cast<int>(y);
        
        if (px < 0 || px >= static_cast<int>(width) || 
            py < 0 || py >= static_cast<int>(height)) {
            return 0.0f;
        }

        uint16_t raw = depthData[py * width + px];
        
        if (raw == 0) {
            std::vector<uint16_t> samples;
            for (int dy = -5; dy <= 5; dy += 5) {
                for (int dx = -5; dx <= 5; dx += 5) {
                    int sx = px + dx;
                    int sy = py + dy;
                    if (sx >= 0 && sx < static_cast<int>(width) && 
                        sy >= 0 && sy < static_cast<int>(height)) {
                        uint16_t s = depthData[sy * width + sx];
                        if (s > 0) samples.push_back(s);
                    }
                }
            }
            if (!samples.empty()) {
                std::sort(samples.begin(), samples.end());
                raw = samples[samples.size()/2];
            }
        }

        return raw * 0.01f; // cm → m
    }

    cv::dnn::Net net_;
    float confThreshold_;
};

// ============================================================================
// TWO-DISTANCE TRIGGER
// ============================================================================

class TwoDistanceTrigger {
public:
    TwoDistanceTrigger(float d1, float d2, float tol) 
        : dist1_(d1), dist2_(d2), tolerance_(tol) {
        
        triggered_[d1] = false;
        triggered_[d2] = false;
        count_[d1] = 0;
        count_[d2] = 0;
        
        std::cout << "🎯 Target Distances:" << std::endl;
        std::cout << "   Distance 1: " << d1 << "m ±" << tol << "m" << std::endl;
        std::cout << "   Distance 2: " << d2 << "m ±" << tol << "m\n" << std::endl;
    }
    
    bool checkTrigger(float depth, float &triggeredDistance) {
        if (std::abs(depth - dist1_) <= tolerance_) {
            if (!triggered_[dist1_]) {
                triggered_[dist1_] = true;
                triggered_[dist2_] = false;
                triggeredDistance = dist1_;
                count_[dist1_]++;
                return true;
            }
        }
        else if (std::abs(depth - dist2_) <= tolerance_) {
            if (!triggered_[dist2_]) {
                triggered_[dist2_] = true;
                triggered_[dist1_] = false;
                triggeredDistance = dist2_;
                count_[dist2_]++;
                return true;
            }
        }
        else if (depth > std::max(dist1_, dist2_) + 0.5f) {
            triggered_[dist1_] = false;
            triggered_[dist2_] = false;
        }
        
        return false;
    }
    
    bool isInZone(float depth, float &zone) const {
        if (std::abs(depth - dist1_) <= tolerance_) {
            zone = dist1_;
            return true;
        }
        if (std::abs(depth - dist2_) <= tolerance_) {
            zone = dist2_;
            return true;
        }
        return false;
    }
    
    void printSummary() const {
        std::cout << "\n╔══════════════════════════════════════════╗" << std::endl;
        std::cout << "║   📊 CAPTURE SUMMARY                     ║" << std::endl;
        std::cout << "╠══════════════════════════════════════════╣" << std::endl;
        std::cout << "║   " << dist1_ << "m: " << std::setw(3) << count_.at(dist1_) 
                  << " captures                    ║" << std::endl;
        std::cout << "║   " << dist2_ << "m: " << std::setw(3) << count_.at(dist2_) 
                  << " captures                    ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════╝\n" << std::endl;
    }

private:
    float dist1_, dist2_, tolerance_;
    std::map<float, bool> triggered_;
    std::map<float, int> count_;
};

// ============================================================================
// CAPTURE MANAGER
// ============================================================================

class CaptureManager {
public:
    CaptureManager(const std::string &dir) : dir_(dir) {
        system(("mkdir -p " + dir_).c_str());
        
        log_.open(dir_ + "/captures.csv");
        log_ << "Timestamp,TargetDistance_m,MeasuredDepth_m,Confidence,ImageFile\n";
        
        std::cout << "📂 Output: " << dir_ << "\n" << std::endl;
    }

    ~CaptureManager() {
        if (log_.is_open()) log_.close();
    }

    void capture(const BallDetection &ball, float targetDist, const cv::Mat &frame) {
        std::string ts = getTimestamp();
        std::stringstream fn;
        fn << "ball_" << targetDist << "m_" << ts << ".jpg";
        std::string filename = fn.str();
        
        cv::Mat img = frame.clone();
        cv::rectangle(img, ball.bbox, cv::Scalar(0, 255, 0), 3);
        cv::circle(img, ball.center, 5, cv::Scalar(0, 0, 255), -1);
        
        std::stringstream label;
        label << std::fixed << std::setprecision(2) 
              << "Depth: " << ball.depth << "m (" 
              << static_cast<int>(ball.confidence*100) << "%)";
        
        cv::putText(img, label.str(), 
                   cv::Point(ball.bbox.x, ball.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        cv::imwrite(dir_ + "/" + filename, img);
        
        log_ << ts << "," << targetDist << "," << ball.depth << "," 
             << ball.confidence << "," << filename << "\n";
        log_.flush();
        
        std::cout << "📸 CAPTURED at " << targetDist << "m → " << filename << std::endl;
    }

private:
    std::string dir_;
    std::ofstream log_;
};

// ============================================================================
// MAIN
// ============================================================================

int main(int, char**) try {
    Config config;
    
    std::cout << "\n╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   🎯 TWO-DISTANCE BALL DETECTOR            ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝\n" << std::endl;

    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);
    ob::Pipeline pipe;
    auto pipeConfig = std::make_shared<ob::Config>();
    pipeConfig->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_DISABLE);

    std::shared_ptr<ob::VideoStreamProfile> depthProfile;
    auto depthProfiles = pipe.getStreamProfileList(OB_SENSOR_DEPTH);
    
    if (depthProfiles && depthProfiles->count() > 0) {
        for (uint32_t i = 0; i < depthProfiles->count(); ++i) {
            auto profile = depthProfiles->getProfile(i);
            if (profile && profile->is<ob::VideoStreamProfile>()) {
                auto vp = profile->as<ob::VideoStreamProfile>();
                if (vp->width() == 640) {
                    depthProfile = vp;
                    break;
                }
            }
        }
        if (!depthProfile) {
            depthProfile = depthProfiles->getProfile(0)->as<ob::VideoStreamProfile>();
        }
        pipeConfig->enableStream(depthProfile);
    }

    if (!depthProfile) {
        throw std::runtime_error("No depth profile");
    }

    pipe.start(pipeConfig);

    std::cout << "✅ Depth: " << depthProfile->width() << "×" 
              << depthProfile->height() << std::endl;

    Window depthWindow("Depth", depthProfile->width(), depthProfile->height());
    
    YOLOBallDetector detector(config.modelPath, config.confThreshold);
    TwoDistanceTrigger trigger(config.distance1, config.distance2, config.tolerance);
    CaptureManager captureManager(config.captureDir);

    cv::VideoCapture cap(0, cv::CAP_AVFOUNDATION);
    if (!cap.isOpened()) {
        throw std::runtime_error("Cannot open camera");
    }

    std::cout << "✅ Camera opened\n" << std::endl;
    std::cout << "Press 'Q' to quit\n" << std::endl;

    uint64_t frameCount = 0;

    while (depthWindow) {
        auto frameSet = pipe.waitForFrames(100);
        if (!frameSet) continue;

        auto depthFrame = frameSet->depthFrame();
        if (!depthFrame) continue;

        depthWindow.addToRender(depthFrame);

        const uint32_t dw = depthFrame->width();
        const uint32_t dh = depthFrame->height();
        const uint16_t *dd = reinterpret_cast<const uint16_t*>(depthFrame->data());

        cv::Mat colorFrame;
        if (!cap.read(colorFrame)) continue;

        frameCount++;

        // Only run detection every 5 frames (for debugging)
        if (frameCount % 5 == 0) {
            std::cout << "\n🔍 Frame " << frameCount << " - Running detection..." << std::endl;
            auto balls = detector.detect(colorFrame, dd, dw, dh);

            if (!balls.empty()) {
                const auto &ball = balls[0];
                
                if (ball.depth > 0.3f && ball.depth < 5.0f) {
                    float triggeredAt;
                    if (trigger.checkTrigger(ball.depth, triggeredAt)) {
                        std::cout << "\n🎯 TRIGGER at " << triggeredAt << "m! "
                                  << "(measured: " << ball.depth << "m)\n" << std::endl;
                        
                        captureManager.capture(ball, triggeredAt, colorFrame);
                    }
                    
                    float zone;
                    if (trigger.isInZone(ball.depth, zone)) {
                        std::cout << "✅ In zone: " << zone << "m (depth: " 
                                  << ball.depth << "m)" << std::endl;
                    } else {
                        std::cout << "📍 Current depth: " << ball.depth << "m" << std::endl;
                    }
                }
            } else {
                std::cout << "❌ No balls detected this frame" << std::endl;
            }
        }

        // Simple visualization without detection overlay
        cv::Mat vis;
        cv::resize(colorFrame, vis, cv::Size(dw, dh));
        cv::imshow("Ball Detection", vis);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    pipe.stop();
    
    trigger.printSummary();
    std::cout << "✅ Done\n" << std::endl;
    
    return 0;

} catch (const std::exception &e) {
    std::cerr << "❌ Error: " << e.what() << std::endl;
    return -1;
}