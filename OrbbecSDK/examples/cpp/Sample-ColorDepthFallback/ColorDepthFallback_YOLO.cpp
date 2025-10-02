// ColorDepthFallback_YOLO.cpp - Production YOLO Ball Detection System
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <thread>
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
    // YOLO Model
    std::string modelPath = "./best.onnx";
    float confThreshold = 0.30f;      // Confidence threshold
    float nmsThreshold = 0.45f;       // NMS threshold
    
    // Ball physical constraints
    float minRadius3D = 0.06f;        // meters
    float maxRadius3D = 0.18f;        // meters
    float minDepth = 0.5f;            // meters
    float maxDepth = 4.0f;            // meters
    
    // Tracking
    int maxFramesLost = 5;            // Frames before losing track
    float maxVelocity = 5.0f;         // m/s
    float outlierThreshold = 0.5f;    // meters
    
    // Capture
    float captureThreshold = 2.0f;    // meters
    float captureHysteresis = 0.05f;  // meters
    std::string captureDir = "./ball_captures";
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void printIntrinsics(const OBCameraIntrinsic &intrinsic) {
    std::cout << "\n=== Depth Camera Intrinsics ===" << std::endl;
    std::cout << "  Resolution: " << intrinsic.width << "×" << intrinsic.height << std::endl;
    std::cout << "  Focal Length: fx=" << intrinsic.fx << ", fy=" << intrinsic.fy << std::endl;
    std::cout << "  Principal Point: cx=" << intrinsic.cx << ", cy=" << intrinsic.cy << std::endl;
    std::cout << "================================\n" << std::endl;
}

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

// ============================================================================
// DEPTH PROJECTOR
// ============================================================================

class DepthProjector {
public:
    DepthProjector(const OBCameraIntrinsic &intr, float valueScale)
        : intrinsic_(intr), valueScale_(valueScale) {}

    std::array<float, 3> project(uint16_t depthRaw, float pixelX, float pixelY) const {
        // Orbbec Astra typically outputs in mm, convert to meters
        const float z = depthRaw * valueScale_ * 0.001f;
        
        if (z <= 0.0f || intrinsic_.fx <= 0.0f || intrinsic_.fy <= 0.0f) {
            return {0.f, 0.f, 0.f};
        }
        
        const float x = (pixelX - intrinsic_.cx) * z / intrinsic_.fx;
        const float y = (pixelY - intrinsic_.cy) * z / intrinsic_.fy;
        
        return {x, y, z};
    }

    const OBCameraIntrinsic& intrinsic() const { return intrinsic_; }
    float valueScale() const { return valueScale_; }

private:
    OBCameraIntrinsic intrinsic_;
    float valueScale_;
};

// ============================================================================
// UNIFIED BALL STRUCTURE
// ============================================================================

struct Ball {
    cv::Point2f center2D;
    std::array<float, 3> position3D;
    std::array<float, 3> velocity3D;
    float radius2D;
    float radius3D;
    float confidence;
    bool isPredicted;
    int trackingFrames;
};

// ============================================================================
// YOLO DETECTOR
// ============================================================================

class YOLODetector {
public:
    YOLODetector(const Config &config) : config_(config) {
        std::cout << "🤖 Loading YOLO model: " << config_.modelPath << std::endl;
        
        // Check if model file exists
        std::ifstream modelFile(config_.modelPath);
        if (!modelFile.good()) {
            throw std::runtime_error("Model file not found: " + config_.modelPath);
        }
        
        try {
            net_ = cv::dnn::readNetFromONNX(config_.modelPath);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            
            modelLoaded_ = true;
            std::cout << "✅ YOLO model loaded successfully!" << std::endl;
            
        } catch (const cv::Exception &e) {
            throw std::runtime_error("Failed to load YOLO model: " + std::string(e.what()));
        }
    }

    std::vector<Ball> detect(const cv::Mat &colorFrame, 
                             const uint16_t *depthData,
                             uint32_t width, uint32_t height,
                             const DepthProjector &projector) {
        
        std::vector<Ball> balls;
        
        if (!modelLoaded_ || colorFrame.empty() || !depthData) {
            return balls;
        }

        // Resize to depth resolution
        cv::Mat resizedColor;
        if (colorFrame.cols != static_cast<int>(width) || 
            colorFrame.rows != static_cast<int>(height)) {
            cv::resize(colorFrame, resizedColor, cv::Size(width, height));
        } else {
            resizedColor = colorFrame;
        }

        // Prepare YOLO input
        cv::Mat blob = cv::dnn::blobFromImage(
            resizedColor, 1.0 / 255.0, cv::Size(640, 640),
            cv::Scalar(0, 0, 0), true, false
        );

        // Run inference
        net_.setInput(blob);
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        if (outputs.empty() || outputs[0].empty()) {
            return balls;
        }

        // Parse YOLOv8 output [1, 5, 8400] -> [8400, 5]
        cv::Mat output = outputs[0];
        if (output.dims == 3 && output.size[0] == 1) {
            output = output.reshape(1, output.size[1]);
        }
        cv::Mat transposed;
        cv::transpose(output, transposed);

        // Scale factors
        float scaleX = static_cast<float>(width) / 640.0f;
        float scaleY = static_cast<float>(height) / 640.0f;

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;

        // Parse detections
        for (int i = 0; i < transposed.rows; ++i) {
            float confidence = transposed.at<float>(i, 4);
            
            if (confidence >= config_.confThreshold) {
                float cx = transposed.at<float>(i, 0) * scaleX;
                float cy = transposed.at<float>(i, 1) * scaleY;
                float w = transposed.at<float>(i, 2) * scaleX;
                float h = transposed.at<float>(i, 3) * scaleY;

                int x = std::max(0, static_cast<int>(cx - w / 2.0f));
                int y = std::max(0, static_cast<int>(cy - h / 2.0f));
                w = std::min(static_cast<float>(width - x), w);
                h = std::min(static_cast<float>(height - y), h);

                boxes.push_back(cv::Rect(x, y, static_cast<int>(w), static_cast<int>(h)));
                confidences.push_back(confidence);
            }
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, config_.confThreshold, config_.nmsThreshold, indices);

        // Convert to Ball objects with 3D info
        for (int idx : indices) {
            Ball ball = createBallFrom2D(boxes[idx], confidences[idx], 
                                        depthData, width, height, projector);
            
            if (ball.position3D[2] > 0.0f) {  // Valid depth
                balls.push_back(ball);
            }
        }

        // Sort by confidence
        std::sort(balls.begin(), balls.end(),
                  [](const Ball &a, const Ball &b) { return a.confidence > b.confidence; });

        return balls;
    }

private:
    Ball createBallFrom2D(const cv::Rect &box, float confidence,
                          const uint16_t *depthData, uint32_t width, uint32_t height,
                          const DepthProjector &projector) {
        
        Ball ball{};
        ball.center2D = cv::Point2f(box.x + box.width / 2.0f, box.y + box.height / 2.0f);
        ball.radius2D = std::max(box.width, box.height) / 2.0f;
        ball.confidence = confidence;
        ball.isPredicted = false;
        ball.trackingFrames = 1;
        ball.velocity3D = {0.f, 0.f, 0.f};

        // Get depth - try center first, then sample around if no depth
        uint16_t rawDepth = sampleDepth(ball.center2D, ball.radius2D, 
                                        depthData, width, height);

        if (rawDepth > 0) {
            ball.position3D = projector.project(rawDepth, ball.center2D.x, ball.center2D.y);
            
            // Calculate 3D radius
            const auto &intr = projector.intrinsic();
            ball.radius3D = (ball.radius2D * ball.position3D[2]) / intr.fx;
            
            // Validate physical constraints
            if (ball.radius3D < config_.minRadius3D || ball.radius3D > config_.maxRadius3D) {
                ball.position3D[2] = 0.0f;  // Mark as invalid
            }
            if (ball.position3D[2] < config_.minDepth || ball.position3D[2] > config_.maxDepth) {
                ball.position3D[2] = 0.0f;  // Mark as invalid
            }
        }

        return ball;
    }

    uint16_t sampleDepth(cv::Point2f center, float radius,
                        const uint16_t *depthData, uint32_t width, uint32_t height) {
        
        int ix = static_cast<int>(center.x);
        int iy = static_cast<int>(center.y);
        
        if (ix < 0 || ix >= static_cast<int>(width) || 
            iy < 0 || iy >= static_cast<int>(height)) {
            return 0;
        }

        uint16_t centerDepth = depthData[iy * width + ix];
        if (centerDepth > 0) {
            return centerDepth;
        }

        // Sample around circle
        int samples = 0;
        int totalDepth = 0;
        
        for (int angle = 0; angle < 360; angle += 30) {
            float rad = angle * M_PI / 180.0f;
            int sx = static_cast<int>(center.x + radius * 0.7f * cos(rad));
            int sy = static_cast<int>(center.y + radius * 0.7f * sin(rad));
            
            if (sx >= 0 && sx < static_cast<int>(width) && 
                sy >= 0 && sy < static_cast<int>(height)) {
                uint16_t sampleDepth = depthData[sy * width + sx];
                if (sampleDepth > 0) {
                    totalDepth += sampleDepth;
                    samples++;
                }
            }
        }
        
        return (samples > 0) ? (totalDepth / samples) : 0;
    }

    Config config_;
    cv::dnn::Net net_;
    bool modelLoaded_ = false;
};

// ============================================================================
// BALL TRACKER
// ============================================================================

class BallTracker {
public:
    BallTracker(const Config &config) : config_(config) {}

    void update(const std::vector<Ball> &detections, double dt) {
        dt = std::max(0.001, std::min(0.1, dt));

        if (detections.empty()) {
            handleNoDetection(dt);
        } else {
            handleDetection(detections[0], dt);
        }
    }

    bool hasTrack() const {
        return tracked_.trackingFrames > 0 && 
               tracked_.trackingFrames - lastSeenFrame_ <= config_.maxFramesLost;
    }

    const Ball& getTrackedBall() const {
        return tracked_;
    }

private:
    void handleNoDetection(double dt) {
        if (tracked_.trackingFrames == 0) return;

        int framesSinceSeen = tracked_.trackingFrames - lastSeenFrame_;
        
        if (framesSinceSeen < config_.maxFramesLost) {
            // Predict using velocity
            for (int i = 0; i < 3; ++i) {
                float clampedVel = std::max(-config_.maxVelocity, 
                                           std::min(config_.maxVelocity, tracked_.velocity3D[i]));
                tracked_.position3D[i] += clampedVel * dt;
            }
            
            tracked_.position3D[2] = std::max(config_.minDepth, 
                                             std::min(config_.maxDepth, tracked_.position3D[2]));
            tracked_.isPredicted = true;
            tracked_.confidence *= 0.85f;
            tracked_.trackingFrames++;
        } else {
            // Lost track
            tracked_.trackingFrames = 0;
        }
    }

    void handleDetection(const Ball &detection, double dt) {
        if (tracked_.trackingFrames == 0) {
            // Initialize tracking
            tracked_ = detection;
            lastSeenFrame_ = 1;
        } else {
            // Check for outliers
            float dx = detection.position3D[0] - tracked_.position3D[0];
            float dy = detection.position3D[1] - tracked_.position3D[1];
            float dz = detection.position3D[2] - tracked_.position3D[2];
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            if (tracked_.isPredicted && dist > config_.outlierThreshold) {
                // Reject outlier
                tracked_.trackingFrames++;
                return;
            }
            
            // Update velocity
            if (dt > 0.001 && !tracked_.isPredicted) {
                float velAlpha = 0.3f;
                tracked_.velocity3D[0] = velAlpha * (dx / dt) + (1-velAlpha) * tracked_.velocity3D[0];
                tracked_.velocity3D[1] = velAlpha * (dy / dt) + (1-velAlpha) * tracked_.velocity3D[1];
                tracked_.velocity3D[2] = velAlpha * (dz / dt) + (1-velAlpha) * tracked_.velocity3D[2];
            }
            
            // Smooth position
            float alpha = 0.7f;
            for (int i = 0; i < 3; ++i) {
                tracked_.position3D[i] = alpha * detection.position3D[i] + 
                                        (1-alpha) * tracked_.position3D[i];
            }
            
            tracked_.center2D = detection.center2D;
            tracked_.radius3D = 0.7f * detection.radius3D + 0.3f * tracked_.radius3D;
            tracked_.confidence = detection.confidence;
            tracked_.isPredicted = false;
            tracked_.trackingFrames++;
            lastSeenFrame_ = tracked_.trackingFrames;
        }
    }

    Config config_;
    Ball tracked_{};
    int lastSeenFrame_ = 0;
};

// ============================================================================
// CAPTURE MANAGER
// ============================================================================

class CaptureManager {
public:
    CaptureManager(const Config &config) : config_(config), captureCount_(0) {
        lastDepth_ = config_.captureThreshold + config_.captureHysteresis + 1.0f;
        
        system(("mkdir -p " + config_.captureDir).c_str());
        
        logFile_.open(config_.captureDir + "/positions.csv");
        logFile_ << "CaptureID,Timestamp,X_m,Y_m,Z_m,Confidence,ImageFile\n";
        
        std::cout << "📂 Captures will be saved to: " << config_.captureDir << "\n" << std::endl;
    }

    ~CaptureManager() {
        if (logFile_.is_open()) {
            logFile_.close();
        }
    }

    bool shouldCapture(float currentDepth) {
        bool crossed = (lastDepth_ > config_.captureThreshold + config_.captureHysteresis) && 
                      (currentDepth <= config_.captureThreshold);
        lastDepth_ = currentDepth;
        return crossed;
    }

    void capture(const Ball &ball, const cv::Mat &frame) {
        captureCount_++;
        
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream timestampSS;
        timestampSS << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
        std::string timestamp = timestampSS.str();
        
        std::stringstream filenameSS;
        filenameSS << "capture_" << std::setfill('0') << std::setw(3) << captureCount_ 
                   << "_" << timestamp << ".jpg";
        std::string filename = filenameSS.str();
        std::string fullPath = config_.captureDir + "/" + filename;
        
        cv::Mat captureFrame = frame.clone();
        drawCaptureInfo(captureFrame, ball, captureCount_);
        
        cv::imwrite(fullPath, captureFrame);
        
        logFile_ << captureCount_ << ","
                 << timestamp << ","
                 << std::fixed << std::setprecision(3)
                 << ball.position3D[0] << ","
                 << ball.position3D[1] << ","
                 << ball.position3D[2] << ","
                 << ball.confidence << ","
                 << filename << "\n";
        logFile_.flush();
        
        printCaptureInfo(ball, filename);
    }

private:
    void drawCaptureInfo(cv::Mat &frame, const Ball &ball, int captureNum) {
        cv::rectangle(frame, cv::Point(10, 10), cv::Point(400, 180), 
                     cv::Scalar(0, 0, 0), -1);
        cv::rectangle(frame, cv::Point(10, 10), cv::Point(400, 180), 
                     cv::Scalar(0, 255, 255), 3);
        
        std::stringstream ss;
        ss << "CAPTURE #" << captureNum;
        cv::putText(frame, ss.str(), cv::Point(20, 45), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        
        std::stringstream posX, posY, posZ, conf;
        posX << "X: " << std::fixed << std::setprecision(3) << ball.position3D[0] << " m";
        posY << "Y: " << std::fixed << std::setprecision(3) << ball.position3D[1] << " m";
        posZ << "Z: " << std::fixed << std::setprecision(3) << ball.position3D[2] << " m (2m THRESHOLD)";
        conf << "Confidence: " << std::fixed << std::setprecision(1) << (ball.confidence * 100) << "%";
        
        cv::putText(frame, posX.str(), cv::Point(20, 80), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, posY.str(), cv::Point(20, 110), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, posZ.str(), cv::Point(20, 140), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, conf.str(), cv::Point(20, 170), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }

    void printCaptureInfo(const Ball &ball, const std::string &filename) {
        std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║              📸 CAPTURE #" << captureCount_ << " - 2.0m THRESHOLD              ║" << std::endl;
        std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║  Position (X, Y, Z): (" << std::fixed << std::setprecision(3)
                  << ball.position3D[0] << ", " << ball.position3D[1] << ", " 
                  << ball.position3D[2] << ") m     ║" << std::endl;
        std::cout << "║  Confidence: " << std::setprecision(1) << (ball.confidence*100) << "%                                            ║" << std::endl;
        std::cout << "║  File: " << std::setw(51) << std::left << filename << "║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n" << std::endl;
    }

    Config config_;
    std::ofstream logFile_;
    float lastDepth_;
    int captureCount_;
};

// ============================================================================
// VISUALIZATION
// ============================================================================

void drawBall(cv::Mat &frame, const Ball &ball, const DepthProjector & /*projector*/) {
    int drawRadius = static_cast<int>(std::max(5.0f, std::min(200.0f, ball.radius2D)));
    
    cv::Scalar color = ball.isPredicted ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);
    
    cv::circle(frame, ball.center2D, drawRadius, color, 3);
    cv::drawMarker(frame, ball.center2D, cv::Scalar(0, 0, 255), 
                   cv::MARKER_CROSS, 30, 3);
    
    std::stringstream ss;
    ss << (ball.isPredicted ? "PRED " : "BALL ") 
       << std::fixed << std::setprecision(2) << ball.position3D[2] << "m ("
       << int(ball.confidence * 100) << "%)";
    
    cv::Point labelPos(ball.center2D.x - 70, ball.center2D.y - drawRadius - 15);
    
    cv::Size textSize = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
    cv::rectangle(frame, 
                 cv::Point(labelPos.x - 5, labelPos.y - textSize.height - 5),
                 cv::Point(labelPos.x + textSize.width + 5, labelPos.y + 5),
                 cv::Scalar(0, 0, 0), -1);
    
    cv::putText(frame, ss.str(), labelPos, cv::FONT_HERSHEY_SIMPLEX, 
                0.6, color, 2);
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

int main(int argc, char **argv) try {
    // Configuration
    Config config;
    
    // Parse command line arguments or find model
    if (argc > 1) {
        config.modelPath = argv[1];
    } else {
        // Try multiple locations for the ONNX model
        std::vector<std::string> searchPaths = {
            "./best.onnx",
            "../best.onnx",
            "../../best.onnx",
            "../../examples/cpp/Sample-ColorDepthFallback/best.onnx",
            "../../../examples/cpp/Sample-ColorDepthFallback/best.onnx",
            "./runs/train/red_ball_v1/weights/best.onnx",
            "../runs/train/red_ball_v1/weights/best.onnx",
            "../../runs/train/red_ball_v1/weights/best.onnx"
        };
        
        bool found = false;
        for (const auto &path : searchPaths) {
            std::ifstream testFile(path);
            if (testFile.good()) {
                config.modelPath = path;
                found = true;
                break;
            }
        }
        
        if (!found) {
            std::cerr << "\n❌ Error: ONNX model not found!\n" << std::endl;
            std::cerr << "Searched in:" << std::endl;
            for (const auto &path : searchPaths) {
                std::cerr << "  - " << path << std::endl;
            }
            std::cerr << "\nPlease export your model using:\n";
            std::cerr << "  python3 export_yolo_to_onnx.py\n" << std::endl;
            return -1;
        }
    }
    
    // Get model file size
    std::ifstream modelFile(config.modelPath, std::ios::binary | std::ios::ate);
    float modelSizeMB = modelFile.tellg() / (1024.0f * 1024.0f);
    modelFile.close();
    
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        🎯 YOLO BALL DETECTION & TRACKING SYSTEM                ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Model Path: " << std::setw(50) << std::left << config.modelPath.substr(0, 50) << "║" << std::endl;
    std::cout << "║  Model Size: " << std::fixed << std::setprecision(1) << modelSizeMB << " MB                                            ║" << std::endl;
    std::cout << "║  Confidence Threshold: " << config.confThreshold << "                                ║" << std::endl;
    std::cout << "║  Capture Distance: " << config.captureThreshold << "m                                     ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;
    
    // Initialize Orbbec
    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);
    ob::Pipeline pipe;
    auto pipeConfig = std::make_shared<ob::Config>();
    pipeConfig->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_DISABLE);

    // Setup depth stream
    std::shared_ptr<ob::VideoStreamProfile> depthProfile;
    auto depthProfiles = pipe.getStreamProfileList(OB_SENSOR_DEPTH);
    
    if (depthProfiles && depthProfiles->count() > 0) {
        for (uint32_t i = 0; i < depthProfiles->count(); ++i) {
            auto profile = depthProfiles->getProfile(i);
            if (profile && profile->is<ob::VideoStreamProfile>()) {
                auto videoProfile = profile->as<ob::VideoStreamProfile>();
                if (videoProfile->width() == 640) {
                    depthProfile = videoProfile;
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
        throw std::runtime_error("No depth profile available");
    }

    pipe.start(pipeConfig);

    // Get intrinsics
    OBCameraIntrinsic intrinsic{};
    try {
        intrinsic = depthProfile->getIntrinsic();
        if (intrinsic.fx <= 0.0f) {
            intrinsic = createDefaultIntrinsics(depthProfile->width(), depthProfile->height());
            std::cout << "⚠️  Using default intrinsics" << std::endl;
        }
    } catch (...) {
        intrinsic = createDefaultIntrinsics(depthProfile->width(), depthProfile->height());
        std::cout << "⚠️  Using default intrinsics" << std::endl;
    }

    printIntrinsics(intrinsic);

    // Initialize components
    Window depthWindow("Depth", depthProfile->width(), depthProfile->height());
    std::unique_ptr<DepthProjector> projector;
    
    YOLODetector detector(config);
    BallTracker tracker(config);
    CaptureManager captureManager(config);

    // Open color camera
    cv::VideoCapture colorCapture(0, cv::CAP_AVFOUNDATION);
    if (!colorCapture.isOpened()) {
        throw std::runtime_error("Failed to open color camera");
    }

    std::cout << "✅ System initialized. Press 'Q' or ESC to quit.\n" << std::endl;

    // Main loop
    uint64_t frameCounter = 0;
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (depthWindow) {
        auto frameSet = pipe.waitForFrames(100);
        if (!frameSet) continue;

        auto depthFrame = frameSet->depthFrame();
        if (!depthFrame) continue;

        depthWindow.addToRender(depthFrame);

        const uint32_t width = depthFrame->width();
        const uint32_t height = depthFrame->height();
        const uint16_t *data = reinterpret_cast<const uint16_t *>(depthFrame->data());
        const float valueScale = depthFrame->getValueScale();

        cv::Mat colorFrame;
        if (!colorCapture.read(colorFrame)) continue;

        if (data && valueScale > 0.0f) {
            // Initialize projector
            if (!projector || std::fabs(projector->valueScale() - valueScale) > 1e-6f) {
                projector = std::make_unique<DepthProjector>(intrinsic, valueScale);
            }

            // Calculate delta time
            auto currentTime = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(currentTime - lastTime).count();
            lastTime = currentTime;

            // Detect and track
            auto detections = detector.detect(colorFrame, data, width, height, *projector);
            tracker.update(detections, dt);

            // Resize color for visualization
            cv::Mat vis;
            if (colorFrame.cols != static_cast<int>(width) || 
                colorFrame.rows != static_cast<int>(height)) {
                cv::resize(colorFrame, vis, cv::Size(width, height));
            } else {
                vis = colorFrame.clone();
            }

            // Handle tracking
            if (tracker.hasTrack()) {
                const Ball &tracked = tracker.getTrackedBall();
                
                // Check for 2m capture
                if (captureManager.shouldCapture(tracked.position3D[2])) {
                    captureManager.capture(tracked, vis);
                }
                
                // Draw tracked ball
                drawBall(vis, tracked, *projector);
                
                // Show distance
                std::stringstream distSS;
                distSS << "Distance: " << std::fixed << std::setprecision(2) 
                       << tracked.position3D[2] << "m";
                cv::putText(vis, distSS.str(), cv::Point(10, height - 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
                
                // Approaching warning
                if (tracked.position3D[2] < 2.2f && tracked.position3D[2] > 1.8f) {
                    cv::putText(vis, ">>> APPROACHING 2m <<<", 
                               cv::Point(width/2 - 150, 50),
                               cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                               cv::Scalar(0, 165, 255), 3);
                }
                
                // Status every 10 frames
                if (frameCounter % 10 == 0) {
                    std::cout << "Tracking | Dist: " << std::fixed << std::setprecision(2) 
                              << tracked.position3D[2] << "m | Conf: " 
                              << int(tracked.confidence*100) << "% | " 
                              << (tracked.isPredicted ? "PRED" : "MEAS") << std::endl;
                }
            } else {
                // Draw detections (no tracking yet)
                for (const auto &det : detections) {
                    drawBall(vis, det, *projector);
                }
            }

            cv::imshow("YOLO Ball Detection & Tracking", vis);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') break;
        }

        ++frameCounter;
    }

    colorCapture.release();
    pipe.stop();
    
    std::cout << "\n✅ System shutdown complete." << std::endl;
    std::cout << "📊 Total frames processed: " << frameCounter << std::endl;
    
    return 0;

} catch (const std::exception &e) {
    std::cerr << "\n❌ Error: " << e.what() << std::endl;
    return -1;
}