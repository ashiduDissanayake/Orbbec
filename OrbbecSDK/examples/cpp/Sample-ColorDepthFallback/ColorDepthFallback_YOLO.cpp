// ColorDepthFallback_YOLO_Simple.cpp - Simple YOLO Ball Detection (No Tracking)
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
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
    std::string modelPath = "./best.onnx";
    float confThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    
    // Ball physical constraints
    float minRadius3D = 0.03f;        // 3cm minimum (relaxed)
    float maxRadius3D = 0.30f;        // 30cm maximum
    float minDepth = 0.2f;            // 20cm minimum distance
    float maxDepth = 5.0f;            // 5m maximum distance
    
    // Capture
    float captureThreshold = 2.0f;
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
        // Convert depth units to meters (cm → m)
        const float z = depthRaw * valueScale_ * 0.01f;
        
        if (z <= 0.0f || intrinsic_.fx <= 0.0f || intrinsic_.fy <= 0.0f) {
            return {0.f, 0.f, 0.f};
        }
        
        const float x = (pixelX - intrinsic_.cx) * z / intrinsic_.fx;
        const float y = (pixelY - intrinsic_.cy) * z / intrinsic_.fy;
        
        return {x, y, z};
    }

    const OBCameraIntrinsic& intrinsic() const { return intrinsic_; }

private:
    OBCameraIntrinsic intrinsic_;
    float valueScale_;
};

// ============================================================================
// BALL STRUCTURE
// ============================================================================

struct Ball {
    cv::Point2f center2D;      // Center in depth image coordinates
    std::array<float, 3> position3D;  // 3D position (X, Y, Z) in meters
    float radius2D;            // Radius in pixels
    float radius3D;            // Radius in meters
    float confidence;          // YOLO confidence score
};

// ============================================================================
// YOLO DETECTOR
// ============================================================================

class YOLODetector {
public:
    YOLODetector(const Config &config) : config_(config) {
        std::cout << "🤖 Loading YOLO model: " << config_.modelPath << std::endl;
        
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
                             uint32_t depthWidth, uint32_t depthHeight,
                             const DepthProjector &projector) {
        
        std::vector<Ball> balls;
        
        if (!modelLoaded_ || colorFrame.empty() || !depthData) {
            return balls;
        }

        // Prepare YOLO input
        cv::Mat blob = cv::dnn::blobFromImage(
            colorFrame, 1.0 / 255.0, cv::Size(640, 640),
            cv::Scalar(0, 0, 0), true, false
        );

        // Run YOLO inference
        net_.setInput(blob);
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        if (outputs.empty() || outputs[0].empty()) {
            return balls;
        }

        // Parse YOLOv8 output
        cv::Mat output = outputs[0];
        if (output.dims == 3 && output.size[0] == 1) {
            output = output.reshape(1, output.size[1]);
        }
        cv::Mat transposed;
        cv::transpose(output, transposed);

        float scaleX = static_cast<float>(colorFrame.cols) / 640.0f;
        float scaleY = static_cast<float>(colorFrame.rows) / 640.0f;

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
                w = std::min(static_cast<float>(colorFrame.cols - x), w);
                h = std::min(static_cast<float>(colorFrame.rows - y), h);

                boxes.push_back(cv::Rect(x, y, static_cast<int>(w), static_cast<int>(h)));
                confidences.push_back(confidence);
            }
        }

        // Apply NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, config_.confThreshold, config_.nmsThreshold, indices);

        // ✅ NEW: Filter by RED color AND create Ball objects
        for (int idx : indices) {
            cv::Rect box = boxes[idx];
            float conf = confidences[idx];
            
            // ✅ CRITICAL: Validate it's actually RED
            if (!isRedObject(colorFrame, box)) {
                continue;  // Skip false positives!
            }
            
            Ball ball = createBallFrom2D(box, conf, 
                                        colorFrame, depthData, 
                                        depthWidth, depthHeight, projector);
            
            if (ball.position3D[2] > 0.0f) {
                balls.push_back(ball);
            }
        }

        // Sort by confidence
        std::sort(balls.begin(), balls.end(),
                  [](const Ball &a, const Ball &b) { return a.confidence > b.confidence; });

        return balls;
    }

private:
    // ✅ NEW: Check if detected region is actually RED
    bool isRedObject(const cv::Mat &colorFrame, const cv::Rect &box) {
        cv::Rect safeBox = box & cv::Rect(0, 0, colorFrame.cols, colorFrame.rows);
        if (safeBox.width == 0 || safeBox.height == 0) {
            return false;
        }
        
        cv::Mat roi = colorFrame(safeBox);
        
        // Convert to HSV
        cv::Mat hsv;
        cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
        
        // Red color masks (red wraps around in HSV: 0-10 and 160-180)
        cv::Mat mask1, mask2;
        cv::inRange(hsv, cv::Scalar(0, 80, 80), cv::Scalar(10, 255, 255), mask1);
        cv::inRange(hsv, cv::Scalar(160, 80, 80), cv::Scalar(180, 255, 255), mask2);
        cv::Mat redMask = mask1 | mask2;
        
        // Calculate red pixel ratio
        int redPixels = cv::countNonZero(redMask);
        int totalPixels = safeBox.width * safeBox.height;
        float redRatio = (float)redPixels / totalPixels;
        
        // Require at least 15% red pixels
        return redRatio > 0.15f;
    }

    Ball createBallFrom2D(const cv::Rect &colorBox, float confidence,
                          const cv::Mat &colorFrame,
                          const uint16_t *depthData, 
                          uint32_t depthWidth, uint32_t depthHeight,
                          const DepthProjector &projector) {
        
        Ball ball{};
        ball.confidence = confidence;

        // Map to depth resolution
        float scaleX = static_cast<float>(depthWidth) / colorFrame.cols;
        float scaleY = static_cast<float>(depthHeight) / colorFrame.rows;

        cv::Rect depthBox;
        depthBox.x = static_cast<int>(colorBox.x * scaleX);
        depthBox.y = static_cast<int>(colorBox.y * scaleY);
        depthBox.width = static_cast<int>(colorBox.width * scaleX);
        depthBox.height = static_cast<int>(colorBox.height * scaleY);

        // Clamp
        depthBox.x = std::max(0, std::min(depthBox.x, static_cast<int>(depthWidth) - 1));
        depthBox.y = std::max(0, std::min(depthBox.y, static_cast<int>(depthHeight) - 1));
        depthBox.width = std::min(depthBox.width, static_cast<int>(depthWidth) - depthBox.x);
        depthBox.height = std::min(depthBox.height, static_cast<int>(depthHeight) - depthBox.y);

        float centerX = depthBox.x + depthBox.width / 2.0f;
        float centerY = depthBox.y + depthBox.height / 2.0f;
        float radius = std::max(depthBox.width, depthBox.height) / 2.0f;

        ball.center2D = cv::Point2f(centerX, centerY);
        ball.radius2D = radius;

        // Sample depth
        uint16_t rawDepth = sampleDepth(centerX, centerY, radius, depthData, depthWidth, depthHeight);

        if (rawDepth > 50) {
            ball.position3D = projector.project(rawDepth, centerX, centerY);
            
            const auto &intr = projector.intrinsic();
            ball.radius3D = (radius * ball.position3D[2]) / intr.fx;
            
            // ✅ RELAXED validation - just check reasonable ranges
            bool validDepth = (ball.position3D[2] >= 0.1f && ball.position3D[2] <= 5.0f);
            bool validRadius = (ball.radius3D >= 0.01f && ball.radius3D <= 0.50f);
            
            if (!validDepth || !validRadius) {
                ball.position3D[2] = 0.0f;
            }
        }

        return ball;
    }

    uint16_t sampleDepth(float centerX, float centerY, float radius,
                        const uint16_t *depthData, uint32_t width, uint32_t height) {
        
        int ix = static_cast<int>(centerX);
        int iy = static_cast<int>(centerY);
        
        if (ix < 0 || ix >= static_cast<int>(width) || 
            iy < 0 || iy >= static_cast<int>(height)) {
            return 0;
        }

        // Try center
        uint16_t centerDepth = depthData[iy * width + ix];
        if (centerDepth > 50) {
            return centerDepth;
        }

        // Sample circle
        std::vector<uint16_t> samples;
        for (int angle = 0; angle < 360; angle += 30) {
            float rad = angle * M_PI / 180.0f;
            int sx = static_cast<int>(centerX + radius * 0.7f * cos(rad));
            int sy = static_cast<int>(centerY + radius * 0.7f * sin(rad));
            
            if (sx >= 0 && sx < static_cast<int>(width) && 
                sy >= 0 && sy < static_cast<int>(height)) {
                uint16_t sample = depthData[sy * width + sx];
                if (sample > 50) {
                    samples.push_back(sample);
                }
            }
        }
        
        if (samples.empty()) {
            return 0;
        }

        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    }

    Config config_;
    cv::dnn::Net net_;
    bool modelLoaded_ = false;
};
// ============================================================================
// CAPTURE MANAGER
// ============================================================================

class CaptureManager {
public:
    CaptureManager(const Config &config) : config_(config), captureCount_(0) {
        lastDepth_ = 999.0f;
        
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
        // Trigger when crossing 2m threshold (from far to near)
        bool crossed = (lastDepth_ > config_.captureThreshold + 0.05f) && 
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
        
        cv::imwrite(fullPath, frame);
        
        logFile_ << captureCount_ << ","
                 << timestamp << ","
                 << std::fixed << std::setprecision(3)
                 << ball.position3D[0] << ","
                 << ball.position3D[1] << ","
                 << ball.position3D[2] << ","
                 << ball.confidence << ","
                 << filename << "\n";
        logFile_.flush();
        
        std::cout << "\n📸 CAPTURE #" << captureCount_ << " at " << ball.position3D[2] << "m" << std::endl;
    }

private:
    Config config_;
    std::ofstream logFile_;
    float lastDepth_;
    int captureCount_;
};

// ============================================================================
// VISUALIZATION
// ============================================================================

void drawBalls(cv::Mat &frame, const std::vector<Ball> &balls, 
               uint32_t depthWidth, uint32_t depthHeight) {
    
    // Scale from depth coordinates to display resolution
    float scaleX = static_cast<float>(frame.cols) / depthWidth;
    float scaleY = static_cast<float>(frame.rows) / depthHeight;
    
    for (size_t i = 0; i < balls.size(); ++i) {
        const Ball &ball = balls[i];
        
        cv::Point2f displayCenter(ball.center2D.x * scaleX, ball.center2D.y * scaleY);
        int displayRadius = static_cast<int>(ball.radius2D * std::max(scaleX, scaleY));
        displayRadius = std::max(10, std::min(200, displayRadius));
        
        // Draw circle
        cv::circle(frame, displayCenter, displayRadius, cv::Scalar(0, 255, 0), 3);
        
        // Draw center cross
        cv::drawMarker(frame, displayCenter, cv::Scalar(0, 0, 255), 
                       cv::MARKER_CROSS, 30, 3);
        
        // Draw label
        std::stringstream ss;
        ss << "BALL #" << (i+1) << " " 
           << std::fixed << std::setprecision(2) << ball.position3D[2] << "m ("
           << int(ball.confidence * 100) << "%)";
        
        cv::Point labelPos(displayCenter.x - 80, displayCenter.y - displayRadius - 15);
        
        // Background box for text
        cv::Size textSize = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::rectangle(frame, 
                     cv::Point(labelPos.x - 5, labelPos.y - textSize.height - 5),
                     cv::Point(labelPos.x + textSize.width + 5, labelPos.y + 5),
                     cv::Scalar(0, 0, 0), -1);
        
        cv::putText(frame, ss.str(), labelPos, cv::FONT_HERSHEY_SIMPLEX, 
                    0.6, cv::Scalar(0, 255, 0), 2);
    }
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

int main(int argc, char **argv) try {
    Config config;
    
    if (argc > 1) {
        config.modelPath = argv[1];
    }
    
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        🎯 YOLO BALL DETECTION SYSTEM (SIMPLE)                 ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  • Detects ball with YOLO                                     ║" << std::endl;
    std::cout << "║  • Measures distance with depth sensor                        ║" << std::endl;
    std::cout << "║  • Captures at 2.0m threshold                                 ║" << std::endl;
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
        }
    } catch (...) {
        intrinsic = createDefaultIntrinsics(depthProfile->width(), depthProfile->height());
    }

    printIntrinsics(intrinsic);

    // Initialize components
    Window depthWindow("Depth", depthProfile->width(), depthProfile->height());
    std::unique_ptr<DepthProjector> projector;
    
    YOLODetector detector(config);
    CaptureManager captureManager(config);

    // Open color camera
    cv::VideoCapture colorCapture(0, cv::CAP_AVFOUNDATION);
    if (!colorCapture.isOpened()) {
        throw std::runtime_error("Failed to open color camera");
    }

    std::cout << "✅ System ready. Press 'Q' or ESC to quit.\n" << std::endl;

    uint64_t frameCounter = 0;

    while (depthWindow) {
        auto frameSet = pipe.waitForFrames(100);
        if (!frameSet) continue;

        auto depthFrame = frameSet->depthFrame();
        if (!depthFrame) continue;

        depthWindow.addToRender(depthFrame);

        const uint32_t depthWidth = depthFrame->width();
        const uint32_t depthHeight = depthFrame->height();
        const uint16_t *depthData = reinterpret_cast<const uint16_t *>(depthFrame->data());
        const float valueScale = depthFrame->getValueScale();

        cv::Mat colorFrame;
        if (!colorCapture.read(colorFrame)) continue;

        if (depthData && valueScale > 0.0f) {
            if (!projector) {
                projector = std::make_unique<DepthProjector>(intrinsic, valueScale);
            }

            // DETECT BALLS
            auto balls = detector.detect(colorFrame, depthData, depthWidth, depthHeight, *projector);

            // Visualize
            cv::Mat vis = colorFrame.clone();
            drawBalls(vis, balls, depthWidth, depthHeight);

            // Print detections
            if (!balls.empty() && frameCounter % 10 == 0) {
                for (size_t i = 0; i < balls.size(); ++i) {
                    std::cout << "Ball #" << (i+1) << ": "
                              << std::fixed << std::setprecision(2) << balls[i].position3D[2] << "m ("
                              << int(balls[i].confidence * 100) << "%)" << std::endl;
                }
            }

            // Check for 2m capture (only for closest ball)
            if (!balls.empty()) {
                if (captureManager.shouldCapture(balls[0].position3D[2])) {
                    captureManager.capture(balls[0], vis);
                }
                
                // Show distance
                std::stringstream distSS;
                distSS << "Distance: " << std::fixed << std::setprecision(2) 
                       << balls[0].position3D[2] << "m";
                cv::putText(vis, distSS.str(), cv::Point(10, vis.rows - 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255), 2);
                
                // Approaching warning
                if (balls[0].position3D[2] < 2.2f && balls[0].position3D[2] > 1.8f) {
                    cv::putText(vis, ">>> APPROACHING 2m <<<", 
                               cv::Point(vis.cols/2 - 180, 50),
                               cv::FONT_HERSHEY_SIMPLEX, 1.2, 
                               cv::Scalar(0, 165, 255), 3);
                }
            }

            cv::imshow("YOLO Ball Detection", vis);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') break;
        }

        ++frameCounter;
    }

    colorCapture.release();
    pipe.stop();
    
    std::cout << "\n✅ System shutdown." << std::endl;
    
    return 0;

} catch (const std::exception &e) {
    std::cerr << "\n❌ Error: " << e.what() << std::endl;
    return -1;
}