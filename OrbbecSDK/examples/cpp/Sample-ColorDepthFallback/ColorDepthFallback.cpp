// ColorDepthFallback.cpp - 2M CAPTURE SYSTEM
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
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
// UTILITY FUNCTIONS
// ============================================================================

void printIntrinsics(const OBCameraIntrinsic &intrinsic) {
  std::cout << "\n=== Depth Intrinsics ===" << std::endl;
  std::cout << "  fx: " << intrinsic.fx << std::endl;
  std::cout << "  fy: " << intrinsic.fy << std::endl;
  std::cout << "  cx: " << intrinsic.cx << std::endl;
  std::cout << "  cy: " << intrinsic.cy << std::endl;
  std::cout << "  Width: " << intrinsic.width << std::endl;
  std::cout << "  Height: " << intrinsic.height << std::endl;
  std::cout << "========================\n" << std::endl;
}

OBCameraIntrinsic createDefaultIntrinsics(uint32_t width, uint32_t height) {
  OBCameraIntrinsic intrinsic{};
  intrinsic.width = static_cast<int>(width);
  intrinsic.height = static_cast<int>(height);
  const float fovDegrees = 60.0f;
  const float fovRadians = fovDegrees * static_cast<float>(M_PI) / 180.0f;
  const float focal =
      static_cast<float>(width) / (2.0f * std::tan(fovRadians / 2.0f));
  intrinsic.fx = focal;
  intrinsic.fy = focal;
  intrinsic.cx = static_cast<float>(width) / 2.0f;
  intrinsic.cy = static_cast<float>(height) / 2.0f;
  return intrinsic;
}

// ============================================================================
// DEPTH PROJECTOR CLASS
// ============================================================================

class DepthProjector {
 public:
  DepthProjector(const OBCameraIntrinsic &intr, float valueScale)
      : intrinsic_(intr), valueScale_(valueScale) {}

  std::array<float, 3> project(uint16_t depthRaw, float pixelX, float pixelY) const {
    const float z = depthRaw * valueScale_ * 0.01f;  // cm → meters
    if (z <= 0.0f || intrinsic_.fx <= 0.0f || intrinsic_.fy <= 0.0f) {
      return {0.f, 0.f, 0.f};
    }
    const float x = (pixelX - intrinsic_.cx) * z / intrinsic_.fx;
    const float y = (pixelY - intrinsic_.cy) * z / intrinsic_.fy;
    return {x, y, z};
  }

  float valueScale() const { return valueScale_; }
  const OBCameraIntrinsic &intrinsic() const { return intrinsic_; }

 private:
  OBCameraIntrinsic intrinsic_{};
  float valueScale_ = 0.f;
};

// ============================================================================
// RED BALL DETECTOR
// ============================================================================

class RedBallDetector {
 public:
  struct Ball {
    cv::Point2f center2D;
    std::array<float, 3> center3D;
    float radius2D;
    float radius3D;
    float depth;
    float redScore;
    float confidence;
  };

  std::vector<Ball> detectBalls(
      const uint16_t *depthData,
      uint32_t width, uint32_t height,
      const cv::Mat &colorFrame,
      const DepthProjector &projector) {
    
    if (!depthData || colorFrame.empty()) return {};

    cv::Mat resizedColor;
    if (colorFrame.cols != static_cast<int>(width) || 
        colorFrame.rows != static_cast<int>(height)) {
      cv::resize(colorFrame, resizedColor, cv::Size(width, height));
    } else {
      resizedColor = colorFrame;
    }

    cv::Mat redMask = detectRedRegions(resizedColor);
    
    cv::Mat redGray;
    redMask.copyTo(redGray);
    cv::GaussianBlur(redGray, redGray, cv::Size(9, 9), 2, 2);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(redGray, circles, cv::HOUGH_GRADIENT, 1,
                     50, 100, 20, 15, 150);

    std::vector<Ball> balls;

    for (const auto &circle : circles) {
      float cx = circle[0];
      float cy = circle[1];
      float radius = circle[2];

      int ix = static_cast<int>(cx);
      int iy = static_cast<int>(cy);
      
      if (ix < 0 || ix >= static_cast<int>(width) || 
          iy < 0 || iy >= static_cast<int>(height)) continue;

      uint16_t rawDepth = depthData[iy * width + ix];
      
      if (rawDepth == 0) {
        int samples = 0;
        int totalDepth = 0;
        
        for (int angle = 0; angle < 360; angle += 30) {
          float rad = angle * M_PI / 180.0f;
          int sx = static_cast<int>(cx + radius * 0.7f * cos(rad));
          int sy = static_cast<int>(cy + radius * 0.7f * sin(rad));
          
          if (sx >= 0 && sx < static_cast<int>(width) && 
              sy >= 0 && sy < static_cast<int>(height)) {
            uint16_t sampleDepth = depthData[sy * width + sx];
            if (sampleDepth > 0) {
              totalDepth += sampleDepth;
              samples++;
            }
          }
        }
        
        if (samples > 0) {
          rawDepth = totalDepth / samples;
        } else {
          continue;
        }
      }

      float depth = rawDepth * 0.01f;

      if (depth < 0.5f || depth > 4.0f) continue;

      const auto &intr = projector.intrinsic();
      float radius3D = (radius * depth) / intr.fx;

      if (radius3D < 0.07f || radius3D > 0.18f) continue;

      cv::Rect roi(std::max(0, static_cast<int>(cx - radius)),
                   std::max(0, static_cast<int>(cy - radius)),
                   std::min(static_cast<int>(radius * 2), 
                           static_cast<int>(width) - std::max(0, static_cast<int>(cx - radius))),
                   std::min(static_cast<int>(radius * 2), 
                           static_cast<int>(height) - std::max(0, static_cast<int>(cy - radius))));
      
      if (roi.width <= 0 || roi.height <= 0) continue;

      float redScore = calculateRedScore(resizedColor, roi);

      if (redScore < 0.25f) continue;

      auto pos3D = projector.project(rawDepth, cx, cy);

      Ball ball;
      ball.center2D = cv::Point2f(cx, cy);
      ball.center3D = pos3D;
      ball.radius2D = radius;
      ball.radius3D = radius3D;
      ball.depth = depth;
      ball.redScore = redScore;
      ball.confidence = redScore;

      balls.push_back(ball);
    }

    std::sort(balls.begin(), balls.end(),
              [](const Ball &a, const Ball &b) { return a.confidence > b.confidence; });

    return balls;
  }

  cv::Mat visualize(const cv::Mat &colorFrame, uint32_t width, uint32_t height,
                   const std::vector<Ball> &balls) {
    
    cv::Mat vis;
    if (colorFrame.cols != static_cast<int>(width) || 
        colorFrame.rows != static_cast<int>(height)) {
      cv::resize(colorFrame, vis, cv::Size(width, height));
    } else {
      vis = colorFrame.clone();
    }

    for (size_t i = 0; i < balls.size(); ++i) {
      const auto &ball = balls[i];

      int drawRadius = static_cast<int>(std::max(5.0f, std::min(200.0f, ball.radius2D)));

      cv::circle(vis, ball.center2D, drawRadius, cv::Scalar(0, 255, 0), 3);
      cv::drawMarker(vis, ball.center2D, cv::Scalar(0, 0, 255), 
                     cv::MARKER_CROSS, 30, 3);

      std::stringstream ss;
      ss << "BALL #" << (i+1) << " " << std::fixed << std::setprecision(2) 
         << ball.depth << "m";
      
      cv::Point labelPos(ball.center2D.x - 60, ball.center2D.y - drawRadius - 15);
      cv::putText(vis, ss.str(), labelPos, cv::FONT_HERSHEY_SIMPLEX, 
                  0.6, cv::Scalar(0, 255, 0), 2);
    }

    return vis;
  }

 private:
  cv::Mat detectRedRegions(const cv::Mat &colorFrame) {
    cv::Mat hsv;
    cv::cvtColor(colorFrame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask1, mask2;
    cv::inRange(hsv, cv::Scalar(0, 50, 50), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(170, 50, 50), cv::Scalar(180, 255, 255), mask2);
    
    cv::Mat redMask = mask1 | mask2;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(redMask, redMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(redMask, redMask, cv::MORPH_CLOSE, kernel);

    return redMask;
  }

  float calculateRedScore(const cv::Mat &colorFrame, const cv::Rect &roi) {
    if (colorFrame.empty() || roi.width <= 0 || roi.height <= 0) return 0.0f;

    cv::Rect safeROI = roi & cv::Rect(0, 0, colorFrame.cols, colorFrame.rows);
    if (safeROI.width == 0 || safeROI.height == 0) return 0.0f;

    cv::Mat roiImg = colorFrame(safeROI);
    cv::Mat hsv;
    cv::cvtColor(roiImg, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask1, mask2;
    cv::inRange(hsv, cv::Scalar(0, 50, 50), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(170, 50, 50), cv::Scalar(180, 255, 255), mask2);
    cv::Mat redMask = mask1 | mask2;

    int redPixels = cv::countNonZero(redMask);
    int totalPixels = safeROI.width * safeROI.height;

    return static_cast<float>(redPixels) / totalPixels;
  }
};

// ============================================================================
// BALL TRACKER
// ============================================================================

class BallTracker {
 public:
  struct TrackedBall {
    cv::Point2f center2D;
    std::array<float, 3> position3D;
    std::array<float, 3> velocity3D;
    float radius;
    int framesSinceLastSeen;
    int totalFramesSeen;
    float confidence;
    bool isPredicted;
  };

  void update(const std::vector<RedBallDetector::Ball> &detections, double dt = 0.033) {
    dt = std::max(0.001, std::min(0.1, dt));
    
    if (detections.empty()) {
      if (trackedBall_.totalFramesSeen > 0) {
        trackedBall_.framesSinceLastSeen++;
        
        if (trackedBall_.framesSinceLastSeen < 5) {
          float maxVel = 5.0f;
          for (int i = 0; i < 3; ++i) {
            trackedBall_.velocity3D[i] = std::max(-maxVel, std::min(maxVel, trackedBall_.velocity3D[i]));
          }
          
          trackedBall_.position3D[0] += trackedBall_.velocity3D[0] * dt;
          trackedBall_.position3D[1] += trackedBall_.velocity3D[1] * dt;
          trackedBall_.position3D[2] += trackedBall_.velocity3D[2] * dt;
          trackedBall_.position3D[2] = std::max(0.5f, std::min(5.0f, trackedBall_.position3D[2]));
          
          trackedBall_.isPredicted = true;
          trackedBall_.confidence *= 0.85f;
        } else {
          trackedBall_.totalFramesSeen = 0;
        }
      }
    } else {
      const auto &det = detections[0];
      
      if (trackedBall_.totalFramesSeen == 0) {
        trackedBall_.center2D = det.center2D;
        trackedBall_.position3D = det.center3D;
        trackedBall_.velocity3D = {0.f, 0.f, 0.f};
        trackedBall_.radius = det.radius3D;
        trackedBall_.framesSinceLastSeen = 0;
        trackedBall_.totalFramesSeen = 1;
        trackedBall_.confidence = det.confidence;
        trackedBall_.isPredicted = false;
      } else {
        float dx = det.center3D[0] - trackedBall_.position3D[0];
        float dy = det.center3D[1] - trackedBall_.position3D[1];
        float dz = det.center3D[2] - trackedBall_.position3D[2];
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (trackedBall_.isPredicted && dist > 0.5f) {
          trackedBall_.framesSinceLastSeen++;
          return;
        }
        
        if (dt > 0.001 && dt < 0.1 && !trackedBall_.isPredicted) {
          float newVelX = dx / dt;
          float newVelY = dy / dt;
          float newVelZ = dz / dt;
          
          float velAlpha = 0.3f;
          trackedBall_.velocity3D[0] = velAlpha * newVelX + (1-velAlpha) * trackedBall_.velocity3D[0];
          trackedBall_.velocity3D[1] = velAlpha * newVelY + (1-velAlpha) * trackedBall_.velocity3D[1];
          trackedBall_.velocity3D[2] = velAlpha * newVelZ + (1-velAlpha) * trackedBall_.velocity3D[2];
          
          float maxVel = 5.0f;
          for (int i = 0; i < 3; ++i) {
            trackedBall_.velocity3D[i] = std::max(-maxVel, std::min(maxVel, trackedBall_.velocity3D[i]));
          }
        }
        
        float alpha = 0.7f;
        trackedBall_.position3D[0] = alpha * det.center3D[0] + (1-alpha) * trackedBall_.position3D[0];
        trackedBall_.position3D[1] = alpha * det.center3D[1] + (1-alpha) * trackedBall_.position3D[1];
        trackedBall_.position3D[2] = alpha * det.center3D[2] + (1-alpha) * trackedBall_.position3D[2];
        trackedBall_.position3D[2] = std::max(0.5f, std::min(5.0f, trackedBall_.position3D[2]));
        
        trackedBall_.center2D = det.center2D;
        trackedBall_.radius = 0.7f * det.radius3D + 0.3f * trackedBall_.radius;
        trackedBall_.radius = std::max(0.08f, std::min(0.15f, trackedBall_.radius));
        trackedBall_.framesSinceLastSeen = 0;
        trackedBall_.totalFramesSeen++;
        trackedBall_.confidence = det.confidence;
        trackedBall_.isPredicted = false;
      }
    }
  }

  bool hasTrack() const {
    return trackedBall_.totalFramesSeen > 0 && trackedBall_.framesSinceLastSeen < 5;
  }

  const TrackedBall& getTrack() const {
    return trackedBall_;
  }

  void reset() {
    trackedBall_.totalFramesSeen = 0;
  }

 private:
  TrackedBall trackedBall_{};
};

// ============================================================================
// CAPTURE MANAGER - 2M THRESHOLD
// ============================================================================

class CaptureManager {
 public:
  CaptureManager(const std::string &outputDir = "./ball_captures") 
      : outputDir_(outputDir), lastDepth_(999.0f), captureCount_(0) {
    
    #ifdef _WIN32
    system(("mkdir " + outputDir_).c_str());
    #else
    system(("mkdir -p " + outputDir_).c_str());
    #endif
    
    logFile_.open(outputDir_ + "/positions.csv");
    logFile_ << "CaptureID,Timestamp,X_m,Y_m,Z_m,ImageFile\n";
    
    std::cout << "📂 Captures will be saved to: " << outputDir_ << "\n" << std::endl;
  }

  ~CaptureManager() {
    if (logFile_.is_open()) {
      logFile_.close();
    }
  }

  bool shouldCapture(float currentDepth) {
    bool crossed = (lastDepth_ > 2.05f) && (currentDepth <= 2.0f);
    lastDepth_ = currentDepth;
    return crossed;
  }

  void capture(const BallTracker::TrackedBall &track, const cv::Mat &frame) {
    
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
    std::string fullPath = outputDir_ + "/" + filename;
    
    cv::Mat captureFrame = frame.clone();
    drawInfo(captureFrame, track, captureCount_);
    
    cv::imwrite(fullPath, captureFrame);
    
    logFile_ << captureCount_ << ","
             << timestamp << ","
             << std::fixed << std::setprecision(3)
             << track.position3D[0] << ","
             << track.position3D[1] << ","
             << track.position3D[2] << ","
             << filename << "\n";
    logFile_.flush();
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║              📸 CAPTURE #" << captureCount_ << " at 2.0m THRESHOLD               ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Ball Position (relative to camera):                          ║" << std::endl;
    std::cout << "║    X = " << std::setw(7) << std::right << std::fixed << std::setprecision(3) 
              << track.position3D[0] << " m                                          ║" << std::endl;
    std::cout << "║    Y = " << std::setw(7) << track.position3D[1] << " m                                          ║" << std::endl;
    std::cout << "║    Z = " << std::setw(7) << track.position3D[2] << " m                                          ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  📁 Saved: " << std::setw(47) << std::left << filename << "║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n" << std::endl;
  }

  void reset() {
    lastDepth_ = 999.0f;
  }

 private:
  void drawInfo(cv::Mat &frame, const BallTracker::TrackedBall &track, int captureNum) {
    
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(350, 150), 
                 cv::Scalar(0, 0, 0), -1);
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(350, 150), 
                 cv::Scalar(0, 255, 255), 2);
    
    cv::putText(frame, "CAPTURE #" + std::to_string(captureNum), 
                cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                cv::Scalar(0, 255, 255), 2);
    
    std::stringstream posX, posY, posZ;
    posX << "X: " << std::fixed << std::setprecision(3) << track.position3D[0] << " m";
    posY << "Y: " << std::fixed << std::setprecision(3) << track.position3D[1] << " m";
    posZ << "Z: " << std::fixed << std::setprecision(3) << track.position3D[2] << " m";
    
    cv::putText(frame, posX.str(), cv::Point(20, 75), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(frame, posY.str(), cv::Point(20, 100), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(frame, posZ.str(), cv::Point(20, 125), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
  }

  std::string outputDir_;
  std::ofstream logFile_;
  float lastDepth_;
  int captureCount_;
};

// ============================================================================
// MAIN PROGRAM
// ============================================================================

int main(int /*argc*/, char ** /*argv*/) try {
  ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);

  ob::Pipeline pipe;
  auto config = std::make_shared<ob::Config>();
  config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_DISABLE);

  std::shared_ptr<ob::VideoStreamProfile> depthProfile;
  try {
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
      if (!depthProfile && depthProfiles->count() > 0) {
        depthProfile = depthProfiles->getProfile(0)->as<ob::VideoStreamProfile>();
      }
      if (depthProfile) config->enableStream(depthProfile);
    }
  } catch (ob::Error &e) {
    std::cerr << "Failed to query depth profiles: " << e.getMessage() << "\n";
  }

  if (!depthProfile) {
    std::cerr << "No depth profile found" << std::endl;
    return -1;
  }

  pipe.start(config);

  OBCameraIntrinsic intrinsic{};
  bool hasIntrinsic = false;

  try {
    intrinsic = depthProfile->getIntrinsic();
    hasIntrinsic = intrinsic.fx > 0.f && intrinsic.fy > 0.f;
    if (hasIntrinsic) std::cout << "✓ Intrinsics acquired" << std::endl;
  } catch (...) {}

  if (!hasIntrinsic) {
    intrinsic = createDefaultIntrinsics(depthProfile->width(), depthProfile->height());
    std::cout << "⚠ Using default intrinsics" << std::endl;
  }

  printIntrinsics(intrinsic);

  Window depthWindow("Depth", depthProfile->width(), depthProfile->height());
  std::unique_ptr<DepthProjector> projector;
  RedBallDetector detector;
  BallTracker tracker;

  cv::VideoCapture colorCapture;
  colorCapture.open(0, cv::CAP_AVFOUNDATION);
  if (!colorCapture.isOpened()) {
    std::cerr << "Failed to open color camera" << std::endl;
    return -1;
  }

  CaptureManager captureManager("./ball_captures");

  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║           📸 2M CAPTURE MODE ACTIVE                            ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║  Bring ball from far to near                                  ║" << std::endl;
  std::cout << "║  Auto-capture when ball reaches 2.0m                          ║" << std::endl;
  std::cout << "║  Position + snapshot saved to ./ball_captures/                ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

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
      if (!projector || std::fabs(projector->valueScale() - valueScale) > 1e-6f) {
        projector = std::make_unique<DepthProjector>(intrinsic, valueScale);
      }

      auto currentTime = std::chrono::high_resolution_clock::now();
      double dt = std::chrono::duration<double>(currentTime - lastTime).count();
      lastTime = currentTime;

      auto balls = detector.detectBalls(data, width, height, colorFrame, *projector);
      tracker.update(balls, dt);

      if (tracker.hasTrack()) {
        const auto &track = tracker.getTrack();
        
        if (captureManager.shouldCapture(track.position3D[2])) {
          cv::Mat vis;
          if (colorFrame.cols != static_cast<int>(width) || 
              colorFrame.rows != static_cast<int>(height)) {
            cv::resize(colorFrame, vis, cv::Size(width, height));
          } else {
            vis = colorFrame.clone();
          }
          
          int drawRadius = static_cast<int>(std::max(5.0f, std::min(200.0f, 
              track.radius * projector->intrinsic().fx / std::max(0.5f, track.position3D[2]))));
          cv::circle(vis, track.center2D, drawRadius, cv::Scalar(0, 255, 0), 3);
          cv::drawMarker(vis, track.center2D, cv::Scalar(0, 0, 255), 
                        cv::MARKER_CROSS, 30, 3);
          
          captureManager.capture(track, vis);
        }
        
        if (frameCounter % 10 == 0) {
          std::cout << "Tracking... Distance: " << std::fixed << std::setprecision(2) 
                    << track.position3D[2] << "m" << std::endl;
        }
      }

      cv::Mat vis;
      if (tracker.hasTrack()) {
        const auto &track = tracker.getTrack();
        
        float safeRadius2D = track.radius * projector->intrinsic().fx / std::max(0.5f, track.position3D[2]);
        safeRadius2D = std::max(5.0f, std::min(200.0f, safeRadius2D));
        
        std::vector<RedBallDetector::Ball> trackedBalls;
        RedBallDetector::Ball trackedBall;
        trackedBall.center2D = track.center2D;
        trackedBall.center3D = track.position3D;
        trackedBall.radius2D = safeRadius2D;
        trackedBall.radius3D = track.radius;
        trackedBall.depth = track.position3D[2];
        trackedBall.redScore = track.confidence;
        trackedBall.confidence = track.confidence;
        trackedBalls.push_back(trackedBall);
        
        vis = detector.visualize(colorFrame, width, height, trackedBalls);
        
        std::stringstream distSS;
        distSS << "Distance: " << std::fixed << std::setprecision(2) 
               << track.position3D[2] << "m";
        cv::putText(vis, distSS.str(), cv::Point(10, height - 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        if (track.position3D[2] < 2.2f && track.position3D[2] > 1.8f) {
          cv::putText(vis, ">>> APPROACHING 2m <<<", cv::Point(width/2 - 150, 50),
                     cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2);
        }
        
      } else {
        vis = detector.visualize(colorFrame, width, height, balls);
      }

      cv::imshow("Ball Detection & Tracking", vis);

      int key = cv::waitKey(1);
      if (key == 27 || key == 'q') break;
    }

    ++frameCounter;
  }

  colorCapture.release();
  pipe.stop();
  
  std::cout << "\n✅ Session complete! Check ./ball_captures/ folder for results." << std::endl;
  
  return 0;

} catch (ob::Error &e) {
  std::cerr << "Orbbec Error: " << e.getMessage() << std::endl;
  return -1;
} catch (cv::Exception &e) {
  std::cerr << "OpenCV Error: " << e.what() << std::endl;
  return -1;
} catch (std::exception &e) {
  std::cerr << "Error: " << e.what() << std::endl;
  return -1;
}