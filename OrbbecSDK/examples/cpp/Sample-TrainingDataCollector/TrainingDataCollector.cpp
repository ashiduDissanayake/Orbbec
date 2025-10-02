// TrainingDataCollector.cpp - Collect YOLO Training Images from Orbbec
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

int main() {
    // Create output directory
    system("mkdir -p ./training_data");
    
    // Open Orbbec color camera (same as your code)
    cv::VideoCapture colorCapture;
    colorCapture.open(0, cv::CAP_AVFOUNDATION);
    
    if (!colorCapture.isOpened()) {
        std::cerr << "❌ Failed to open color camera!" << std::endl;
        return -1;
    }
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       RED BALL TRAINING DATA COLLECTOR (Orbbec)              ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Controls:                                                    ║" << std::endl;
    std::cout << "║    SPACE    - Capture image                                  ║" << std::endl;
    std::cout << "║    A        - Enable auto-capture (1 img/sec)                ║" << std::endl;
    std::cout << "║    S        - Stop auto-capture                              ║" << std::endl;
    std::cout << "║    Q / ESC  - Quit                                           ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Collection Tips:                                             ║" << std::endl;
    std::cout << "║    • Capture ball at different distances (0.5m - 4m)         ║" << std::endl;
    std::cout << "║    • Vary angles (left, right, center, high, low)            ║" << std::endl;
    std::cout << "║    • Change lighting (bright, dim, shadows)                  ║" << std::endl;
    std::cout << "║    • Different backgrounds (table, floor, wall)              ║" << std::endl;
    std::cout << "║    • Include partial occlusions (hand holding, etc)          ║" << std::endl;
    std::cout << "║    • Target: 200-300 images                                  ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n" << std::endl;
    
    int imageCount = 0;
    bool autoCapture = false;
    auto lastCaptureTime = std::chrono::steady_clock::now();
    auto flashUntil = std::chrono::steady_clock::now();
    
    cv::Mat frame;
    
    while (true) {
        // Read frame from Orbbec color camera
        if (!colorCapture.read(frame)) {
            std::cerr << "Failed to read frame!" << std::endl;
            break;
        }
        
        cv::Mat display = frame.clone();
        auto now = std::chrono::steady_clock::now();
        
        // Add overlay info
        cv::putText(display, "Images: " + std::to_string(imageCount) + "/300", 
                   cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1.2, 
                   cv::Scalar(0, 255, 0), 3);
        
        std::string modeText = autoCapture ? "AUTO-CAPTURE ON" : "MANUAL MODE";
        cv::Scalar modeColor = autoCapture ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 255, 255);
        cv::putText(display, modeText, cv::Point(10, 85), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.9, modeColor, 2);
        
        // Flash effect after capture
        if (now < flashUntil) {
            cv::rectangle(display, cv::Point(0, 0), 
                         cv::Point(display.cols, display.rows),
                         cv::Scalar(0, 255, 0), 30);
        }
        
        // Auto-capture mode
        if (autoCapture) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - lastCaptureTime).count();
            
            if (elapsed > 1000) {  // 1 second interval
                // Generate filename
                auto timestamp = std::chrono::system_clock::now();
                auto timestamp_c = std::chrono::system_clock::to_time_t(timestamp);
                std::stringstream filenameSS;
                filenameSS << "./training_data/ball_" 
                          << std::setfill('0') << std::setw(4) << imageCount 
                          << "_" << std::put_time(std::localtime(&timestamp_c), "%Y%m%d_%H%M%S")
                          << ".jpg";
                std::string filename = filenameSS.str();
                
                // Save image
                cv::imwrite(filename, frame);
                imageCount++;
                
                std::cout << "✅ Captured: " << filename 
                         << " (Total: " << imageCount << ")" << std::endl;
                
                lastCaptureTime = now;
                flashUntil = now + std::chrono::milliseconds(200);
            }
        }
        
        cv::imshow("Training Data Collection - Red Ball (Orbbec)", display);
        
        // Handle key presses
        int key = cv::waitKey(1) & 0xFF;
        
        if (key == 32) {  // SPACE - manual capture
            auto timestamp = std::chrono::system_clock::now();
            auto timestamp_c = std::chrono::system_clock::to_time_t(timestamp);
            std::stringstream filenameSS;
            filenameSS << "./training_data/ball_" 
                      << std::setfill('0') << std::setw(4) << imageCount 
                      << "_" << std::put_time(std::localtime(&timestamp_c), "%Y%m%d_%H%M%S")
                      << ".jpg";
            std::string filename = filenameSS.str();
            
            cv::imwrite(filename, frame);
            imageCount++;
            
            std::cout << "✅ Captured: " << filename 
                     << " (Total: " << imageCount << ")" << std::endl;
            
            flashUntil = now + std::chrono::milliseconds(200);
        }
        else if (key == 'a' || key == 'A') {  // Enable auto-capture
            autoCapture = true;
            lastCaptureTime = now;
            std::cout << "🔄 Auto-capture enabled (1 image/second)" << std::endl;
        }
        else if (key == 's' || key == 'S') {  // Stop auto-capture
            autoCapture = false;
            std::cout << "⏸️  Auto-capture stopped" << std::endl;
        }
        else if (key == 'q' || key == 'Q' || key == 27) {  // Quit
            break;
        }
    }
    
    colorCapture.release();
    cv::destroyAllWindows();
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Collection Complete: " << imageCount << " images captured                  " << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Saved to: ./training_data/                                   ║" << std::endl;
    std::cout << "║  Next step: Annotate images using Roboflow                   ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n" << std::endl;
    
    return 0;
}