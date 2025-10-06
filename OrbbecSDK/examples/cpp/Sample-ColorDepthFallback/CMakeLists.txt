cmake_minimum_required(VERSION 3.16)
project(ColorDepthFallback CXX)

# Original HSV-based version
add_executable(${PROJECT_NAME}
    ColorDepthFallback.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${OrbbecSDK_LIBS}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${OrbbecSDK_INCLUDE_DIRS}
)

set_target_properties(ColorDepthFallback PROPERTIES
    CXX_STANDARD 17
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib/macOS"
    INSTALL_RPATH "@executable_path/../../lib/macOS"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

# YOLO-based version (NEW)
add_executable(ColorDepthFallback_YOLO
    ColorDepthFallback_YOLO.cpp
)

target_link_libraries(ColorDepthFallback_YOLO
    ${OpenCV_LIBS}
    ${OrbbecSDK_LIBS}
)

target_include_directories(ColorDepthFallback_YOLO PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${OrbbecSDK_INCLUDE_DIRS}
)

set_target_properties(ColorDepthFallback_YOLO PROPERTIES
    CXX_STANDARD 17
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib/macOS"
    INSTALL_RPATH "@executable_path/../../lib/macOS"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Copy ONNX model to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/best.onnx 
               ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/best.onnx COPYONLY)
