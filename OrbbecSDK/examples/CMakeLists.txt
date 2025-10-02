cmake_minimum_required(VERSION 3.15)
project(OrbbecSDK-Samples LANGUAGES CXX C)

if(UNIX)
  set(CMAKE_CXX_FLAGS "-std=c++11 -pthread ${CMAKE_CXX_FLAGS}")
endif()

if(NOT DEFINED OrbbecSDK_FOUND)
    # set(OrbbecSDK_DIR "path/to/OrbbecSDK/lib" )
    message(STATUS "OrbbecSDK_DIR: ${OrbbecSDK_DIR}")
    find_package(OrbbecSDK REQUIRED)
endif()

# find opencv
# set(OpenCV_DIR "your/path/to/opencv/lib")
find_package(OpenCV QUIET)

if(UNIX)
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  if(APPLE)
    # Ensure macOS binaries can locate the packaged SDK dylibs in the source layout
    set(CMAKE_BUILD_RPATH "@loader_path/../../lib/macOS")
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../lib;@loader_path/../../lib/macOS")
  else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib")
  endif()
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

# Only add cpp subdirectory (c examples were deleted)
add_subdirectory(cpp)

install(DIRECTORY
  ${OrbbecSDK_LIBS_DIR}/ DESTINATION
  bin
  FILES_MATCHING PATTERN "*.so"
  PATTERN "*.so.*"
  PATTERN "*.dylib"
  PATTERN "*.*.dylib"
  PATTERN "*.dll"
  PATTERN ".lib"
  )
