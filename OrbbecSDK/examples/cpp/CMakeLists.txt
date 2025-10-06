cmake_minimum_required(VERSION 3.15)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================================
# CUSTOM SAMPLES - Ball Tracking Project
# ============================================================================

# opencv required
if(${OpenCV_FOUND})
    # Your custom ball tracking samples
    add_subdirectory(Sample-ColorDepthFallback)
    add_subdirectory(Sample-TrainingDataCollector)
    add_subdirectory(Yolo)
else()
    message(WARNING "OpenCV not found - custom samples require OpenCV")
endif(${OpenCV_FOUND})


