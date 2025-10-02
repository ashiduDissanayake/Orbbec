# Training Data Collector
add_executable(TrainingDataCollector
    TrainingDataCollector.cpp
)

target_link_libraries(TrainingDataCollector
    ${OpenCV_LIBS}
    ${OrbbecSDK_LIBS}
)

target_include_directories(TrainingDataCollector PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${OrbbecSDK_INCLUDE_DIRS}
)

set_property(TARGET TrainingDataCollector PROPERTY CXX_STANDARD 17)
set_property(TARGET TrainingDataCollector PROPERTY CXX_STANDARD_REQUIRED ON)