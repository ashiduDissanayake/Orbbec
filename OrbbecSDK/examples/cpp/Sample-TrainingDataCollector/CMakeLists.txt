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

set_target_properties(TrainingDataCollector PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Fix library path after build on macOS
if(APPLE)
    add_custom_command(TARGET TrainingDataCollector POST_BUILD
        COMMAND install_name_tool -change @loader_path/libOrbbecSDK.1.10.dylib 
                @executable_path/../../lib/macOS/libOrbbecSDK.1.10.dylib 
                $<TARGET_FILE:TrainingDataCollector>
        COMMENT "Fixing library path for TrainingDataCollector"
    )
endif()