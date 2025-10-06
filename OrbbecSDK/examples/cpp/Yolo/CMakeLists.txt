# Add YOLO version
add_executable(YOLO Yolo.cpp)

target_include_directories(YOLO PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(YOLO 
    ${OpenCV_LIBS}
    ${OrbbecSDK_LIBS}
)

set_target_properties(YOLO PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib/macOS"
    INSTALL_RPATH "@executable_path/../../lib/macOS"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Fix library path after build on macOS
if(APPLE)
    add_custom_command(TARGET YOLO POST_BUILD
        COMMAND install_name_tool -change @loader_path/libOrbbecSDK.1.10.dylib 
                @executable_path/../../lib/macOS/libOrbbecSDK.1.10.dylib 
                $<TARGET_FILE:YOLO>
        COMMENT "Fixing library path for YOLO"
    )
endif()