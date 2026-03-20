cmake_minimum_required(VERSION 3.10)
project(SegmentationPostProcessing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)

# Build executable
add_executable(postprocessing
    postprocessing.cpp
)

target_include_directories(postprocessing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(postprocessing PRIVATE
    ${OpenCV_LIBS}
)

# Install target
install(TARGETS postprocessing DESTINATION bin)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
