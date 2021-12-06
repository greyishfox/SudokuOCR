cmake_minimum_required(VERSION 3.5)

project(SudokuSolver LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# set build type to Debug/Release
set(CMAKE_BUILD_TYPE "Debug")

# Project Dependencies
find_package(Qt5 REQUIRED COMPONENTS Widgets PrintSupport)

# -------------- Threads ------------- #
find_package(Threads)

# -------------- Opencv -------------- #
set(OpenCV_DIR "../opencv_build/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Qt flags
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Executable
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/app/ocr.cpp
    src/app/ocr.h
    src/app/solver.cpp
    src/app/solver.h
    src/app/imageprocessing.cpp
    src/app/imageprocessing.h
    src/ui/widget.cpp
    src/ui/widget.h
    src/ui/widget.ui
    src/ui/qcustomplot-source/qcustomplot.cpp
    src/ui/qcustomplot-source/qcustomplot.h
    )

target_include_directories(${PROJECT_NAME} PRIVATE
    src/
    src/app
    src/ui/
    src/ui/qcustomplot-source/
    )

target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Widgets Qt5::PrintSupport)
# Link the openCV libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
# Link the OS specific thread libraries
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# Add file
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/img/sudoku_sample_image.jpeg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/img/OCR_training_set03.jpg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Define required c++ standard to C++11
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# Set compile options, enable warnings
target_compile_options(${PROJECT_NAME} PRIVATE
$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>: -Wall>
$<$<CXX_COMPILER_ID:MSVC>: /W4>
)
