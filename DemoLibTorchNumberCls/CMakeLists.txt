cmake_minimum_required(VERSION 3.2)

project(hello)

set(CMAKE_CXX_STANDARD 14)



set(depend_library_path "/ai_data/ai_depend")

find_package(Torch "1.8.1" REQUIRED PATHS ${depend_library_path}/libtorch_cxx11_abi/)

# TORCH_LIBRARIES TORCH_INCLUDE_DIRS

find_package( OpenCV "3.4.2" REQUIRED PATHS ${depend_library_path}/opencv-3.4.2/install/)

# OpenCV_INCLUDE_DIRS OpenCV_LIBRARIES_DIRS OpenCV_LIBRARIES

add_executable(main main.cpp)

target_include_directories(main PUBLIC 
    ${TORCH_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
)

# target_link_directories(main PUBLIC
#     ${OpenCV_LIBRARIES_DIRS}
# )

target_link_libraries(main PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
)


