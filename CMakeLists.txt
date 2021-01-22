project(optix7tutorial)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
mark_as_advanced(CUDA_SDK_ROOT_DIR)

# CMAKE configs
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/third_party/cmake")
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Find GLM
find_package(GLM REQUIRED)
message(STATUS "GLM included at ${GLM_INCLUDE_DIR}")

# Find GLFW3
find_package(GLFW3 REQUIRED)
message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")

# Find CUDA
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
message(STATUS "Found CUDA in ${CUDA_INCLUDE_DIR}")
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)
mark_as_advanced(CUDA_SDK_ROOT_DIR)

# Find OptiX
find_package(OptiX REQUIRED VERSION 7.0)
include_directories(${OptiX_INCLUDE})
message(STATUS "Found OptiX in ${OPTIX_INCLUDE_DIR}")

# Find OpenGL
set(OpenGL_GL_PREFERENCE GLVND)
if(WIN32)
  set(LIBS glfw3 opengl32)
elseif(UNIX AND NOT APPLE)
  find_package(OpenGL REQUIRED)
  add_definitions(${OPENGL_DEFINITIONS})
  find_package(X11 REQUIRED)
  set(LIBS ${GLFW3_LIBRARY} X11 Xrandr Xinerama Xi Xxf86vm Xcursor GL dl pthread)
  set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")
endif(WIN32)

# compile common libraries
include_directories(common)
add_subdirectory(common/glfWindow EXCLUDE_FROM_ALL)
include(${CMAKE_MODULE_PATH}/configure_optix.cmake)

# compile nvidia support libraries
configure_file(third_party/sutil/sampleConfig.h.in third_party/sutil/sampleConfig.h @ONLY)
include_directories(third_party/sutil)
add_subdirectory(third_party/sutil/support)
add_subdirectory(third_party/sutil)

# set all LIBS
set(LIBS ${LIBS} ${optix_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

# build tutorials
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(tutorials/01_getting_started)
add_subdirectory(tutorials/02_triangle)
add_subdirectory(tutorials/03_sphere)
add_subdirectory(tutorials/04_lights)
