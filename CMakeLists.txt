cmake_minimum_required (VERSION 3.12)
project(particles LANGUAGES CXX CUDA)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(PythonLibs 3.8)
find_package(PkgConfig REQUIRED)
find_package(CUDA REQUIRED)
find_package(SDL2 REQUIRED)

pkg_check_modules(SDL2_TTF REQUIRED SDL2_ttf)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(
  "-Wall"
  "$<$<CONFIG:Debug>:-O0;-g3;-ggdb3>"
  "$<$<CONFIG:Release>:-O3;-mtune=native;-march=native;-DNDEBUG>"
)

#add_compile_options(-fsanitize=undefined)
#add_link_options(-fsanitize=address,undefined -static-libsan)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories(/usr/include/eigen3)
include_directories(src)
include_directories(src/lib)
include_directories(include)
include_directories("${CUDA_INCLUDE_DIRS}")

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_executable(
  test_all
  
  src/test/test_main.cpp
  src/test/test_ParticleEngine.cpp
  src/lib/CpuParticleEngine.cpp
  src/lib/GpuParticleEngine1.cpp
  src/lib/Particle.cpp
  
  src/kernels/kernels1.cu
  src/kernels/kernels.h
  
  src/lib/Particle.h  
  src/lib/CpuParticleEngine.h
  src/lib/GpuParticleEngine1.h

)

set_target_properties(test_all PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(test_all PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(test_all ${PYTHON_LIBRARIES})
target_link_libraries(test_all GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_all)



add_executable(sim)
target_sources(
  sim PRIVATE
  
  src/main.cpp
  src/lib/CpuParticleEngine.cpp
  src/lib/GpuParticleEngine1.cpp
  src/lib/Particle.cpp
  src/gui/MainWindow.cpp
  
  src/kernels/kernels1.cu
  src/kernels/kernels.h
  
  src/lib/Particle.h  
  src/lib/CpuParticleEngine.h
  src/lib/GpuParticleEngine1.h
  src/gui/MainWindow.h
  src/Parameters.h
)

target_include_directories(sim PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(sim PRIVATE ${SDL2_INCLUDE_DIRS})
target_include_directories(sim PRIVATE ${SDL2_TTF_INCLUDE_DIRS})
target_link_libraries(sim ${PYTHON_LIBRARIES})
target_link_libraries(sim ${SDL2_LIBRARIES})
target_link_libraries(sim ${SDL2_TTF_LIBRARIES})
