cmake_minimum_required(VERSION 3.16)
project(ldls_driver CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to a llama.cpp checkout (tested commit: see README); it is built as part of this project.
# Default: ./llama.cpp inside this folder, else ../llama.cpp next to it.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
    set(LLAMA_CPP_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")
else()
    set(LLAMA_CPP_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp")
endif()
set(LLAMA_CPP_DIR "${LLAMA_CPP_DEFAULT}" CACHE PATH "llama.cpp source directory")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp not found at '${LLAMA_CPP_DIR}'. Clone it into ./llama.cpp (see README) or pass -DLLAMA_CPP_DIR=<path>.")
endif()

set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL            OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON    OFF CACHE BOOL "" FORCE)

# CPU only, on every platform (macOS would otherwise build the Metal GPU backend)
set(GGML_METAL            OFF CACHE BOOL "" FORCE)
set(GGML_CUDA             OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN           OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)

add_executable(ldls_driver driver/ldls_driver.cpp)
target_include_directories(ldls_driver PRIVATE ${LLAMA_CPP_DIR}/vendor)
target_link_libraries(ldls_driver PRIVATE llama)
