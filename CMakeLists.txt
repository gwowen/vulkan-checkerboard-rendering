# Build:
#   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ./build/vulkan-cube-checkerboarding
#   ./build/vulkan-toilet-checkerboarding

cmake_minimum_required(VERSION 3.24)
project(vulkan_cube_checkerboarding LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW REQUIRED glfw3)
endif()

add_library(glm INTERFACE)
target_include_directories(glm INTERFACE "${CMAKE_SOURCE_DIR}/third_party/glm")
add_library(glm::glm ALIAS glm)

find_program(SLANGC_EXECUTABLE
    NAMES slangc
    HINTS /opt/slang/bin $ENV{VULKAN_SDK}/bin
    REQUIRED
)

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
set(SLANG_SHADERS
    "${CMAKE_SOURCE_DIR}/shaders/cube.slang"
    "${CMAKE_SOURCE_DIR}/shaders/reconstruct.slang"
)

set(SPIRV_SHADERS)
foreach(SLANG_FILE IN LISTS SLANG_SHADERS)
    get_filename_component(SHADER_NAME "${SLANG_FILE}" NAME_WE)
    set(SPV_FILE "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT "${SPV_FILE}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
        COMMAND "${SLANGC_EXECUTABLE}"
            "${SLANG_FILE}"
            -target spirv
            -profile spirv_1_4
            -emit-spirv-directly
            -fvk-use-entrypoint-name
            -matrix-layout-column-major
            -obfuscate
            -entry vertMain
            -entry fragMain
            -o "${SPV_FILE}"
        DEPENDS "${SLANG_FILE}"
        COMMENT "Compiling ${SHADER_NAME}.slang -> SPIR-V"
        VERBATIM
    )
    list(APPEND SPIRV_SHADERS "${SPV_FILE}")
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPIRV_SHADERS})

function(add_checkerboard_app NAME)
    add_executable(${NAME} src/main.cpp)
    add_dependencies(${NAME} shaders)
    target_include_directories(${NAME} PRIVATE ${Vulkan_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(${NAME} PRIVATE
        SHADER_DIR="${SHADER_OUTPUT_DIR}"
        GLFW_INCLUDE_VULKAN
        ${ARGN}
    )
    target_compile_options(${NAME} PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
    target_link_libraries(${NAME} PRIVATE Vulkan::Vulkan glm::glm)
    if(TARGET glfw)
        target_link_libraries(${NAME} PRIVATE glfw)
    else()
        target_include_directories(${NAME} PRIVATE ${GLFW_INCLUDE_DIRS})
        target_link_libraries(${NAME} PRIVATE ${GLFW_LIBRARIES})
    endif()
endfunction()

add_checkerboard_app(vulkan-cube-checkerboarding)
add_checkerboard_app(vulkan-toilet-checkerboarding SCENE_TOILET=1)
