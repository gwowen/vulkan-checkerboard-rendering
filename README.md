# Vulkan Checkerboard Cube

A Vulkan C++ sample that draws a rotating cube or toilet at **480p (854×480)** and reconstructs it to a **720p (1280×720)** window with temporal checkerboard upscaling.

Shaders are written in [Slang](https://shader-slang.org/) and compiled to SPIR-V at build time. The project uses **CMake** with the **Ninja** generator.

## How it works

Each frame the mesh is shaded into a 480p color + depth target. The projection is jittered by half a pixel, alternating every frame, so consecutive frames sample complementary checkerboard positions.

A fullscreen resolve pass then rebuilds 720p:

1. **Fresh cells** (`(x + y + frame) % 2 == 0`) keep the current 480p sample.
2. **Missing cells** reproject the previous frame using depth and the object's motion.
3. History is rejected on occlusion or out-of-bounds UVs, then replaced with a spatial neighborhood average.
4. Accepted history is neighborhood-clamped and blended slightly toward the current sample.

Press **Space** to compare this resolve against a naive bilinear upsample of the same 480p target.

## Requirements

- CMake 3.24+
- Ninja
- A C++20 compiler
- Vulkan 1.3 (dynamic rendering, shader draw parameters)
- GLFW 3
- `slangc` on `PATH`, or in `/opt/slang/bin` / `$VULKAN_SDK/bin`

GLM is vendored under `third_party/glm`.

## Build

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Debug builds enable the Khronos validation layer:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Run

```bash
./build/vulkan-cube-checkerboarding
./build/vulkan-toilet-checkerboarding
```

The toilet build uses a procedural bowl, seat, lid, tank, and handle, and spins around the Y axis.

| Key | Action |
| --- | --- |
| `Space` | Toggle checkerboard resolve vs bilinear upsample |
| `Esc` | Quit |

The window title shows which upsample path is active.

## Layout

```
shaders/cube.slang          Mesh vertex + fragment
shaders/reconstruct.slang   480p → 720p resolve
src/main.cpp                Vulkan renderer
src/geometry.hpp            Cube and toilet meshes
third_party/glm/            Header-only math
```

CMake compiles each Slang file to `build/shaders/<name>.spv` with `vertMain` and `fragMain` entry points.

## License

This project is licensed under the [MIT License](LICENSE).

Vendored GLM under `third_party/glm` remains under its own license (Happy Bunny / MIT).
