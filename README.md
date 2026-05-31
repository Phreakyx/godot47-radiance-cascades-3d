# Godot 4.7 – Radiance Cascades 3D GI

A screen-space **Radiance Cascades** global illumination experiment for Godot 4.7 beta.

## How it works

1. A `WorldEnvironment` + `DirectionalLight3D` + a few meshes form the demo scene.
2. `RCManager` (Node) hooks into `RenderingServer` post-render callbacks to grab the
   **depth** and **normal** buffers from the current `Viewport` via `get_render_target_texture`.
3. Two compute passes run every frame:
   - **rc_raytrace.glsl** – for each pixel, casts N rays (per cascade level) in the
     hemisphere defined by the G-buffer normal, stepping through screen space.
   - **rc_merge.glsl** – merges cascade levels top-down, bilinear-interpolating
     parent probes into child probes.
4. A fullscreen `CanvasLayer` quad samples cascade level-0 and adds it as diffuse
   indirect light via a `screen_texture` shader trick.

## Structure

```
project.godot
scenes/
  main.tscn          – demo scene (meshes, lights, camera)
scripts/
  rc_manager.gd      – RenderingDevice compute orchestration
shaders/
  rc_raytrace.glsl   – per-cascade ray march compute shader
  rc_merge.glsl      – cascade merge compute shader
  rc_composite.gdshader – fullscreen composite (CanvasLayer quad)
```

## Requirements

- Godot **4.7 beta 4** or later (Vulkan Forward+)
- GPU that supports `VK_EXT_shader_atomic_float` (most 2019+ discrete GPUs)

## Usage

1. Open the project in Godot 4.7 beta 4.
2. Open `scenes/main.tscn` and run.
3. Tweak `RCManager` exported vars (`NUM_CASCADES`, `RAYS_C0`, `MAX_MARCH_STEPS`).
