#[compute]
#version 450

// -----------------------------------------------------------------------
// Radiance Cascades 3D -- Ray-trace pass
// Godot 4.7 / Vulkan Forward+
//
// For each pixel this shader casts RAYS_PER_PROBE rays in the hemisphere
// defined by the G-buffer world-space normal.  Results are stored into a
// 2D-array texture where each layer holds one cascade level.
//
// Layout of a single cascade layer (rgba16f image2DArray):
//   Same XY resolution as the viewport.
//   Each pixel stores the *mean radiance* gathered along its rays.
// -----------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// ---- Push constants ---------------------------------------------------
layout(push_constant, std430) uniform PC {
    uint  cascade_index;    // 0 = finest
    uint  num_cascades;     // total cascade count
    uint  rays_c0;          // rays for cascade 0 (doubles each level)
    uint  max_steps;        // max ray-march steps
    float step_scale;       // world-space step length at cascade 0
    float screen_w;
    float screen_h;
    float near_plane;
    float far_plane;
    float pad0;
    float pad1;
    float pad2;
} pc;

// ---- Bindings ---------------------------------------------------------
// binding 0: output radiance cascade (rgba16f 2D array)
layout(rgba16f, set = 0, binding = 0) uniform writeonly image2DArray u_cascade_out;

// binding 1: G-buffer albedo (RGBA8 rendered by Godot)
layout(set = 0, binding = 1) uniform sampler2D u_albedo;

// binding 2: G-buffer depth (R32F / D32F from Godot's depth pre-pass)
layout(set = 0, binding = 2) uniform sampler2D u_depth;

// binding 3: G-buffer world-space normals (RGB16F)
layout(set = 0, binding = 3) uniform sampler2D u_normal;

// ---- Helpers ----------------------------------------------------------

// Fibonacci hemisphere sampling -- gives a quasi-uniform set of
// directions for a given ray index and total ray count.
vec3 fibonacci_hemisphere(uint i, uint n) {
    float golden = 2.399963229728;
    float theta  = acos(1.0 - (float(i) + 0.5) / float(n));
    float phi    = golden * float(i);
    return vec3(sin(theta) * cos(phi),
                cos(theta),
                sin(theta) * sin(phi));
}

// Build a TBN from a world-space normal (no tangent input needed).
mat3 tbn_from_normal(vec3 n) {
    vec3 up = abs(n.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t  = normalize(cross(up, n));
    vec3 b  = cross(n, t);
    return mat3(t, n, b);
}

// Reconstruct view-space position from depth + UV.
vec3 view_pos_from_depth(vec2 uv, float depth) {
    // Godot stores depth in [0,1] NDC.  We rebuild clip-space then
    // project back -- approximation using only near/far is fine for
    // screen-space marching.
    float z_ndc = depth * 2.0 - 1.0;
    float z_eye = (2.0 * pc.near_plane * pc.far_plane) /
                  (pc.far_plane + pc.near_plane - z_ndc * (pc.far_plane - pc.near_plane));
    vec2 ndc_xy = uv * 2.0 - 1.0;
    // Rough perspective un-project (assumes ~90 deg half-FOV; good enough
    // for relative screen-space directions).
    return vec3(ndc_xy * z_eye, -z_eye);
}

// Project a view-space point back to screen UV.
vec2 project_to_uv(vec3 vp) {
    // Inverse of view_pos_from_depth's assumption.
    vec2 ndc = vp.xy / (-vp.z);
    return ndc * 0.5 + 0.5;
}

// -----------------------------------------------------------------------
void main() {
    ivec2 px    = ivec2(gl_GlobalInvocationID.xy);
    vec2  res   = vec2(pc.screen_w, pc.screen_h);
    if (px.x >= int(pc.screen_w) || px.y >= int(pc.screen_h)) return;

    vec2 uv = (vec2(px) + 0.5) / res;

    // --- Sample G-buffer -----------------------------------------------
    float depth = texture(u_depth, uv).r;
    // Skip sky pixels (depth == 1.0 after clear in Forward+)
    if (depth >= 0.9999) {
        imageStore(u_cascade_out, ivec3(px, int(pc.cascade_index)), vec4(0.0));
        return;
    }

    vec3  albedo   = texture(u_albedo, uv).rgb;
    vec3  ws_normal = normalize(texture(u_normal, uv).rgb * 2.0 - 1.0);

    // View-space origin of this pixel.
    vec3 origin_vs = view_pos_from_depth(uv, depth);
    // Small offset along normal to avoid self-intersection.
    vec3 normal_vs = ws_normal; // approximation: treat world==view for dir

    // --- Per-cascade parameters ----------------------------------------
    // Rays double every cascade; step length doubles too so higher cascades
    // cover more scene volume.
    uint  rays      = pc.rays_c0 << pc.cascade_index;
    float step_len  = pc.step_scale * float(1u << pc.cascade_index);
    // Start distance: cascade N begins where cascade N-1 ended.
    float t_min = 0.0;
    if (pc.cascade_index > 0u) {
        // Each cascade starts at the far end of the previous one
        uint prev_rays  = pc.rays_c0 << (pc.cascade_index - 1u);
        t_min = step_len * 0.5 * float(pc.max_steps);
    }
    float t_max = t_min + step_len * float(pc.max_steps);

    mat3 tbn = tbn_from_normal(normal_vs);

    vec3 total_radiance = vec3(0.0);
    float hits = 0.0;

    for (uint r = 0u; r < rays; ++r) {
        vec3 local_dir = fibonacci_hemisphere(r, rays);
        vec3 ray_dir   = normalize(tbn * local_dir);

        // Screen-space ray march
        float t     = t_min + step_len * 0.5; // start mid-cell
        vec3  color = vec3(0.0);
        bool  hit   = false;

        for (uint s = 0u; s < pc.max_steps; ++s) {
            vec3 sample_pos = origin_vs + ray_dir * t;
            if (sample_pos.z >= 0.0) break; // behind camera

            vec2 sample_uv = project_to_uv(sample_pos);
            if (any(lessThan(sample_uv, vec2(0.0))) ||
                any(greaterThan(sample_uv, vec2(1.0)))) break;

            float sample_depth = texture(u_depth, sample_uv).r;
            vec3  sample_vs    = view_pos_from_depth(sample_uv, sample_depth);

            // Intersection test: sampled surface is close to our ray point
            float dist = abs(sample_vs.z - sample_pos.z);
            if (dist < step_len * 1.5 && sample_depth < 0.9999) {
                color = texture(u_albedo, sample_uv).rgb;
                hit   = true;
                break;
            }
            t += step_len;
        }

        if (hit) {
            total_radiance += color;
            hits += 1.0;
        }
    }

    vec3 mean_radiance = hits > 0.0 ? total_radiance / float(rays) : vec3(0.0);
    // Store: rgb = gathered radiance, a = coverage (for merge alpha compositing)
    float coverage = hits / float(rays);
    imageStore(u_cascade_out,
               ivec3(px, int(pc.cascade_index)),
               vec4(mean_radiance, coverage));
}
