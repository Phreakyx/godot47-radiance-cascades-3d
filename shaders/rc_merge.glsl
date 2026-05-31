#[compute]
#version 450

// -----------------------------------------------------------------------
// Radiance Cascades 3D -- Merge pass
// Merges cascade level [child_idx] with the already-merged [parent_idx].
// Run from coarsest-1 down to 0.
// -----------------------------------------------------------------------

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant, std430) uniform PC {
    uint  child_idx;
    uint  parent_idx;
    uint  num_cascades;
    float screen_w;
    float screen_h;
    float blend_weight; // how much parent contributes (0..1)
    float pad0;
    float pad1;
} pc;

// Read-only source array (all levels freshly ray-traced this frame)
layout(rgba16f, set = 0, binding = 0) uniform readonly  image2DArray u_src;
// Write accumulated result (same array, different layers -- safe because
// we only write child while reading parent which is already done)
layout(rgba16f, set = 0, binding = 1) uniform writeonly image2DArray u_dst;

// -----------------------------------------------------------------------
// Bilinear fetch from a cascade layer using a float UV.
vec4 fetch_cascade(int layer, vec2 uv, vec2 res) {
    // Manual bilinear interpolation on the integer-addressed image2DArray.
    vec2 pf = uv * res - 0.5;
    ivec2 p0 = ivec2(floor(pf));
    vec2  t  = fract(pf);
    ivec2 p1 = p0 + ivec2(1);
    p0 = clamp(p0, ivec2(0), ivec2(res) - 1);
    p1 = clamp(p1, ivec2(0), ivec2(res) - 1);

    vec4 s00 = imageLoad(u_src, ivec3(p0,             layer));
    vec4 s10 = imageLoad(u_src, ivec3(ivec2(p1.x, p0.y), layer));
    vec4 s01 = imageLoad(u_src, ivec3(ivec2(p0.x, p1.y), layer));
    vec4 s11 = imageLoad(u_src, ivec3(p1,             layer));

    return mix(mix(s00, s10, t.x), mix(s01, s11, t.x), t.y);
}

void main() {
    ivec2 px  = ivec2(gl_GlobalInvocationID.xy);
    vec2  res = vec2(pc.screen_w, pc.screen_h);
    if (px.x >= int(pc.screen_w) || px.y >= int(pc.screen_h)) return;

    vec2 uv = (vec2(px) + 0.5) / res;

    vec4 child  = imageLoad(u_src, ivec3(px, int(pc.child_idx)));
    vec4 parent = fetch_cascade(int(pc.parent_idx), uv, res);

    // Merge rule:
    //   If child hit something (coverage > 0) keep child radiance.
    //   Where child missed, blend in parent radiance (which covers longer range).
    //   Alpha channel carries "unresolved" fraction for next merge.
    float child_miss  = 1.0 - child.a;  // fraction of rays that hit nothing
    vec3  merged_rgb  = child.rgb + parent.rgb * child_miss * pc.blend_weight;
    float merged_a    = child.a * parent.a; // both levels missed

    imageStore(u_dst, ivec3(px, int(pc.child_idx)), vec4(merged_rgb, merged_a));
}
