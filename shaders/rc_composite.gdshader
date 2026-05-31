// -----------------------------------------------------------------------
// RC3D Composite shader
// Applies the merged cascade-0 radiance as additive indirect lighting.
// Attach to a fullscreen quad in a CanvasLayer above the 3D viewport.
// -----------------------------------------------------------------------
shader_type canvas_item;

// Cascade-0 merged radiance (set from RCManager each frame)
uniform sampler2DArray u_cascade : hint_default_black;
// Strength multiplier
uniform float gi_strength : hint_range(0.0, 4.0) = 1.0;
// Blend mode: 0 = additive, 1 = multiply
uniform int blend_mode = 0;

void fragment() {
    vec4 scene = texture(TEXTURE, UV);  // the 3D viewport texture
    // Sample cascade level 0 (finest) from the 2D array layer 0
    vec3 gi = texture(u_cascade, vec3(UV, 0.0)).rgb * gi_strength;

    vec3 result;
    if (blend_mode == 0) {
        result = scene.rgb + gi;
    } else {
        result = scene.rgb * (1.0 + gi);
    }
    COLOR = vec4(result, scene.a);
}
