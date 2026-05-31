shader_type canvas_item;

// Scene colour fed from GDScript each frame
uniform sampler2D u_scene : hint_default_black;
// Cascade-0 GI from compute pass
uniform sampler2DArray u_cascade : hint_default_black;
uniform float gi_strength : hint_range(0.0, 4.0) = 1.0;
uniform int blend_mode = 0;  // 0=additive  1=multiply

void fragment() {
    vec4 scene = texture(u_scene, UV);
    vec3 gi    = texture(u_cascade, vec3(UV, 0.0)).rgb * 10.0;

    vec3 result;
    if (blend_mode == 0) {
        result = scene.rgb + gi * scene.rgb;
    } else {
        result = scene.rgb * (1.0 + gi);
    }
    COLOR = vec4(result, 1.0);  // alpha MUST be 1.0
}