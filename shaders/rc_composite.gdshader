shader_type canvas_item;
// Renders the 3D scene + RC GI overlay.
// u_scene is fed every frame from rc_manager.gd via set_shader_parameter.
uniform sampler2D      u_scene   : hint_default_black;
uniform sampler2DArray u_cascade : hint_default_black;
uniform float gi_strength : hint_range(0.0, 8.0) = 1.5;
uniform int   blend_mode  = 0; // 0 = additive, 1 = multiply

void fragment() {
	vec4 scene = texture(u_scene,   UV);
	vec3 gi    = texture(u_cascade, vec3(UV, 0.0)).rgb * gi_strength;
	vec3 result;
	if (blend_mode == 0) {
		result = scene.rgb + gi * scene.rgb;
	} else {
		result = scene.rgb * (1.0 + gi);
	}
	COLOR = vec4(result, 1.0);
}
