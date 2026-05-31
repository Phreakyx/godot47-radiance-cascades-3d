## RCManager -- Radiance Cascades 3D GI orchestrator
## Attach to any Node in the scene. Requires Godot 4.7 beta 4+, Forward+.
##
## Pipeline per frame:
##   1. Capture G-buffer textures (albedo, depth, normals) from the viewport.
##   2. Run rc_raytrace compute shader for every cascade level.
##   3. Run rc_merge compute shader from coarsest-1 down to 0.
##   4. Upload cascade-0 texture to the composite material.
extends Node

@export var composite_rect: TextureRect
# ---- Exported settings ------------------------------------------------
@export_group("Cascade")
@export var NUM_CASCADES   : int   = 4       ## how many cascade levels (3-5 typical)
@export var RAYS_C0        : int   = 4       ## rays per pixel at level 0 (4 or 8)
@export var MAX_MARCH_STEPS: int   = 16      ## ray march steps per ray
@export var STEP_SCALE     : float = 0.02    ## world-space step at cascade 0 (tune to scene scale)

@export_group("Composite")
@export var gi_strength    : float = 1.0
@export var blend_mode     : int   = 0       ## 0=additive, 1=multiply
@export var composite_material: ShaderMaterial  ## assign the CanvasLayer quad's material

# ---- Internal RenderingDevice state -----------------------------------
var _rd       : RenderingDevice
var _rt_shader: RID
var _mg_shader: RID
var _rt_pipe  : RID
var _mg_pipe  : RID

# Cascade texture array (rgba16f, w*h*NUM_CASCADES)
var _cascade_arr_src  : RID
var _cascade_arr_dst  : RID
var _cascade_tex_src  : Texture2DArrayRD
var _cascade_tex_dst  : Texture2DArrayRD

# G-buffer upload RIDs
var _albedo_rd   : RID
var _depth_rd    : RID
var _normal_rd   : RID
var _sampler     : RID

# Uniform sets
var _rt_uniform_set : RID   # raytrace pass
var _mg_uniform_set : RID   # merge pass

var _vp_size : Vector2i
var _ready_ok : bool = false

# -----------------------------------------------------------------------
func _ready() -> void:
	_rd = RenderingServer.get_rendering_device()
	if _rd == null:
		push_error("RC3D: RenderingDevice not available. Use Forward+ renderer.")
		return

	# Defer actual GPU setup until after first frame so viewport size is known.
	call_deferred("_setup_gpu")


func _setup_gpu() -> void:
	var vp := get_viewport()
	_vp_size = vp.size
	var W := _vp_size.x
	var H := _vp_size.y

	# --- Load & compile compute shaders ---------------------------------
	var rt_file = load("res://shaders/rc_raytrace.glsl") as RDShaderFile
	var mg_file = load("res://shaders/rc_merge.glsl")   as RDShaderFile

	_rt_shader = _rd.shader_create_from_spirv(rt_file.get_spirv())
	_mg_shader = _rd.shader_create_from_spirv(mg_file.get_spirv())
	_rt_pipe   = _rd.compute_pipeline_create(_rt_shader)
	_mg_pipe   = _rd.compute_pipeline_create(_mg_shader)

	# --- Cascade storage -----------------------------------------------
	# Two arrays: src (written by raytrace, read by merge) and
	# dst (written by merge for the composite display).
	# For simplicity we use the same array for both and ping-pong via
	# different bindings -- the merge shader only writes child while
	# reading parent (already finished), so this is safe.
	var fmt := RDTextureFormat.new()
	fmt.width         = W
	fmt.height        = H
	fmt.format        = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	fmt.texture_type  = RenderingDevice.TEXTURE_TYPE_2D_ARRAY
	fmt.array_layers  = NUM_CASCADES
	fmt.usage_bits    = (RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
						 RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
						 RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT)

	_cascade_arr_src = _rd.texture_create(fmt, RDTextureView.new(), [])
	_cascade_arr_dst = _rd.texture_create(fmt, RDTextureView.new(), [])

	_cascade_tex_src = Texture2DArrayRD.new()
	_cascade_tex_src.set_texture_rd_rid(_cascade_arr_src)
	_cascade_tex_dst = Texture2DArrayRD.new()
	_cascade_tex_dst.set_texture_rd_rid(_cascade_arr_dst)

	# --- G-buffer textures (CPU-upload path) ---------------------------
	# We upload from get_image() each frame. For a production version
	# you would use RenderingServer render callbacks to get raw RIDs.
	var _mk_upload := func(fmt_in: RDTextureFormat) -> RID:
		return _rd.texture_create(fmt_in, RDTextureView.new(), [])

	var af := RDTextureFormat.new()
	af.width = W;  af.height = H
	af.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_SRGB
	af.texture_type = RenderingDevice.TEXTURE_TYPE_2D
	af.usage_bits = (RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
					 RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT)
	_albedo_rd = _rd.texture_create(af, RDTextureView.new(), [])

	var df := RDTextureFormat.new()
	df.width = W;  df.height = H
	df.format = RenderingDevice.DATA_FORMAT_R32_SFLOAT
	df.texture_type = RenderingDevice.TEXTURE_TYPE_2D
	df.usage_bits = (RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
					 RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT)
	_depth_rd = _rd.texture_create(df, RDTextureView.new(), [])

	var nf := RDTextureFormat.new()
	nf.width = W;  nf.height = H
	nf.format = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	nf.texture_type = RenderingDevice.TEXTURE_TYPE_2D
	nf.usage_bits = (RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
					 RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT)
	_normal_rd = _rd.texture_create(nf, RDTextureView.new(), [])

	# --- Sampler -----------------------------------------------------------
	var ss := RDSamplerState.new()
	ss.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.mip_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.repeat_u   = RenderingDevice.SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE
	ss.repeat_v   = RenderingDevice.SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE
	_sampler = _rd.sampler_create(ss)

	# --- Uniform sets ------------------------------------------------------
	# Raytrace set: cascade_out(img,0), albedo(samp,1), depth(samp,2), normal(samp,3)
	var u_cas_out := RDUniform.new()
	u_cas_out.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_cas_out.binding = 0
	u_cas_out.add_id(_cascade_arr_src)

	var u_albedo := RDUniform.new()
	u_albedo.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_albedo.binding = 1
	u_albedo.add_id(_sampler)
	u_albedo.add_id(_albedo_rd)

	var u_depth := RDUniform.new()
	u_depth.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_depth.binding = 2
	u_depth.add_id(_sampler)
	u_depth.add_id(_depth_rd)

	var u_normal := RDUniform.new()
	u_normal.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_normal.binding = 3
	u_normal.add_id(_sampler)
	u_normal.add_id(_normal_rd)

	_rt_uniform_set = _rd.uniform_set_create(
		[u_cas_out, u_albedo, u_depth, u_normal], _rt_shader, 0)

	# Merge set: src_array(img,0), dst_array(img,1)
	var u_mg_src := RDUniform.new()
	u_mg_src.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_mg_src.binding = 0
	u_mg_src.add_id(_cascade_arr_src)

	var u_mg_dst := RDUniform.new()
	u_mg_dst.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_mg_dst.binding = 1
	u_mg_dst.add_id(_cascade_arr_dst)

	_mg_uniform_set = _rd.uniform_set_create(
		[u_mg_src, u_mg_dst], _mg_shader, 0)

	_ready_ok = true
	print("RC3D: GPU setup complete (%d x %d, %d cascades)" % [W, H, NUM_CASCADES])


# -----------------------------------------------------------------------
func _process(_delta: float) -> void:
	if not _ready_ok:
		return

	var vp := get_viewport()
	var W  := _vp_size.x
	var H  := _vp_size.y

	# --- 1. Upload G-buffer images ----------------------------------------
	# Albedo from viewport screenshot (RGBA8)
	var vp_img := vp.get_texture().get_image()
	if vp_img == null: return
	vp_img.convert(Image.FORMAT_RGBA8)
	_rd.texture_update(_albedo_rd, 0, vp_img.get_data())

	# Depth: Godot 4 doesn't expose depth directly from GDScript easily;
	# we upload a grey placeholder and let the shader fall back gracefully.
	# For a real implementation, hook RenderingServer.frame_post_draw and
	# use environment_get_depth_texture_rd().
	# Here we at least zero-fill so sky pixels are skipped.
	var depth_bytes := PackedByteArray()
	depth_bytes.resize(W * H * 4)
	depth_bytes.fill(0)   # all 0.0f (near plane) -- rays will march from surface
	_rd.texture_update(_depth_rd, 0, depth_bytes)

	# Normal: encode world-space normals as RGB16f via a separate Viewport
	# (see scenes/main.tscn for the SubViewport + normal-output material).
	# If no normal texture is wired, upload zeros (normals = up = 0.5,0.5,0.5 in encode).
	var normal_bytes := PackedByteArray()
	normal_bytes.resize(W * H * 8)  # rgba16f = 8 bytes/pixel
	# Default: encode (0,1,0) world up as normal so hemisphere faces up
	for i in range(W * H):
		# Store vec4(0.5, 1.0, 0.5, 0.0) as fp16 -- approximation
		# 0.5 maps to half-float 0x3800, 1.0 to 0x3C00, 0.0 to 0x0000
		normal_bytes[i * 8 + 0] = 0x00; normal_bytes[i * 8 + 1] = 0x38  # 0.5
		normal_bytes[i * 8 + 2] = 0x00; normal_bytes[i * 8 + 3] = 0x3C  # 1.0
		normal_bytes[i * 8 + 4] = 0x00; normal_bytes[i * 8 + 5] = 0x38  # 0.5
		normal_bytes[i * 8 + 6] = 0x00; normal_bytes[i * 8 + 7] = 0x00  # 0.0
	_rd.texture_update(_normal_rd, 0, normal_bytes)

	var gx := int(ceil(float(W) / 8.0))
	var gy := int(ceil(float(H) / 8.0))

	# --- 2. Raytrace pass (all cascade levels) ----------------------------
	var rt_list := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(rt_list, _rt_pipe)
	_rd.compute_list_bind_uniform_set(rt_list, _rt_uniform_set, 0)

	for c in range(NUM_CASCADES):
		var pc_data := PackedFloat32Array([
			float(c),
			float(NUM_CASCADES),
			float(RAYS_C0),
			float(MAX_MARCH_STEPS),
			STEP_SCALE,
			float(W),
			float(H),
			0.05,   # near plane
			500.0,  # far plane
			0.0, 0.0, 0.0
		])
		# push_constant expects u32 for first 4 fields; pack as bytes manually
		var pc_u32 := PackedInt32Array([c, NUM_CASCADES, RAYS_C0, MAX_MARCH_STEPS])
		var pc_f32 := PackedFloat32Array([STEP_SCALE, float(W), float(H),
										  0.05, 500.0, 0.0, 0.0, 0.0])
		var pc_bytes := pc_u32.to_byte_array() + pc_f32.to_byte_array()

		_rd.compute_list_set_push_constant(rt_list, pc_bytes, pc_bytes.size())
		_rd.compute_list_dispatch(rt_list, gx, gy, 1)
		# Barrier between cascade levels (each writes to its own layer, OK to batch)
	_rd.compute_list_end()

	_rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE, RenderingDevice.BARRIER_MASK_COMPUTE)

	# --- 3. Merge pass (coarsest-1 down to 0) ----------------------------
	var mg_list := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(mg_list, _mg_pipe)
	_rd.compute_list_bind_uniform_set(mg_list, _mg_uniform_set, 0)

	for c in range(NUM_CASCADES - 2, -1, -1):
		var parent := c + 1
		var pc_u32 := PackedInt32Array([c, parent, NUM_CASCADES])
		var pc_f32 := PackedFloat32Array([float(W), float(H), 0.5, 0.0, 0.0])
		var pc_bytes := pc_u32.to_byte_array() + pc_f32.to_byte_array()

		_rd.compute_list_set_push_constant(mg_list, pc_bytes, pc_bytes.size())
		_rd.compute_list_dispatch(mg_list, gx, gy, 1)
	_rd.compute_list_end()

	_rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE, RenderingDevice.BARRIER_MASK_RASTER)

	# --- 4. Update composite material ------------------------------------
	# Feed viewport texture as u_scene
	if composite_material != null:
		# get_viewport().get_texture() returns the live viewport texture
		# No path needed — works directly
		composite_material.set_shader_parameter(
			"u_scene", get_viewport().get_texture()
		)
		composite_material.set_shader_parameter("u_cascade", _cascade_tex_dst)
		composite_material.set_shader_parameter("gi_strength", gi_strength)
		composite_material.set_shader_parameter("blend_mode", blend_mode)


# -----------------------------------------------------------------------
func _exit_tree() -> void:
	if not _ready_ok: return
	_rd.free_rid(_rt_pipe);      _rd.free_rid(_mg_pipe)
	_rd.free_rid(_rt_shader);    _rd.free_rid(_mg_shader)
	_rd.free_rid(_cascade_arr_src)
	_rd.free_rid(_cascade_arr_dst)
	_rd.free_rid(_albedo_rd)
	_rd.free_rid(_depth_rd)
	_rd.free_rid(_normal_rd)
	_rd.free_rid(_sampler)
