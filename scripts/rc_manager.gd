## RCManager -- Radiance Cascades 3D GI
## Attach to the root Node3D. Requires Godot 4.7 beta 4+, Forward+.
extends Node

@export_group("Cascade")
@export var NUM_CASCADES    : int   = 4
@export var RAYS_C0         : int   = 4
@export var MAX_MARCH_STEPS : int   = 16
@export var STEP_SCALE      : float = 0.025

@export_group("Composite")
@export var gi_strength        : float = 1.5
@export var blend_mode         : int   = 0
@export var composite_material : ShaderMaterial

# RD state
var _rd              : RenderingDevice
var _rt_shader       : RID
var _mg_shader       : RID
var _rt_pipe         : RID
var _mg_pipe         : RID
var _cascade_arr_src : RID
var _cascade_arr_dst : RID
var _cascade_tex_dst : Texture2DArrayRD
var _albedo_rd       : RID
var _depth_rd        : RID
var _normal_rd       : RID
var _sampler         : RID
var _rt_uniform_set  : RID
var _mg_uniform_set  : RID
var _vp_size         : Vector2i
var _ready_ok        : bool = false

# Pre-built static byte arrays so we never alloc in _process
var _depth_bytes  : PackedByteArray
var _normal_bytes : PackedByteArray

func _ready() -> void:
	_rd = RenderingServer.get_rendering_device()
	if _rd == null:
		push_error("RC3D: RenderingDevice not available -- use Forward+ renderer.")
		return
	call_deferred("_setup_gpu")

func _setup_gpu() -> void:
	var vp    := get_viewport()
	_vp_size   = vp.size
	if _vp_size.x == 0 or _vp_size.y == 0:
		await get_tree().process_frame
		_vp_size = vp.size
	var W := _vp_size.x
	var H := _vp_size.y
	print("RC3D: setting up GPU at %d x %d" % [W, H])

	# Shaders
	var rt_file := load("res://shaders/rc_raytrace.glsl") as RDShaderFile
	var mg_file := load("res://shaders/rc_merge.glsl")   as RDShaderFile
	_rt_shader = _rd.shader_create_from_spirv(rt_file.get_spirv())
	_mg_shader = _rd.shader_create_from_spirv(mg_file.get_spirv())
	_rt_pipe   = _rd.compute_pipeline_create(_rt_shader)
	_mg_pipe   = _rd.compute_pipeline_create(_mg_shader)

	# Cascade texture arrays (rgba16f)
	var fmt := RDTextureFormat.new()
	fmt.width        = W
	fmt.height       = H
	fmt.format       = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	fmt.texture_type = RenderingDevice.TEXTURE_TYPE_2D_ARRAY
	fmt.array_layers = NUM_CASCADES
	fmt.usage_bits   = (RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
						RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
						RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT)
	_cascade_arr_src = _rd.texture_create(fmt, RDTextureView.new(), [])
	_cascade_arr_dst = _rd.texture_create(fmt, RDTextureView.new(), [])
	_cascade_tex_dst = Texture2DArrayRD.new()
	_cascade_tex_dst.set_texture_rd_rid(_cascade_arr_dst)

	# G-buffer upload textures
	var af := RDTextureFormat.new()
	af.width = W; af.height = H
	af.format       = RenderingDevice.DATA_FORMAT_R8G8B8A8_SRGB
	af.texture_type = RenderingDevice.TEXTURE_TYPE_2D
	af.usage_bits   = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT
	_albedo_rd = _rd.texture_create(af, RDTextureView.new(), [])

	var df := RDTextureFormat.new()
	df.width = W; df.height = H
	df.format       = RenderingDevice.DATA_FORMAT_R32_SFLOAT
	df.texture_type = RenderingDevice.TEXTURE_TYPE_2D
	df.usage_bits   = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT
	_depth_rd = _rd.texture_create(df, RDTextureView.new(), [])

	var nf := RDTextureFormat.new()
	nf.width = W; nf.height = H
	nf.format       = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	nf.texture_type = RenderingDevice.TEXTURE_TYPE_2D
	nf.usage_bits   = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT
	_normal_rd = _rd.texture_create(nf, RDTextureView.new(), [])

	# Pre-build static upload buffers -- avoids any per-frame GDScript allocation
	# Depth: all zeros (0.0f) = treat every pixel as at near plane
	_depth_bytes = PackedByteArray()
	_depth_bytes.resize(W * H * 4)
	_depth_bytes.fill(0)

	# Normal: encode (0,1,0) world-up as fp16 rgba = (0.5, 1.0, 0.5, 0.0)
	# We write one pixel pattern then use resize+fill trick via byte copy
	# fp16: 0.5 = 0x3800, 1.0 = 0x3C00, 0.0 = 0x0000  (little-endian)
	var pixel := PackedByteArray([0x00, 0x38, 0x00, 0x3C, 0x00, 0x38, 0x00, 0x00])
	_normal_bytes = PackedByteArray()
	_normal_bytes.resize(W * H * 8)
	# Fill by tiling the 8-byte pixel -- use a small loop over rows, bulk copy columns
	for i in range(W * H):
		var off := i * 8
		_normal_bytes[off + 0] = 0x00; _normal_bytes[off + 1] = 0x38
		_normal_bytes[off + 2] = 0x00; _normal_bytes[off + 3] = 0x3C
		_normal_bytes[off + 4] = 0x00; _normal_bytes[off + 5] = 0x38
		_normal_bytes[off + 6] = 0x00; _normal_bytes[off + 7] = 0x00
	# NOTE: this loop runs ONCE at startup, not every frame

	# Sampler
	var ss := RDSamplerState.new()
	ss.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.repeat_u   = RenderingDevice.SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE
	ss.repeat_v   = RenderingDevice.SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE
	_sampler = _rd.sampler_create(ss)

	# Raytrace uniform set
	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u0.binding = 0; u0.add_id(_cascade_arr_src)

	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u1.binding = 1; u1.add_id(_sampler); u1.add_id(_albedo_rd)

	var u2 := RDUniform.new()
	u2.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u2.binding = 2; u2.add_id(_sampler); u2.add_id(_depth_rd)

	var u3 := RDUniform.new()
	u3.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u3.binding = 3; u3.add_id(_sampler); u3.add_id(_normal_rd)

	_rt_uniform_set = _rd.uniform_set_create([u0, u1, u2, u3], _rt_shader, 0)

	# Merge uniform set
	var m0 := RDUniform.new()
	m0.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	m0.binding = 0; m0.add_id(_cascade_arr_src)

	var m1 := RDUniform.new()
	m1.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	m1.binding = 1; m1.add_id(_cascade_arr_dst)

	_mg_uniform_set = _rd.uniform_set_create([m0, m1], _mg_shader, 0)

	_ready_ok = true
	print("RC3D: GPU setup complete (%d x %d, %d cascades)" % [W, H, NUM_CASCADES])

func _process(_delta: float) -> void:
	if not _ready_ok:
		return

	var W := _vp_size.x
	var H := _vp_size.y

	# Upload albedo from viewport (only GPU->CPU copy, unavoidable without CompositorEffect)
	var vp_img := get_viewport().get_texture().get_image()
	if vp_img == null: return
	vp_img.convert(Image.FORMAT_RGBA8)
	_rd.texture_update(_albedo_rd, 0, vp_img.get_data())

	# Upload pre-built static depth + normal buffers (no per-frame GDScript work)
	_rd.texture_update(_depth_rd,  0, _depth_bytes)
	_rd.texture_update(_normal_rd, 0, _normal_bytes)

	var gx := int(ceil(float(W) / 8.0))
	var gy := int(ceil(float(H) / 8.0))

	# Raytrace pass
	var rt_list := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(rt_list, _rt_pipe)
	_rd.compute_list_bind_uniform_set(rt_list, _rt_uniform_set, 0)
	for c in range(NUM_CASCADES):
		var pc := PackedInt32Array([c, NUM_CASCADES, RAYS_C0, MAX_MARCH_STEPS]).to_byte_array()
		pc += PackedFloat32Array([STEP_SCALE, float(W), float(H), 0.05, 500.0, 0.0, 0.0, 0.0]).to_byte_array()
		_rd.compute_list_set_push_constant(rt_list, pc, pc.size())
		_rd.compute_list_dispatch(rt_list, gx, gy, 1)
	_rd.compute_list_end()
	_rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE, RenderingDevice.BARRIER_MASK_COMPUTE)

	# Merge pass
	var mg_list := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(mg_list, _mg_pipe)
	_rd.compute_list_bind_uniform_set(mg_list, _mg_uniform_set, 0)
	for c in range(NUM_CASCADES - 2, -1, -1):
		var pc := PackedInt32Array([c, c + 1, NUM_CASCADES]).to_byte_array()
		pc += PackedFloat32Array([float(W), float(H), 0.5, 0.0, 0.0]).to_byte_array()
		_rd.compute_list_set_push_constant(mg_list, pc, pc.size())
		_rd.compute_list_dispatch(mg_list, gx, gy, 1)
	_rd.compute_list_end()
	_rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE, RenderingDevice.BARRIER_MASK_RASTER)

	# Feed composite shader
	if composite_material != null:
		composite_material.set_shader_parameter("u_scene",    get_viewport().get_texture())
		composite_material.set_shader_parameter("u_cascade",  _cascade_tex_dst)
		composite_material.set_shader_parameter("gi_strength", gi_strength)
		composite_material.set_shader_parameter("blend_mode",  blend_mode)

func _exit_tree() -> void:
	if not _ready_ok: return
	for r in [_rt_pipe, _mg_pipe, _rt_shader, _mg_shader,
			  _cascade_arr_src, _cascade_arr_dst,
			  _albedo_rd, _depth_rd, _normal_rd, _sampler]:
		if r.is_valid(): _rd.free_rid(r)
