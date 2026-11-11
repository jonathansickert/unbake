"""
Render a video of one asset from a fixed camera while the eval env maps (env_maps/eval/) rotate a full
turn around the up axis, the env maps side by side. Each env map is scaled to the same solid-angle weighted
mean luminance as in unbake/evaluation (ENV_MAP_MEAN_LUMINANCE), so dim maps are not nearly black.

Run from the repo root:
    /Applications/Blender.app/Contents/MacOS/Blender -b -P data_toolkit/render_envmap_video.py -- \
        baked_env/<sha256>/pbr.glb [out.mp4]

Output: out.mp4 (default: next to the .glb as <name>_envmaps.mp4).
"""
import os
import sys
import glob
import math
import tempfile
import numpy as np
import bpy
from mathutils import Vector

EVAL_ENV_MAPS = sorted(glob.glob('env_maps/eval/*.exr'))
ENV_MAP_MEAN_LUMINANCE = 1.0
FRAMES, FPS = 96, 24                 # one full env map turn
RESOLUTION = 512                     # per env map panel
SAMPLES = 64
YAW, PITCH, FOV = 45.0, 20.0, 40.0   # fixed camera, in degrees
SHOW_BACKGROUND = True               # show the rotating env map behind the asset

args = sys.argv[sys.argv.index('--') + 1:]
glb = args[0]
out_path = os.path.abspath(args[1] if len(args) > 1 else os.path.splitext(glb)[0] + '_envmaps.mp4')


def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = RESOLUTION
    scene.render.film_transparent = not SHOW_BACKGROUND
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.view_transform = 'Standard'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for backend in ('OPTIX', 'CUDA', 'METAL'):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == backend
        scene.cycles.device = 'GPU'
        break
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(glb))
    bpy.context.view_layer.update()

    # Fixed camera framing the bounding sphere
    meshes = [o for o in scene.objects if o.type == 'MESH']
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector(tuple(min(c[i] for c in corners) for i in range(3)))
    hi = Vector(tuple(max(c[i] for c in corners) for i in range(3)))
    center, radius = (lo + hi) / 2, (hi - lo).length / 2
    distance = 1.1 * radius / math.sin(math.radians(FOV) / 2)
    yaw, pitch = math.radians(YAW), math.radians(PITCH)
    direction = Vector((math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)))
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
    cam.data.angle = math.radians(FOV)
    cam.data.clip_start, cam.data.clip_end = distance * 0.01, distance * 10
    cam.location = center + direction * distance
    cam.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()
    scene.collection.objects.link(cam)
    scene.camera = cam

    # World: env map -> rotation -> background
    world = bpy.data.worlds.new('World')
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    mapping = nodes.new('ShaderNodeMapping')
    env = nodes.new('ShaderNodeTexEnvironment')
    bg = nodes.new('ShaderNodeBackground')
    links.new(nodes.new('ShaderNodeTexCoord').outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], env.inputs['Vector'])
    links.new(env.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs[0], nodes.new('ShaderNodeOutputWorld').inputs['Surface'])
    scene.world = world
    return scene, mapping, env, bg


def luminance_scale(image):
    """Same normalization as unbake.evaluation.render.load_envmap: solid-angle weighted mean luminance."""
    w, h = image.size
    px = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    lum = px.reshape(h, w, 4)[..., :3].astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    solid_angle = np.cos((np.arange(h) + 0.5) / h * np.pi - np.pi / 2)[:, None]
    return ENV_MAP_MEAN_LUMINANCE / ((lum * solid_angle).sum() / (solid_angle.sum() * w))


def encode_video(frame_dirs, names, path):
    """Place the frame sequences side by side with labels and encode an H.264 mp4."""
    video = bpy.data.scenes.new('video')
    video.render.resolution_x, video.render.resolution_y = RESOLUTION * len(frame_dirs), RESOLUTION
    video.render.resolution_percentage = 100
    video.render.fps = FPS
    video.frame_start, video.frame_end = 1, FRAMES
    video.view_settings.view_transform = 'Standard'
    video.render.image_settings.file_format = 'FFMPEG'
    video.render.ffmpeg.format, video.render.ffmpeg.codec = 'MPEG4', 'H264'
    video.render.ffmpeg.constant_rate_factor = 'HIGH'
    video.render.filepath = path
    strips = video.sequence_editor_create().sequences
    n = len(frame_dirs)
    for i, (frames, name) in enumerate(zip(frame_dirs, names)):
        files = sorted(os.listdir(frames))
        strip = strips.new_image(name, os.path.join(frames, files[0]), channel=i + 1, frame_start=1,
                                 fit_method='ORIGINAL')
        for f in files[1:]:
            strip.elements.append(f)
        strip.transform.offset_x = (i - (n - 1) / 2) * RESOLUTION
        strip.blend_type = 'ALPHA_OVER'
        label = strips.new_effect(f'label_{name}', 'TEXT', channel=n + i + 1, frame_start=1, frame_end=FRAMES + 1)
        label.text = name
        label.font_size = max(16, RESOLUTION // 20)
        label.location = ((i + 0.5) / n, 0.04)
        label.align_x, label.align_y = 'CENTER', 'BOTTOM'
        label.use_shadow = True
    bpy.ops.render.render(animation=True, scene=video.name)


scene, mapping, env_node, bg = setup_scene()
names = [os.path.splitext(os.path.basename(p))[0].removesuffix('_2k') for p in EVAL_ENV_MAPS]
with tempfile.TemporaryDirectory() as tmp:
    frame_dirs = []
    for path, name in zip(EVAL_ENV_MAPS, names):
        env_node.image = bpy.data.images.load(os.path.abspath(path))
        bg.inputs['Strength'].default_value = luminance_scale(env_node.image)
        frame_dirs.append(os.path.join(tmp, name))
        for f in range(FRAMES):
            print(f'{name}: frame {f + 1}/{FRAMES}', flush=True)
            mapping.inputs['Rotation'].default_value[2] = 2 * math.pi * f / FRAMES
            scene.render.filepath = os.path.join(frame_dirs[-1], f'{f:04d}.png')
            bpy.ops.render.render(write_still=True)
    encode_video(frame_dirs, names, out_path)
print(f'Saved {out_path}')
