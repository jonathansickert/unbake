"""
Bake env-map lighting *including specular* into PBR assets via multi-view rendering + reprojection,
mimicking how photogrammetry scans end up with lighting baked into their textures.

For each asset and env map (same random rotations as bake_diffuse.py):
  1. Bake per-texel world position and normal into the asset's own UVs.
  2. Path trace the PBR asset from NUM_VIEWS cameras on a sphere (color + position pass).
  3. Project every texel into every view, keep it where the rendered position matches (visibility),
     and blend the views weighted by cos(view, normal)^SHARPNESS (0 = plain average, smeared highlights;
     large = close to a single best view, like texture mosaicking in photogrammetry tools).
  4. Texels no camera saw fall back to the diffuse bake (albedo * irradiance).
The result is exported as an unlit glTF; geometry and UVs are unchanged.

Run from the repo root:
    /Applications/Blender.app/Contents/MacOS/Blender -b -P data_toolkit/bake_multiview.py -- baked_env env_maps

Reads <root>/<sha256>/pbr.glb, writes <env>_rot<deg>_multiview.glb and meta_multiview.json into the same
folder. Assets that already have a meta_multiview.json are skipped.
"""
import os
import sys
import glob
import json
import math
import random
import tempfile
import numpy as np
import bpy
from mathutils import Vector
import OpenImageIO as oiio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bake_diffuse as B   # noqa: E402  (shared bake / env map / export helpers)

NUM_VIEWS = 32
RENDER_RES = 1024
SAMPLES = 128       # path tracing samples per view
SHARPNESS = 4.0     # view blending exponent on cos(view, normal)
FOV = 40.0
TEX_RES = B.RESOLUTION


def attribute_shader(kind, lo, ext):
    """Emission shader encoding world position (normalized to the bbox) or shading normal into [0, 1]."""
    def fn(nt, bsdf, surface):
        geo = nt.nodes.new('ShaderNodeNewGeometry')
        a = nt.nodes.new('ShaderNodeVectorMath')
        b = nt.nodes.new('ShaderNodeVectorMath')
        a.operation, b.operation = ('SUBTRACT', 'DIVIDE') if kind == 'position' else ('MULTIPLY', 'ADD')
        nt.links.new(geo.outputs['Position' if kind == 'position' else 'Normal'], a.inputs[0])
        a.inputs[1].default_value = lo if kind == 'position' else (0.5, 0.5, 0.5)
        nt.links.new(a.outputs['Vector'], b.inputs[0])
        b.inputs[1].default_value = ext if kind == 'position' else (0.5, 0.5, 0.5)
        em = nt.nodes.new('ShaderNodeEmission')
        nt.links.new(b.outputs['Vector'], em.inputs['Color'])
        return em.outputs[0]
    return fn


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = math.pi * (3 - math.sqrt(5)) * i
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)


def read_exr(path):
    inp = oiio.ImageInput.open(path)
    names = list(inp.spec().channelnames)
    px = inp.read_image(0, 0, 0, len(names), 'float')   # (H, W, C), row 0 = top
    inp.close()
    find = lambda layer, chans: px[..., [next(i for i, n in enumerate(names) if n.endswith(f'{layer}.{c}'))
                                        for c in chans]]
    return find('Combined', 'RGBA'), find('Position', 'XYZ')


def setup_render(scene):
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = RENDER_RES
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    scene.render.image_settings.color_depth = '32'
    scene.view_layers[0].use_pass_position = True


def process(asset, env_maps, out_dir):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = B.scene = bpy.context.scene
    B.setup()
    setup_render(scene)
    bpy.ops.import_scene.gltf(filepath=asset)
    meshes = [o for o in scene.objects if o.type == 'MESH' and o.data.uv_layers]
    mats = {s.material for o in meshes for s in o.material_slots
            if s.material and s.material.use_nodes
            and any(n.type == 'BSDF_PRINCIPLED' for n in s.material.node_tree.nodes)}
    meshes = [o for o in meshes if all(s.material in mats for s in o.material_slots)]
    mats = sorted(mats, key=lambda m: m.name)
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.context.view_layer.update()

    corners = np.array([list(o.matrix_world @ Vector(c)) for o in meshes for c in o.bound_box])
    lo, hi = corners.min(0), corners.max(0)
    ext = np.maximum(hi - lo, 1e-6)
    center, radius = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2

    # Per-texel attributes in the asset's own UVs (texels with alpha > 0 include the bake margin)
    scene.cycles.samples = 1
    albedo = B.bake(mats, 'EMIT', B.emission_of('Base Color'))
    alpha = B.bake(mats, 'EMIT', B.emission_of('Alpha'))
    pos = B.bake(mats, 'EMIT', attribute_shader('position', tuple(lo), tuple(ext)))
    nrm = B.bake(mats, 'EMIT', attribute_shader('normal', None, None))
    texels = {m: albedo[m][..., 3] > 0 for m in mats}
    P = np.concatenate([lo + pos[m][..., :3][texels[m]] * ext for m in mats])
    N = np.concatenate([nrm[m][..., :3][texels[m]] * 2 - 1 for m in mats])
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-6)
    split = np.cumsum([texels[m].sum() for m in mats])[:-1]
    covered_all = np.concatenate([albedo[m][..., 3][texels[m]] > 0.99 for m in mats])

    # Cameras on a sphere around the bounding sphere
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
    cam.data.angle = math.radians(FOV)
    dist = 1.05 * radius / math.sin(math.radians(FOV) / 2)
    cam.data.clip_start, cam.data.clip_end = dist * 0.01, dist * 4
    scene.collection.objects.link(cam)
    scene.camera = cam
    tan_half = math.tan(math.radians(FOV) / 2)
    res = RENDER_RES

    rng = random.Random(os.path.basename(out_dir))
    meta = []
    for env in env_maps:
        rotation = rng.uniform(0, 360)
        name = f'{os.path.splitext(os.path.basename(env))[0]}_rot{int(rotation):03d}_multiview'
        B.set_env_map(env, rotation)
        scene.cycles.samples = SAMPLES

        acc = np.zeros((len(P), 3))
        wsum = np.zeros(len(P))
        with tempfile.TemporaryDirectory() as tmp:
            for k, d in enumerate(fibonacci_sphere(NUM_VIEWS)):
                cam.location = Vector(center + d * dist)
                cam.rotation_euler = Vector(d).to_track_quat('Z', 'Y').to_euler()
                bpy.context.view_layer.update()
                scene.render.filepath = os.path.join(tmp, f'view{k:03d}.exr')
                bpy.ops.render.render(write_still=True)
                rgba, ppos = read_exr(scene.render.filepath)
                os.remove(scene.render.filepath)

                # Project texels: camera looks down -Z, x right, y up
                W2C = np.array(cam.matrix_world.inverted())
                pc = P @ W2C[:3, :3].T + W2C[:3, 3]
                depth = -pc[:, 2]
                front = depth > 1e-6
                sx = (pc[:, 0] / np.maximum(depth, 1e-6) / tan_half + 1) / 2 * res - 0.5
                sy = (1 - pc[:, 1] / np.maximum(depth, 1e-6) / tan_half) / 2 * res - 0.5
                x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
                fx, fy = sx - x0, sy - y0
                # Visibility: a bilinear tap counts if the rendered surface there is this texel's surface
                tol = 3 * depth * 2 * tan_half / res + 1e-4 * radius
                color = np.zeros((len(P), 3))
                cw = np.zeros(len(P))
                for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                                  (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                    xi, yi = x0 + dx, y0 + dy
                    ok = front & (xi >= 0) & (xi < res) & (yi >= 0) & (yi < res)
                    xi, yi = np.clip(xi, 0, res - 1), np.clip(yi, 0, res - 1)
                    ok &= rgba[yi, xi, 3] > 0.999
                    ok &= np.linalg.norm(ppos[yi, xi] - P, axis=1) < tol
                    w = np.where(ok, w, 0)
                    color += w[:, None] * rgba[yi, xi, :3]
                    cw += w
                seen = cw > 0.5
                view_dir = center + d * dist - P
                view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True)
                cos = np.clip((N * view_dir).sum(1), 0, 1)
                w = np.where(seen, cos ** SHARPNESS + 1e-8, 0)
                acc += w[:, None] * color / np.maximum(cw, 1e-6)[:, None]
                wsum += w
                print(f'  {name} view {k + 1}/{NUM_VIEWS}: {seen.mean():.1%} of texels visible', flush=True)

        radiance = acc / np.maximum(wsum, 1e-12)[:, None]
        unseen = wsum == 0
        unseen_frac = float(unseen[covered_all].mean())
        if unseen.any():   # fall back to the diffuse bake where no camera saw the texel
            scene.cycles.samples = B.SAMPLES
            scene.render.bake.use_pass_direct = scene.render.bake.use_pass_indirect = True
            scene.render.bake.use_pass_color = True
            irr = B.bake(mats, 'DIFFUSE', B.irradiance)
            diffuse = np.concatenate([(albedo[m][..., :3] * irr[m][..., :3])[texels[m]] for m in mats])
            radiance[unseen] = diffuse[unseen]

        # Same exposure normalization as bake_diffuse.py: 99.5th percentile of covered texels -> 1
        exposure = 1.0 / max(float(np.percentile(radiance[covered_all], 99.5)), 1e-6)
        textures = {}
        for m, rad in zip(mats, np.split(radiance, split)):
            img_rgb = np.zeros((TEX_RES, TEX_RES, 3), np.float32)
            img_rgb[texels[m]] = rad
            covered = albedo[m][..., 3] > 0.99
            has_alpha = bool((alpha[m][..., 0][covered] < 0.99).any())
            rgba_tex = np.concatenate([B.to_srgb(img_rgb * exposure), alpha[m][..., :1]], axis=-1)
            img = bpy.data.images.new(m.name, TEX_RES, TEX_RES, alpha=has_alpha)
            img.alpha_mode = 'STRAIGHT' if has_alpha else 'NONE'
            img.pixels.foreach_set(rgba_tex.astype(np.float32).ravel())
            img.pack()
            textures[m] = img
        B.export_unlit(meshes, mats, textures, os.path.join(out_dir, f'{name}.glb'))
        for img in textures.values():
            bpy.data.images.remove(img)
        meta.append({'name': name, 'env_map': os.path.basename(env), 'rotation_deg': rotation,
                     'exposure': exposure, 'num_views': NUM_VIEWS, 'sharpness': SHARPNESS,
                     'unseen_fraction': unseen_frac})

    with open(os.path.join(out_dir, 'meta_multiview.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    root, env_dir = sys.argv[sys.argv.index('--') + 1:][:2]
    for asset in sorted(glob.glob(os.path.join(root, '*', 'pbr.glb'))):
        if not os.path.exists(os.path.join(os.path.dirname(asset), 'meta_multiview.json')):
            print(f'Baking {asset}', flush=True)
            process(asset, B.env_maps_in(env_dir), os.path.dirname(asset))
