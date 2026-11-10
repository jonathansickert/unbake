"""
Bake env-map lighting into PBR assets, producing "scanned-looking" assets with a single color texture.

For each asset and env map (randomly rotated), Cycles bakes the diffuse irradiance (direct + indirect
light, cast shadows, ambient occlusion) into the asset's own UVs. The color texture becomes
albedo * irradiance, exported as an unlit glTF material. Geometry and UVs are unchanged, so each
output pairs with its PBR input.

Run from the repo root:
    /Applications/Blender.app/Contents/MacOS/Blender -b -P data_toolkit/bake_diffuse.py -- baked_env env_maps

Reads <root>/<sha256>/pbr.glb, writes <env>_rot<deg>.glb and meta.json (env map, rotation, exposure)
into the same folder. Assets that already have a meta.json are skipped.
"""
import os
import sys
import glob
import json
import math
import random
import numpy as np
import bpy

RESOLUTION = 1024   # baked texture size per material
SAMPLES = 64        # irradiance samples per texel

scene = bpy.context.scene


def setup():
    scene.render.engine = 'CYCLES'
    scene.cycles.max_bounces = 4
    scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
    scene.render.bake.margin = 16
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for backend in ('OPTIX', 'CUDA', 'METAL'):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        if any(d.type == backend for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = 'GPU'
            break


def set_env_map(path, rotation_deg):
    world = scene.world or bpy.data.worlds.new('World')
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs['Rotation'].default_value[2] = math.radians(rotation_deg)
    env = nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(path, check_existing=True)
    bg = nodes.new('ShaderNodeBackground')
    links.new(nodes.new('ShaderNodeTexCoord').outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], env.inputs['Vector'])
    links.new(env.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs[0], nodes.new('ShaderNodeOutputWorld').inputs['Surface'])


def bake(mats, type_, shader_fn):
    """Temporarily route each material through shader_fn(mat, bsdf, surface) -> shader socket and bake.
    Returns {mat: (H, W, 4) float array}; alpha is 1 on baked texels, fractional in the margin."""
    temp = {}
    for mat in mats:
        nt = mat.node_tree
        out = nt.get_output_node('CYCLES')
        surface = out.inputs['Surface'].links[0].from_socket
        bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
        before = set(nt.nodes)
        nt.links.new(shader_fn(nt, bsdf, surface), out.inputs['Surface'])
        target = nt.nodes.new('ShaderNodeTexImage')
        target.image = bpy.data.images.new('bake', RESOLUTION, RESOLUTION, float_buffer=True, alpha=True)
        target.image.generated_color = (0, 0, 0, 0)
        nt.nodes.active = target
        temp[mat] = (surface, out, set(nt.nodes) - before)
    bpy.ops.object.bake(type=type_)
    result = {}
    for mat, (surface, out, nodes) in temp.items():
        target = mat.node_tree.nodes.active
        px = np.empty(RESOLUTION * RESOLUTION * 4, dtype=np.float32)
        target.image.pixels.foreach_get(px)
        px = px.reshape(RESOLUTION, RESOLUTION, 4)
        px[..., :3] /= np.maximum(px[..., 3:], 1e-6)   # margin texels come premultiplied by partial alpha
        result[mat] = px
        bpy.data.images.remove(target.image)
        for n in nodes:
            mat.node_tree.nodes.remove(n)
        mat.node_tree.links.new(surface, out.inputs['Surface'])
    return result


def emission_of(input_name):
    """Shader that emits a Principled input as-is (exact, lighting independent)."""
    def fn(nt, bsdf, surface):
        em = nt.nodes.new('ShaderNodeEmission')
        src = bsdf.inputs[input_name]
        if src.is_linked:
            nt.links.new(src.links[0].from_socket, em.inputs['Color'])
        else:
            v = src.default_value
            em.inputs['Color'].default_value = tuple(v) if hasattr(v, '__len__') else (v, v, v, 1)
        return em.outputs[0]
    return fn


def irradiance(nt, bsdf, surface):  # noqa: same signature as emission_of's shaders
    """Camera (bake) rays see white diffuse with the normal map, so metals get valid irradiance too;
    all other rays see the real material, keeping shadows and color bleeding correct."""
    diffuse = nt.nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.inputs['Color'].default_value = (1, 1, 1, 1)
    if bsdf.inputs['Normal'].is_linked:
        nt.links.new(bsdf.inputs['Normal'].links[0].from_socket, diffuse.inputs['Normal'])
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(nt.nodes.new('ShaderNodeLightPath').outputs['Is Camera Ray'], mix.inputs['Fac'])
    nt.links.new(surface, mix.inputs[1])
    nt.links.new(diffuse.outputs[0], mix.inputs[2])
    return mix.outputs[0]


def to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def export_unlit(meshes, mats, textures, path):
    """Swap every material for an unlit one showing its baked texture, export, swap back."""
    unlit = {}
    for mat in mats:
        name, mat.name = mat.name, mat.name + '_pbr'   # baked material takes over the original name
        u = bpy.data.materials.new(name)
        u.use_nodes = True
        nt = u.node_tree
        nt.nodes.clear()
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.image = textures[mat]
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        if tex.image.alpha_mode != 'NONE':
            u.blend_method = 'BLEND'
            mix = nt.nodes.new('ShaderNodeMixShader')
            nt.links.new(tex.outputs['Alpha'], mix.inputs['Fac'])
            nt.links.new(nt.nodes.new('ShaderNodeBsdfTransparent').outputs[0], mix.inputs[1])
            nt.links.new(tex.outputs['Color'], mix.inputs[2])
            nt.links.new(mix.outputs[0], out.inputs['Surface'])
        else:
            nt.links.new(tex.outputs['Color'], out.inputs['Surface'])   # -> KHR_materials_unlit
        unlit[mat] = u
    slots = [s for o in meshes for s in o.material_slots if s.material in unlit]
    originals = [s.material for s in slots]
    for s in slots:
        s.material = unlit[s.material]
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB')
    for s, m in zip(slots, originals):
        s.material = m
    for mat, u in unlit.items():
        name = u.name
        bpy.data.materials.remove(u)
        mat.name = name


def process(asset, env_maps, out_dir):
    global scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    setup()
    bpy.ops.import_scene.gltf(filepath=asset)
    meshes = [o for o in scene.objects if o.type == 'MESH' and o.data.uv_layers]
    mats = {s.material for o in meshes for s in o.material_slots
            if s.material and s.material.use_nodes
            and any(n.type == 'BSDF_PRINCIPLED' for n in s.material.node_tree.nodes)}
    meshes = [o for o in meshes if all(s.material in mats for s in o.material_slots)]
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]

    scene.cycles.samples = 1
    albedo = bake(mats, 'EMIT', emission_of('Base Color'))
    alpha = bake(mats, 'EMIT', emission_of('Alpha'))
    scene.cycles.samples = SAMPLES
    scene.render.bake.use_pass_direct = scene.render.bake.use_pass_indirect = True
    scene.render.bake.use_pass_color = True

    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(os.path.basename(out_dir))
    meta = []
    for env in env_maps:
        rotation = rng.uniform(0, 360)
        name = f'{os.path.splitext(os.path.basename(env))[0]}_rot{int(rotation):03d}'
        set_env_map(env, rotation)
        irr = bake(mats, 'DIFFUSE', irradiance)
        color = {m: albedo[m][..., :3] * irr[m][..., :3] for m in mats}
        # One exposure per bake so the 8-bit texture doesn't clip: 99.5th percentile of baked texels -> 1
        baked = np.concatenate([color[m][albedo[m][..., 3] > 0.99].ravel() for m in mats])
        exposure = 1.0 / max(float(np.percentile(baked, 99.5)), 1e-6)

        textures = {}
        for m in mats:
            covered = albedo[m][..., 3] > 0.99
            has_alpha = bool((alpha[m][..., 0][covered] < 0.99).any())
            rgba = np.concatenate([to_srgb(color[m] * exposure), alpha[m][..., :1]], axis=-1)
            img = bpy.data.images.new(m.name, RESOLUTION, RESOLUTION, alpha=has_alpha)
            img.alpha_mode = 'STRAIGHT' if has_alpha else 'NONE'
            img.pixels.foreach_set(rgba.astype(np.float32).ravel())
            img.pack()
            textures[m] = img
        export_unlit(meshes, mats, textures, os.path.join(out_dir, f'{name}.glb'))
        for img in textures.values():
            bpy.data.images.remove(img)
        meta.append({'name': name, 'env_map': os.path.basename(env), 'rotation_deg': rotation,
                     'exposure': exposure})

    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def env_maps_in(env_dir):
    return sorted(glob.glob(os.path.join(env_dir, '*.exr')) + glob.glob(os.path.join(env_dir, '*.hdr')))


if __name__ == '__main__':
    root, env_dir = sys.argv[sys.argv.index('--') + 1:][:2]
    for asset in sorted(glob.glob(os.path.join(root, '*', 'pbr.glb'))):
        if not os.path.exists(os.path.join(os.path.dirname(asset), 'meta.json')):
            print(f'Baking {asset}', flush=True)
            process(asset, env_maps_in(env_dir), os.path.dirname(asset))
