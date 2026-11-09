"""
Minimal GLB reading/writing that keeps the input file's structure intact.

Reading flattens the scene graph into world-space triangle primitives (with their UVs and
material). Writing patches the materials of the ORIGINAL file in place, so nodes, meshes,
vertex data and the UV layout stay byte-identical; only textures/material params change.
"""
import io
import json
import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image

_GLB_MAGIC = b"glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

_COMPONENT_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


@dataclass
class Primitive:
    mesh_index: int
    primitive_index: int
    material: int | None
    positions: np.ndarray  # (V, 3) float64, world space (glTF Y-up)
    uvs: np.ndarray | None  # (V, 2) float32, glTF convention (origin top-left)
    faces: np.ndarray  # (F, 3) int64


class Glb:
    def __init__(self, path: str):
        data = open(path, "rb").read()
        magic, version, _ = struct.unpack("<4sII", data[:12])
        assert magic == _GLB_MAGIC and version == 2, f"{path} is not a glTF 2.0 binary (.glb)"
        offset, self.json, self.bin = 12, None, b""
        while offset < len(data):
            length, ctype = struct.unpack("<II", data[offset:offset + 8])
            chunk = data[offset + 8:offset + 8 + length]
            if ctype == _CHUNK_JSON:
                self.json = json.loads(chunk)
            elif ctype == _CHUNK_BIN:
                self.bin = chunk
            offset += 8 + length
        assert len(self.json.get("buffers", [])) <= 1 and "uri" not in self.json.get("buffers", [{}])[0], \
            "only self-contained .glb files (single embedded buffer) are supported"

    # ---------- reading ----------
    def buffer_view_bytes(self, index: int) -> bytes:
        bv = self.json["bufferViews"][index]
        start = bv.get("byteOffset", 0)
        return self.bin[start:start + bv["byteLength"]]

    def accessor(self, index: int) -> np.ndarray:
        acc = self.json["accessors"][index]
        assert "sparse" not in acc, "sparse accessors are not supported"
        dtype = np.dtype(_COMPONENT_DTYPES[acc["componentType"]])
        ncomp = _TYPE_SIZES[acc["type"]]
        bv = self.json["bufferViews"][acc["bufferView"]]
        stride = bv.get("byteStride", ncomp * dtype.itemsize)
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        raw = np.frombuffer(self.bin, dtype=np.uint8, count=stride * (acc["count"] - 1) + ncomp * dtype.itemsize, offset=start)
        out = np.lib.stride_tricks.as_strided(raw, shape=(acc["count"], ncomp * dtype.itemsize), strides=(stride, 1))
        out = out.copy().view(dtype).reshape(acc["count"], ncomp)
        if acc.get("normalized"):  # normalized integer UVs etc.
            out = out.astype(np.float32) / np.iinfo(dtype).max
        return out

    def image(self, index: int) -> Image.Image:
        return Image.open(io.BytesIO(self.buffer_view_bytes(self.json["images"][index]["bufferView"])))

    def material_base_color_image(self, material: int | None) -> int | None:
        """Image index of a material's baseColorTexture (the baked texture), if any."""
        if material is None:
            return None
        tex = self.json["materials"][material].get("pbrMetallicRoughness", {}).get("baseColorTexture")
        return None if tex is None else self.json["textures"][tex["index"]].get("source")

    def world_primitives(self) -> list[Primitive]:
        """All triangle primitives of the default scene, transformed to world space."""
        out = []

        def visit(node_index, parent):
            node = self.json["nodes"][node_index]
            world = parent @ _node_matrix(node)
            if "mesh" in node:
                for pi, prim in enumerate(self.json["meshes"][node["mesh"]]["primitives"]):
                    assert prim.get("mode", 4) == 4, "only triangle primitives are supported"
                    pos = self.accessor(prim["attributes"]["POSITION"]).astype(np.float64)
                    pos = (np.c_[pos, np.ones(len(pos))] @ world.T)[:, :3]
                    if "indices" in prim:
                        faces = self.accessor(prim["indices"]).reshape(-1, 3).astype(np.int64)
                    else:
                        faces = np.arange(len(pos)).reshape(-1, 3)
                    if np.linalg.det(world[:3, :3]) < 0:  # mirrored transform flips winding
                        faces = faces[:, ::-1].copy()
                    uv = prim["attributes"].get("TEXCOORD_0")
                    out.append(Primitive(node["mesh"], pi, prim.get("material"), pos,
                                         None if uv is None else self.accessor(uv).astype(np.float32), faces))
            for child in node.get("children", []):
                visit(child, world)

        scene = self.json["scenes"][self.json.get("scene", 0)]
        for root in scene["nodes"]:
            visit(root, np.eye(4))
        return out

    # ---------- writing ----------
    def set_pbr_textures(self, material: int, base_color: Image.Image, metallic_roughness: Image.Image) -> None:
        """
        Make `material` a plain metallic-roughness PBR material using the given textures.
        Reuses the material's existing texture sampler; drops KHR_materials_unlit.
        """
        mat = self.json["materials"][material]
        old_tex = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        sampler = self.json["textures"][old_tex["index"]].get("sampler") if old_tex else None
        tex_coord = old_tex.get("texCoord", 0) if old_tex else 0

        def add_texture(img: Image.Image, name: str) -> int:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            self.json.setdefault("bufferViews", []).append({"buffer": 0, "byteLength": len(buf.getvalue()), "_data": buf.getvalue()})
            self.json.setdefault("images", []).append({"bufferView": len(self.json["bufferViews"]) - 1, "mimeType": "image/png", "name": name})
            tex = {"source": len(self.json["images"]) - 1}
            if sampler is not None:
                tex["sampler"] = sampler
            self.json.setdefault("textures", []).append(tex)
            return len(self.json["textures"]) - 1

        name = mat.get("name", f"material{material}")
        mat["pbrMetallicRoughness"] = {
            "baseColorTexture": {"index": add_texture(base_color, f"{name}_baseColor"), "texCoord": tex_coord},
            "metallicRoughnessTexture": {"index": add_texture(metallic_roughness, f"{name}_metallicRoughness"), "texCoord": tex_coord},
            "metallicFactor": 1.0,
            "roughnessFactor": 1.0,
        }
        mat.get("extensions", {}).pop("KHR_materials_unlit", None)
        if not mat.get("extensions"):
            mat.pop("extensions", None)

    def save(self, path: str) -> None:
        j = self.json
        # Drop images/textures no material references any more (e.g. the baked texture)
        self._prune_unused_textures()
        used_ext = {k for m in j.get("materials", []) for k in m.get("extensions", {})}
        for key in ("extensionsUsed", "extensionsRequired"):
            if key in j:
                j[key] = [e for e in j[key] if e != "KHR_materials_unlit" or e in used_ext]
                if not j[key]:
                    del j[key]
        # Rebuild the binary chunk from the buffer views (4-byte aligned)
        new_bin = bytearray()
        for bv in j.get("bufferViews", []):
            data = bv.pop("_data", None)
            if data is None:
                data = self.bin[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
            new_bin += b"\0" * (-len(new_bin) % 4)
            bv["byteOffset"] = len(new_bin)
            new_bin += data
        new_bin += b"\0" * (-len(new_bin) % 4)
        j["buffers"] = [{"byteLength": len(new_bin)}]
        json_bytes = json.dumps(j, separators=(",", ":")).encode()
        json_bytes += b" " * (-len(json_bytes) % 4)
        total = 12 + 8 + len(json_bytes) + 8 + len(new_bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", _GLB_MAGIC, 2, total))
            f.write(struct.pack("<II", len(json_bytes), _CHUNK_JSON) + json_bytes)
            f.write(struct.pack("<II", len(new_bin), _CHUNK_BIN) + bytes(new_bin))

    def _prune_unused_textures(self) -> None:
        j = self.json
        tex_refs = []  # every {"index": t} dict that points at a texture
        for m in j.get("materials", []):
            for holder in (m, m.get("pbrMetallicRoughness", {})):
                for key, val in holder.items():
                    if isinstance(val, dict) and "index" in val and key.endswith("Texture"):
                        tex_refs.append(val)
        used_tex = sorted({r["index"] for r in tex_refs})
        tex_map = {old: new for new, old in enumerate(used_tex)}
        for r in tex_refs:
            r["index"] = tex_map[r["index"]]
        j["textures"] = [j["textures"][t] for t in used_tex]

        used_img = sorted({t["source"] for t in j["textures"] if "source" in t})
        img_map = {old: new for new, old in enumerate(used_img)}
        for t in j["textures"]:
            if "source" in t:
                t["source"] = img_map[t["source"]]
        j["images"] = [j["images"][i] for i in used_img]

        # Drop buffer views only the removed images used (accessors keep theirs)
        keep_bv = {a["bufferView"] for a in j.get("accessors", []) if "bufferView" in a}
        keep_bv |= {im["bufferView"] for im in j["images"] if "bufferView" in im}
        bv_order = [i for i in range(len(j.get("bufferViews", []))) if i in keep_bv]
        bv_map = {old: new for new, old in enumerate(bv_order)}
        j["bufferViews"] = [j["bufferViews"][i] for i in bv_order]
        for a in j.get("accessors", []):
            if "bufferView" in a:
                a["bufferView"] = bv_map[a["bufferView"]]
        for im in j["images"]:
            if "bufferView" in im:
                im["bufferView"] = bv_map[im["bufferView"]]
        for key in ("textures", "images", "samplers"):
            if key in j and not j[key]:
                del j[key]


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # column-major
    t = np.array(node.get("translation", [0, 0, 0]), dtype=np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = r * s
    m[:3, 3] = t
    return m
