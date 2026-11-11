"""
Access to the UnBake dataset (see download_dataset.py).

Layout: one folder per asset, named by its hash.

    data/<item id>/
        pbr.glb                 ground-truth asset with PBR materials
        meta.json               list of lighting conditions, one per baked asset:
                                {"name", "env_map", "rotation_deg", "exposure"}
        <name>.glb              the asset with that lighting baked into an (unlit) base color
                                texture; same geometry and UV layout as pbr.glb

    for item in UnbakeDataset("data"):
        gt = item.ground_truth()                  # Glb
        for lighting in item.lightings:
            baked = lighting.baked()              # Glb
"""
import json
import os
from dataclasses import dataclass

from .gltf_io import Glb


@dataclass(frozen=True)
class Lighting:
    """One lighting condition of an item and its baked-lighting asset."""
    name: str            # e.g. "golden_bay_2k_rot143"; also the file stem of the baked asset
    env_map: str         # env map file name, e.g. "golden_bay_2k.exr"
    rotation_deg: float  # env map rotation around the up axis
    exposure: float
    baked_path: str

    def baked(self) -> Glb:
        return Glb(self.baked_path)

    def env_map_path(self, env_map_dir: str = "env_maps") -> str:
        return os.path.join(env_map_dir, self.env_map)


@dataclass(frozen=True)
class DatasetItem:
    id: str
    path: str
    lightings: tuple[Lighting, ...]

    @property
    def ground_truth_path(self) -> str:
        return os.path.join(self.path, "pbr.glb")

    def ground_truth(self) -> Glb:
        return Glb(self.ground_truth_path)

    def lighting(self, name: str) -> Lighting:
        for lighting in self.lightings:
            if lighting.name == name:
                return lighting
        raise KeyError(f"item {self.id} has no lighting {name!r}")


class UnbakeDataset:
    def __init__(self, root: str = "data", ids: list[str] | None = None):
        """`ids` restricts the dataset to these items (default: every folder with a meta.json)."""
        self.root = root
        if ids is None:
            ids = sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "meta.json")))
        self.ids = list(ids)
        assert self.ids, f"no dataset items (folders with meta.json) found in {root}/"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int | str) -> DatasetItem:
        item_id = index if isinstance(index, str) else self.ids[index]
        path = os.path.join(self.root, item_id)
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        lightings = tuple(
            Lighting(
                name=m["name"],
                env_map=m["env_map"],
                rotation_deg=float(m["rotation_deg"]),
                exposure=float(m["exposure"]),
                baked_path=os.path.join(path, f"{m['name']}.glb"),
            )
            for m in meta
        )
        item = DatasetItem(id=item_id, path=path, lightings=lightings)
        missing = [p for p in (item.ground_truth_path, *(l.baked_path for l in lightings)) if not os.path.isfile(p)]
        assert not missing, f"item {item_id} is incomplete, missing: {', '.join(missing)}"
        return item

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
