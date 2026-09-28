"""Terrain curriculum covering every mjlab terrain family plus curbs."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains import config as presets
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)
from mjlab.terrains.utils import make_border, make_plane

_FLOOR_RGBA = (0.55, 0.55, 0.52, 1.0)
_CURB_RGBA = (0.85, 0.6, 0.25, 1.0)


@dataclass(kw_only=True)
class BoxHurdlesTerrainCfg(SubTerrainCfg):
    """Concentric curbs and door thresholds on level ground."""

    height_range: tuple[float, float] = (0.02, 0.15)
    """Curb height at difficulty 0 and 1, in metres. Must stay positive."""
    width_range: tuple[float, float] = (0.08, 0.4)
    spacing_range: tuple[float, float] = (0.7, 1.1)
    platform_width: float = 1.5
    border_width: float = 0.25

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")
        low, high = self.height_range
        height = low + difficulty * (high - low)
        center_x, center_y = self.size[0] / 2, self.size[1] / 2
        floor = make_plane(body, self.size, 0.0, center_zero=False)[0]
        geometries = [TerrainGeometry(geom=floor, color=_FLOOR_RGBA)]
        limit = min(self.size) / 2 - self.border_width
        radius = self.platform_width / 2
        while True:
            radius += rng.uniform(*self.spacing_range)
            width = rng.uniform(*self.width_range)
            if radius + width / 2 > limit:
                break
            curb_height = height * rng.uniform(0.8, 1.0)
            outer = 2 * radius + width
            inner = 2 * radius - width
            for geom in make_border(
                body,
                (outer, outer),
                (inner, inner),
                curb_height,
                (center_x, center_y, curb_height / 2),
            ):
                geometries.append(TerrainGeometry(geom=geom, color=_CURB_RGBA))
        return TerrainOutput(
            origin=np.array([center_x, center_y, 0.0]), geometries=geometries
        )


def terrain_generator_cfg() -> TerrainGeneratorCfg:
    """One curriculum column per terrain type, ten difficulty rows each."""
    stairs = dict(
        step_height_range=(0.02, 0.16),
        step_width=0.3,
        platform_width=2.5,
        border_width=1.0,
    )
    open_stairs = dict(
        step_height_range=(0.04, 0.15),
        step_width_range=(0.35, 0.7),
        platform_width=3.0,
    )
    pits = dict(floor_depth=2.0)
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        curriculum=True,
        sub_terrains={
            "flat": presets.flat(proportion=0.04),
            # Pyramids spawn on top, so the robot descends; inverted ones climb.
            "stairs_down": presets.pyramid_stairs(proportion=0.09, **stairs),
            "stairs_up": presets.pyramid_stairs_inv(
                proportion=0.09, **stairs
            ),
            "open_stairs_down": presets.open_stairs(
                proportion=0.04, **open_stairs
            ),
            "open_stairs_up": presets.open_stairs(
                proportion=0.04, inverted=True, **open_stairs
            ),
            "random_stairs": presets.random_stairs(
                proportion=0.05, step_height_range=(0.04, 0.15)
            ),
            "slope_down": presets.hf_pyramid_slope(
                proportion=0.07, slope_range=(0.0, 0.4)
            ),
            "slope_up": presets.hf_pyramid_slope_inv(
                proportion=0.07, slope_range=(0.0, 0.4)
            ),
            "rough": presets.random_rough(
                proportion=0.07,
                noise_range=(0.01, 0.07),
                noise_step=0.01,
                scale_with_difficulty=True,
            ),
            "waves": presets.wave_terrain(
                proportion=0.04, amplitude_range=(0.0, 0.15)
            ),
            "hills": presets.perlin_noise(
                proportion=0.06, height_range=(0.0, 0.6), resolution=0.1
            ),
            "obstacles": presets.discrete_obstacles(
                proportion=0.06, obstacle_height_range=(0.02, 0.15)
            ),
            "box_grid": presets.box_random_grid(
                proportion=0.05,
                grid_width=0.5,
                grid_height_range=(0.0, 0.08),
                merge_similar_heights=True,
            ),
            "scattered_boxes": presets.random_spread_boxes(
                proportion=0.05, box_height_range=(0.03, 0.2)
            ),
            "curbs": BoxHurdlesTerrainCfg(proportion=0.05),
            "tilted_tiles": presets.tilted_grid(
                proportion=0.04, tilt_range_deg=12.0, height_range=0.15, **pits
            ),
            "stepping_stones": presets.stepping_stones(
                proportion=0.04,
                stone_size_range=(0.45, 0.8),
                stone_distance_range=(0.1, 0.3),
                **pits,
            ),
            "beams": presets.narrow_beams(
                proportion=0.03, beam_width_range=(0.3, 0.8), **pits
            ),
            "rings": presets.nested_rings(
                proportion=0.03,
                ring_width_range=(0.4, 0.8),
                gap_range=(0.05, 0.35),
                **pits,
            ),
        },
        add_lights=True,
    )
