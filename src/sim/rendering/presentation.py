"""Opt-in rendering style for paper figures and live sim2sim screen captures.

Only appearance and camera settings are changed. Keep collision geometry,
inertia, joints, constraints, actuators and solver settings intact.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


def _element(parent, tag):
    element = parent.find(tag)
    return element if element is not None else ET.SubElement(parent, tag)


def expand_presentation_includes(root: ET.Element, source: Path) -> None:
    """Expose included robot visuals and keep assets valid in a temporary MJCF."""
    def expand(parent, directory, stack):
        for child in list(parent):
            if child.tag == "include":
                path = (directory / child.attrib["file"]).resolve()
                if path in stack:
                    raise ValueError(f"Cyclic MJCF include: {path}")
                included = ET.parse(path).getroot()
                expand(included, path.parent, stack | {path})
                index = list(parent).index(child)
                parent.remove(child)
                for offset, element in enumerate(list(included)):
                    parent.insert(index + offset, element)
            else:
                expand(child, directory, stack)

    expand(root, source.parent, {source.resolve()})
    compiler = _element(root, "compiler")
    for attribute in ("meshdir", "texturedir"):
        compiler.set(attribute, str((source.parent / compiler.get(attribute, ".")).resolve()))


def apply_paper_style(root: ET.Element) -> None:
    _element(root, "statistic").attrib.update(center="0 0 0.9", extent="3")
    visual = _element(root, "visual")
    _element(visual, "headlight").attrib.update(
        ambient="0.28 0.28 0.28", diffuse="0.45 0.45 0.45", specular="0.15 0.15 0.15"
    )
    _element(visual, "rgba").set("haze", "0.91 0.90 0.88 1")
    _element(visual, "quality").attrib.update(shadowsize="4096", offsamples="4")
    _element(visual, "map").attrib.update(
        haze="0.35", znear="0.003", zfar="100", shadowclip="2"
    )
    _element(visual, "global").attrib.update(
        fovy="38", offwidth="1920", offheight="1080"
    )

    # Official MJCF contains multiple asset/worldbody sections.
    for asset in root.findall("asset"):
        for texture in list(asset.findall("texture")):
            if texture.get("type") == "skybox":
                asset.remove(texture)
    asset = _element(root, "asset")
    ET.SubElement(asset, "texture", name="paper_sky", type="skybox", builtin="gradient",
                  rgb1="0.88 0.87 0.85", rgb2="0.98 0.97 0.95", width="512", height="3072")
    ET.SubElement(asset, "texture", name="paper_checker", type="2d", builtin="checker",
                  rgb1="0.91 0.90 0.88", rgb2="0.77 0.79 0.79", width="512", height="512")
    ET.SubElement(asset, "material", name="paper_floor", texture="paper_checker",
                  texuniform="true", texrepeat="1 1", reflectance="0",
                  specular="0.08", shininess="0.15")
    ET.SubElement(asset, "material", name="paper_shell", specular="0.28", shininess="0.45")

    for world in root.findall("worldbody"):
        for light in list(world.findall("light")):
            world.remove(light)
        for geom in world.findall("geom"):
            if geom.get("type") == "plane":
                geom.set("material", "paper_floor")
                geom.set("rgba", "1 1 1 1")
    world = _element(root, "worldbody")
    ET.SubElement(world, "light", name="paper_key", pos="-3 -4 7", dir="0.4 0.5 -1",
                  directional="true", diffuse="0.65 0.63 0.60", specular="0.2 0.2 0.2",
                  castshadow="true", bulbradius="0.15")
    ET.SubElement(world, "light", name="paper_fill", pos="3 2 5", dir="-0.5 -0.3 -1",
                  directional="true", diffuse="0.25 0.27 0.29", specular="0.05 0.05 0.05",
                  castshadow="false")

    for site in root.findall(".//site"):
        site.set("rgba", "0 0 0 0")

    for body in root.findall(".//body"):
        geoms = body.findall("geom")
        visual_meshes = {g.get("mesh") for g in geoms
                         if g.get("type") == "mesh" and g.get("contype") == "0"}
        for geom in geoms:
            mesh = geom.get("mesh", "")
            if geom.get("type") == "mesh":
                # Render one shell per mesh instead of overlapping visual/collision copies.
                if mesh in visual_meshes and geom.get("contype") != "0":
                    geom.set("rgba", "0 0 0 0")
                    continue
                dark = any(part in mesh for part in (
                    "hand", "wrist_yaw", "ankle_pitch", "hip_pitch", "Link-pitch",
                    "head_link", "logo_link",
                ))
                geom.set("rgba", "0.23 0.25 0.27 1" if dark else "0.72 0.74 0.75 1")
                geom.set("material", "paper_shell")
            elif geom.get("type") == "capsule" and geom.get("contype") == "0":
                geom.set("rgba", "0.25 0.27 0.28 1")
            else:
                # Hide diagnostic dots and simplified collision primitives, not their contacts.
                geom.set("rgba", "0 0 0 0")


def configure_presentation_camera(camera, model, config: dict) -> None:
    """Use a fixed world target by default; follow keeps the same viewing angle."""
    camera.azimuth = config.get("PRESENTATION_AZIMUTH", 135.0)
    camera.elevation = config.get("PRESENTATION_ELEVATION", -15.0)
    camera.distance = config.get("PRESENTATION_DISTANCE", 4.5)
    camera.lookat[:] = config.get("PRESENTATION_LOOKAT", (0.0, 0.0, 0.85))
    if config.get("PRESENTATION_CAMERA", "fixed") == "follow":
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = model.body("pelvis").id
    else:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.trackbodyid = -1
