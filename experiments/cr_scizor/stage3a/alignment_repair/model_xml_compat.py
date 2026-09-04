"""Runtime-compatible restoration of an episode's frozen model XML.

The source XMLs declare several file-backed cube textures that MuJoCo 2.3.2
refuses because the texture PNG dimensions are not multiples of its cube-map
grid size.  Replay is headless, and this narrowly converts only those visual
texture declarations to 2-D while preserving every body, joint, geom, state,
and episode-specific model element.  The returned XML remains derived from and
is reset through the episode's own ``model_file``.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET


def prepare_model_for_reset(env, model_file: str | None) -> tuple[str | None, list[str]]:
    if model_file is None:
        return None, []
    xml = env.env.edit_model_xml(model_file)
    tree = ET.fromstring(xml)
    converted: list[str] = []
    for texture in tree.findall(".//texture"):
        if texture.get("type") == "cube" and texture.get("file"):
            converted.append(str(texture.get("name", "unnamed")))
            texture.set("type", "2d")
    return ET.tostring(tree, encoding="utf8").decode("utf8"), converted


def _task_models(task):
    models, seen = [], set()
    def add(model):
        if id(model) not in seen:
            seen.add(id(model)); models.append(model)
    for model in getattr(task, "mujoco_objects", []): add(model)
    for robot in getattr(task, "mujoco_robots", []):
        add(robot)
        for model in getattr(robot, "models", []): add(model)
    return models


def _filtered_model_names(model, field: str, available: set[str]) -> list[str]:
    """Return model names present in the loaded XML without mutating model properties.

    Robosuite exposes these lists as read-only properties which add a naming
    prefix to the corresponding private lists.  Filtering the private list is
    possible but would change the live task after XML restoration.  The task
    ID mapping implementation only needs the filtered names for this one call,
    so build them directly here instead.
    """
    values = list(getattr(model, field, []) or [])
    return [value for value in values if value in available]


def _temporarily_filter_private_names(model, field: str, available: set[str], snapshots: list[tuple[object, str, object]]) -> None:
    """Filter a model property by changing its backing list for one reset call."""
    private_field = f"_{field}"
    if not hasattr(model, private_field):
        return
    values = getattr(model, private_field)
    if not isinstance(values, list):
        return
    prefix = getattr(model, "naming_prefix", "")
    filtered = [
        value
        for value in values
        if (value if value.startswith(prefix) else f"{prefix}{value}") in available
    ]
    snapshots.append((model, private_field, values))
    setattr(model, private_field, filtered)


def _filter_robot_model_sites_for_loaded_xml(env) -> list[tuple[object, str, object]]:
    """Limit live robot bookkeeping lists to sites present in the frozen XML."""
    site_names = {name for name in env.env.sim.model.site_names if name is not None}
    snapshots: list[tuple[object, str, object]] = []
    for robot in env.env.robots:
        gripper = getattr(robot, "gripper", None)
        if gripper is not None:
            _temporarily_filter_private_names(gripper, "sites", site_names, snapshots)
    return snapshots


def reset_to_episode_model(env, states, model_file: str | None) -> list[str]:
    """Reset through the episode XML while accommodating old robot visual names.

    Robosuite 1.4's current Panda model renamed visual geoms relative to the
    frozen demonstration XML.  During the one model reload, constrain ID
    mapping to names actually present in the just-loaded episode simulator.
    This affects bookkeeping only; dynamics are defined by the source XML and
    the supplied flattened state.
    """
    runtime_model, texture_patch = prepare_model_for_reset(env, model_file)
    payload = {"states": states}
    if runtime_model is not None:
        payload["model"] = runtime_model
    # ``EnvRobosuite.reset_to`` performs a hard reset before loading the XML.
    # That reset recreates ``env.env.model``, so an instance-level override is
    # discarded.  Patch the Task class for the complete reset sequence instead.
    from robosuite.models.tasks.task import Task

    original = Task.generate_id_mappings
    snapshots: list[tuple[object, str, object]] = []

    def compatible_generate_id_mappings(self, sim):
        snapshot_start = len(snapshots)
        geom_names = {name for name in sim.model.geom_names if name is not None}
        site_names = {name for name in sim.model.site_names if name is not None}
        for model in _task_models(self):
            _temporarily_filter_private_names(model, "visual_geoms", geom_names, snapshots)
            _temporarily_filter_private_names(model, "contact_geoms", geom_names, snapshots)
            _temporarily_filter_private_names(model, "sites", site_names, snapshots)
        # Keep the old task implementation's outputs, but calculate them from
        # names that exist in the frozen XML. This is a model-ID bookkeeping
        # adaptation only: it does not modify XML, flattened state, or actions.
        from robosuite.utils.mjcf_utils import get_ids

        self._instances_to_ids = {}
        self._geom_ids_to_instances = {}
        self._site_ids_to_instances = {}
        self._classes_to_ids = {}
        self._geom_ids_to_classes = {}
        self._site_ids_to_classes = {}
        for model in _task_models(self):
            cls = str(type(model)).split("'")[1].split(".")[-1]
            inst = model.name
            id_groups = [
                get_ids(sim=sim, elements=_filtered_model_names(model, "visual_geoms", geom_names) + _filtered_model_names(model, "contact_geoms", geom_names), element_type="geom"),
                get_ids(sim=sim, elements=_filtered_model_names(model, "sites", site_names), element_type="site"),
            ]
            group_types = ("geom", "site")
            ids_to_instances = (self._geom_ids_to_instances, self._site_ids_to_instances)
            ids_to_classes = (self._geom_ids_to_classes, self._site_ids_to_classes)
            assert inst not in self._instances_to_ids, f"Instance {inst} already registered; should be unique"
            self._instances_to_ids[inst] = {}
            if cls not in self._classes_to_ids:
                self._classes_to_ids[cls] = {group_type: [] for group_type in group_types}
            for ids, group_type, ids_to_inst, ids_to_cls in zip(id_groups, group_types, ids_to_instances, ids_to_classes):
                self._instances_to_ids[inst][group_type] = ids
                self._classes_to_ids[cls][group_type] += ids
                for idn in ids:
                    assert idn not in ids_to_inst, f"ID {idn} already registered; should be unique"
                    ids_to_inst[idn] = inst
                    ids_to_cls[idn] = cls

    Task.generate_id_mappings = compatible_generate_id_mappings
    try:
        env.reset_to(payload)
        # ``reset_from_xml_string`` restores the task model, but robot
        # wrappers persist separately.  Their gripper visualisation lists use
        # current-version site names, so retain the source-XML-compatible list
        # for later calls to ``env.step`` as well.
        snapshots.extend(_filter_robot_model_sites_for_loaded_xml(env))
    finally:
        Task.generate_id_mappings = original
    return texture_patch
