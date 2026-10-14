"""Simulator odometry message; preserves the SONIC DDS type and field order."""

from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

@dataclass
@annotate.final
@annotate.autoid("sequential")
class OdoState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.OdoState_"):
    version: types.array[types.uint32, 2]
    tick: types.uint32
    position: types.array[types.float32, 3]
    orientation: types.array[types.float32, 4]  # quaternion [x, y, z, w]
    linear_velocity: types.array[types.float32, 3]
    angular_velocity: types.array[types.float32, 3]
    crc: types.uint32


def default_odometry() -> OdoState_:
    """Create the odometry sample used on rt/odostate."""
    return OdoState_(
        [0, 0], 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0,
    )
