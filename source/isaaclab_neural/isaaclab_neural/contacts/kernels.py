import numpy as np
import warp as wp
import newton
from newton import GeoType, ShapeFlags
from newton._src.geometry.kernels import get_box_vertex

@wp.kernel(enable_backward=False)
def generate_contact_points(
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_margin: wp.array(dtype=float),
    shape_flags: wp.array(dtype=wp.int32),
    num_shapes_per_env: int,
    num_contacts_per_env: int,
    ground_shape_index: int,
    shapes_start_offset: int,
    up_vector: wp.vec3,
    # outputs
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_thickness0: wp.array(dtype=float),
    contact_thickness1: wp.array(dtype=float),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth: wp.array(dtype=float),
):
    """Generate contact points for a given environment, assuming each shape is a regular geometric shape."""
    
    env_id = wp.tid()

    shape_offset = num_shapes_per_env * env_id + shapes_start_offset
    contact_idx = num_contacts_per_env * env_id
    for i in range(num_shapes_per_env):
        body = shape_body[shape_offset + i]

        if body == -1:
            # static shapes are ignored, e.g. ground
            continue

        if shape_flags[shape_offset + i] & ShapeFlags.COLLIDE_SHAPES == 0:
            # filter out visual meshes
            continue

        geo_type = shape_type[shape_offset + i]
        geo_scale = shape_scale[shape_offset + i]
        geo_margin = shape_margin[shape_offset + i]
        shape_tf = shape_transform[shape_offset + i]

        if geo_type == GeoType.SPHERE:
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_get_translation(shape_tf)
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness0[contact_idx] = geo_margin + geo_scale[0]
            contact_thickness1[contact_idx] = 0.0
            contact_idx += 1

        if geo_type == GeoType.CAPSULE:
            # add points at the two ends of the capsule
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, 0.0,geo_scale[1])
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness0[contact_idx] = geo_margin + geo_scale[0]
            contact_thickness1[contact_idx] = 0.0
            contact_idx += 1

            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, 0.0, -geo_scale[1])
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness0[contact_idx] = geo_margin + geo_scale[0]
            contact_thickness1[contact_idx] = 0.0
            contact_idx += 1

        if geo_type == GeoType.BOX:
            # add box corner points
            for j in range(8):
                p = get_box_vertex(j, geo_scale)
                contact_shape0[contact_idx] = shape_offset + i
                contact_shape1[contact_idx] = ground_shape_index
                contact_point0[contact_idx] = wp.transform_point(shape_tf, p)
                contact_point1[contact_idx] = wp.vec3(0.0)
                contact_normal[contact_idx] = up_vector
                contact_depth[contact_idx] = 1000.0
                contact_thickness0[contact_idx] = geo_margin
                contact_thickness1[contact_idx] = 0.0
                contact_idx += 1

        if geo_type == GeoType.CYLINDER:
            # Capsule treatment: 2 anchors at the centerline endpoints, with
            # thickness = margin + radius. See docs/design/CYLINDER_CONTACT_FIX.md.
            # NOTE: upright cylinders under-report clearance by r (endpoint
            # disks are treated as hemispheres).
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, 0.0, geo_scale[1])
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness0[contact_idx] = geo_margin + geo_scale[0]
            contact_thickness1[contact_idx] = 0.0
            contact_idx += 1

            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, 0.0, -geo_scale[1])
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness0[contact_idx] = geo_margin + geo_scale[0]
            contact_thickness1[contact_idx] = 0.0
            contact_idx += 1

        # COM fallback for mesh / unknown geo types
        if (
            geo_type != GeoType.BOX
            and geo_type != GeoType.CAPSULE
            and geo_type != GeoType.SPHERE
            and geo_type != GeoType.CYLINDER
        ):
            contact_shape0[contact_idx] = shape_offset + i
            contact_shape1[contact_idx] = ground_shape_index
            contact_point0[contact_idx] = wp.transform_point(
                shape_tf, wp.vec3(0.0, 0.0, 0.0)
            )
            contact_point1[contact_idx] = wp.vec3(0.0)
            contact_normal[contact_idx] = up_vector
            contact_depth[contact_idx] = 1000.0
            contact_thickness0[contact_idx] = 0.0
            contact_thickness1[contact_idx] = 0.0
            contact_idx += 1


@wp.kernel(enable_backward=False)
def collision_detection_ground_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    ground_shape_index: int,
    contact_shape0: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    ground_up_vec: wp.vec3,
    # outputs
    contact_shape1: wp.array(dtype=int),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth: wp.array(dtype=float),
):
    """Collision detection against the flat ground plane, only for anchors generated by generate_contact_points."""

    contact_id = wp.tid()
    shape = contact_shape0[contact_id]
    ground_tf = shape_transform[ground_shape_index]
    body = shape_body[shape]
    point_world = wp.transform_point(body_q[body], contact_point0[contact_id])

    # contact shape1 is always ground
    contact_shape1[contact_id] = ground_shape_index

    # get contact normal in world frame
    # ground_up_vec = wp.vec3(0.0, 0.0, 1.0)
    # NOTE: in newton, contact normal points from shape 0 to shape 1
    # ground_up_vec = wp.vec3(0.0, 0.0, -1.0)

    contact_normal[contact_id] = wp.transform_vector(ground_tf, ground_up_vec)

    # transform point to ground shape frame
    T_world_to_ground = wp.transform_inverse(ground_tf)
    point_plane = wp.transform_point(T_world_to_ground, point_world)

    # get contact depth
    contact_depth[contact_id] = wp.dot(point_plane, ground_up_vec)

    # project to plane
    projected_point = point_plane - contact_depth[contact_id] * ground_up_vec

    # transform to world frame (applying the shape transform)
    # contact_point1 is supposed to be in shape 1 frame, here it is in world frame as a special case for ground plane
    contact_point1[contact_id] = wp.transform_point(ground_tf, projected_point)