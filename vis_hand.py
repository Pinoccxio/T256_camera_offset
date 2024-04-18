import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

lines = np.array([
    # Thumb
    [1, 2], [2, 3],
    # Index
    [4, 5], [5, 6], [6, 7],
    # Middle
    [8, 9], [9, 10], [10, 11],
    # Ring
    [12, 13], [13, 14], [14, 15],
    # Little
    [16, 17], [17, 18], [18, 19],
    # Connections between proximals
    [1, 4], [4, 8], [8, 12], [12, 16],
    # connect palm
    [0, 1], [16, 0]
])

tip_length = {
    "thumb": 0.0286,
    "index": 0.0186,
    "middle": 0.0217,
    "ring": 0.0217,
    "little": 0.0157
}
Distal_map = {
    3: "thumb",
    7: "index",
    11: "middle",
    15: "ring",
    19: "little"
}


def create_or_update_cylinder(start, end, radius=0.003, cylinder_list=None, cylinder_idx=-1):
    # Calculate the length of the cylinder
    cyl_length = np.linalg.norm(end - start)

    # Create a new cylinder
    new_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=cyl_length, resolution=20, split=4)
    new_cylinder.paint_uniform_color([1, 0, 0])

    new_cylinder.translate(np.array([0, 0, cyl_length / 2]))

    # Compute the direction vector of the line segment and normalize it
    direction = end - start
    direction /= np.linalg.norm(direction)

    # Compute the rotation axis and angle
    up = np.array([0, 0, 1])  # Default up vector of cylinder
    rotation_axis = np.cross(up, direction)
    rotation_angle = np.arccos(np.dot(up, direction))

    # Compute the rotation matrix
    if np.linalg.norm(rotation_axis) != 0:
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        new_cylinder.rotate(rotation_matrix, center=np.array([0, 0, 0]))

    # Translate the new cylinder to the start position
    new_cylinder.translate(start)

    # Copy new cylinder to the original one if it exists
    return new_cylinder


import pickle as pkl

if __name__ == '__main__':

    viser = o3d.visualization.Visualizer()
    viser.create_window("out")

    # for d in data_list:
    #     id = d["id"]
    #     handtype = d["hand"]
    #     data[handtype][id] = np.concatenate([np.asarray(d["position"]),
    #                                          np.asarray(d["orientation"])[[1, 2, 3, 0]]])

    with open("../1.pkl", 'rb') as f:
        datas = pkl.load(f)
    ss = 0
    for data in datas:
        meshes = []
        for id in range(len(data["right"])):
            T = np.eye(4)
            T[:3, 3] = data["right"][id + 1][:3]
            T[:3, :3] = Rotation.from_quat(data["right"][id + 1][3:]).as_matrix()
            meshes.append(o3d.geometry.TriangleMesh().create_sphere(0.005).compute_vertex_normals().transform(T))
            meshes.append(o3d.geometry.TriangleMesh().create_coordinate_frame(0.01).transform(T))

            if id in Distal_map.keys():
                part = Distal_map[id]
                l = tip_length[part]
                start = T[:3,3]
                end = T[:3,3] + -T[:3,1] * l
                T2 = T.copy()
                T2[:3,3] = end
                meshes.append(create_or_update_cylinder(start, end))
                meshes.append(o3d.geometry.TriangleMesh().create_sphere(0.005).compute_vertex_normals().transform(T2))
        for i, (x, y) in enumerate(lines):
            start = data["right"][x + 1][:3]
            end = data["right"][y + 1][:3]
            meshes.append(create_or_update_cylinder(start, end))
        viser.clear_geometries()
        [viser.add_geometry(m) for m in meshes]

        if ss == 0:
            viser.run()
            para: o3d.visualization.ViewControl = viser.get_view_control()
            params = para.convert_to_pinhole_camera_parameters()
            ss = 1
        else:
            para: o3d.visualization.ViewControl = viser.get_view_control()
            para.convert_from_pinhole_camera_parameters(params)
            viser.poll_events()
            params = para.convert_to_pinhole_camera_parameters()
