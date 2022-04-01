"""Module which creates mesh lines from a line set
Open3D relies upon using glLineWidth to set line width on a LineSet
However, this method is now deprecated and not fully supporeted in newer OpenGL versions
See:
    Open3D Github Pull Request - https://github.com/intel-isl/Open3D/pull/738
    Other Framework Issues - https://github.com/openframeworks/openFrameworks/issues/3460

This module aims to solve this by converting a line into a triangular mesh (which has thickness)
The basic idea is to create a cylinder for each line segment, translate it, and then rotate it.

License: MIT

"""
import numpy as np
import open3d as o3d

try:
    from open3d.cuda.pybind.geometry import TriangleMesh
except ModuleNotFoundError:
    from open3d.cpu.pybind.geometry import TriangleMesh
except ImportError:
    from open3d.cpu.pybind.geometry import TriangleMesh

from scipy.spatial.transform import Rotation


def rotation_matrix_from_vectors(src=np.array([0., 0., 1.]), tgt=np.array([1., 0., 0.])):
    R = Rotation.align_vectors(tgt[None, :], src[None, :])[0].as_matrix()
    return R


# def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
#     """
#     Aligns vector a to vector b with axis angle rotation
#     """
#     if np.array_equal(a, b):
#         return None, None
#     axis_ = np.cross(a, b)
#     axis_ = axis_ / (np.linalg.norm(axis_) + 1e-6)
#     angle = np.arccos(np.dot(a, b))
#
#     return axis_, angle


# def normalized(a, axis=-1, order=2):
#     """Normalizes a numpy array of points"""
#     l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
#     l2[l2 == 0] = 1
#     return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[.1, .1, .1], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        z_axis = np.array([0., 0., 1.])
        # Create triangular mesh cylinder segments of line
        for ii in range(self.lines.shape[0]):
            P1, P2 = self.points[self.lines[ii, 0]], self.points[self.lines[ii, 1]]

            segment_length = np.sum((P2 - P1) ** 2) ** .5

            cylinder: TriangleMesh = TriangleMesh.create_cylinder(radius=self.radius, height=segment_length, split=1)

            color = self.colors if self.colors.ndim == 1 else self.colors[ii, :]
            cylinder.paint_uniform_color(color)

            R = rotation_matrix_from_vectors(src=z_axis, tgt=P2 - P1)
            cylinder.rotate(R)
            T = (P1 + P2) / 2
            cylinder.translate(T)

            # o3d.visualization.draw_geometries([cylinder])

            self.cylinder_segments.append(cylinder)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def main():
    print("Demonstrating LineMesh vs LineSet")
    # Create Line Set
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
              [0, 1, 1], [1, 1, 1]]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Create Line Mesh 1
    points = np.array(points) + [0, 0, 2]
    line_mesh1 = LineMesh(points, lines, colors, radius=0.02)
    line_mesh1_geoms = line_mesh1.cylinder_segments

    # Create Line Mesh 1
    points = np.array(points) + [0, 2, 0]
    line_mesh2 = LineMesh(points, radius=0.03)
    line_mesh2_geoms = line_mesh2.cylinder_segments

    o3d.visualization.draw_geometries(
        [line_set, *line_mesh1_geoms, *line_mesh2_geoms])


if __name__ == "__main__":
    main()
