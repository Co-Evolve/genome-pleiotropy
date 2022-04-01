import atexit
import colorsys
import os
import time
from typing import List

import cv2
import numpy as np
import torch

from pleiotropy.environment.visualization_utils.linemesh import LineMesh

if torch.cuda.is_available():
    from open3d.cuda.pybind.geometry import TriangleMesh, LineSet
    from open3d.cuda.pybind.visualization import ViewControl, Visualizer
else:
    from open3d.cpu.pybind.geometry import TriangleMesh
    from open3d.cpu.pybind.visualization import ViewControl
import open3d as o3d


class GraphicsEngine3D:
    def __init__(self, params, base_path):
        self.params = params
        self.base_path = base_path if base_path is not None else ''
        self.path = f"{self.params['evaluation']['visualization']['save_dir']}/" \
                    f"{self.base_path.split('/')[-1]}"

        self.v2: Visualizer = None

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=True)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray(colorsys.hsv_to_rgb(224 / 360, .3, 1.))
        self.ctr: ViewControl = self.vis.get_view_control()
        self.total_shift = 0

        self.floor: List[TriangleMesh] = []
        n_stripes = 40
        for ii in range(n_stripes):
            self.floor.append(o3d.geometry.TriangleMesh.create_box(
                width=10 / n_stripes, height=.1, depth=10))
            self.floor[-1].translate((-5 + ii * (10 / n_stripes), -0.075, -5))

            c1 = colorsys.hsv_to_rgb(70 / 360, .2, .5)
            c2 = colorsys.hsv_to_rgb(70 / 360, .2, .475)

            if ii % 2 == 0:
                self.floor[-1].paint_uniform_color(c1)
            else:
                self.floor[-1].paint_uniform_color(c2)

            self.vis.add_geometry(self.floor[-1])

        self.bbot = TriangleMesh()
        self.vis.add_geometry(self.bbot)

        self.__init_camera()

        self.video_writer = None
        self.still_frame = None

        if params["evaluation"]["visualization"]["save"]:
            os.makedirs(params["evaluation"]["visualization"]["save_dir"], exist_ok=True)

    def __init_camera(self):
        self.ctr.set_lookat(np.array([0.5, 0., 0.5]))

        self.ctr.set_front(np.array([.33, .3, .33]))

        self.ctr.set_zoom(0.025)

    def __make_stillframe(self):
        L: LineSet = o3d.geometry.LineSet.create_from_triangle_mesh(self.bbot)
        lines, points = np.asarray(L.lines), np.asarray(L.points)
        vectors = points[lines[:, 1]] - points[lines[:, 0]]
        vlen = np.around(np.max(np.abs(vectors), axis=1), decimals=3)
        m = vlen >= self.params["environment"]["voxel_size"] * .6
        L.lines = o3d.utility.Vector2iVector(lines[m])

        self.v2 = o3d.visualization.Visualizer()

        self.v2.create_window("stillframe")

        self.v2.add_geometry(self.bbot)
        for mesh in LineMesh(points, L.lines, radius=.0005).cylinder_segments:
            self.v2.add_geometry(mesh)
        # self.v2.add_geometry(L)

        # render_options: RenderOption = self.v2.get_render_option()
        # render_options.mesh_show_wireframe = True

        ctr: ViewControl = self.v2.get_view_control()
        ctr.set_lookat(np.array([0.5, 0.09, 0.5]))
        ang = 0.35 * np.pi
        ctr.set_front(np.array([np.cos(-ang), 0., np.sin(-ang)]))

        ctr.camera_local_translate(forward=-0.04, right=0., up=-0.01)
        # ctr.set_zoom(1.5)
        self.v2.poll_events()
        img = (np.asarray(self.v2.capture_screen_float_buffer())[:, :, ::-1] * 255).astype(
            np.uint8)

        time.sleep(1)

        cv2.imwrite(f"{self.path}/still.jpg", img)
        self.v2.destroy_window()
        self.v2.close()

        return img

    def render(self, triangles: np.ndarray, normals: np.ndarray, colors: np.ndarray, x_shift):
        for tile in self.floor:
            tile.translate((-x_shift + self.total_shift, 0., 0.))
            self.vis.update_geometry(tile)
        self.total_shift = x_shift

        points = triangles.reshape((-1, 3))
        colors = np.stack((colors,) * 3, axis=1).reshape(-1, 3)

        # move camera to track bot
        middle = np.mean(points, axis=0)
        self.ctr.set_lookat(np.array([middle[0], 0.1, middle[2]]))

        # update position of bbot
        self.bbot.vertices = o3d.utility.Vector3dVector(points)
        self.bbot.triangles = o3d.utility.Vector3iVector(
            np.arange(points.shape[0]).reshape(-1, 3)[:, ::-1])
        self.bbot.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.bbot.compute_vertex_normals()
        self.bbot.triangle_normals = o3d.utility.Vector3dVector(normals)
        self.vis.update_geometry(self.bbot)

        self.vis.poll_events()
        if self.params["evaluation"]["visualization"]["save"]:

            os.makedirs(self.path, exist_ok=True)

            # on the first frame, make a picture of the bbot
            if self.still_frame is None:
                self.still_frame = self.__make_stillframe()
            else:
                # return

                img = np.asarray(self.vis.capture_screen_float_buffer())[:, :, ::-1]
                img *= 255
                img = img.astype(np.uint8)

                if self.video_writer is None:
                    path = f"{self.path}/video.mp4"
                    self.video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'),
                                                        60, img.shape[:2][::-1])
                    atexit.register(self.video_writer.release)

                self.video_writer.write(img)

    def shutdown(self):
        self.vis.destroy_window()
        # self.v2.destroy_window()
        self.vis.close()
        # self.v2.close()
        if self.video_writer is not None:
            self.video_writer.release()
