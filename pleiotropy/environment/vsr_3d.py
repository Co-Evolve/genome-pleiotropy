from copy import copy
from typing import Dict

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
from matplotlib import cm

from pleiotropy.environment.visualization_utils.graphics3D import GraphicsEngine3D

real = ti.f32

dim = 3
# this will be overwritten
n_grid = 64
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2e-3
# TODO: update
mu = 10
la = 10
gravity = 10
act_strength = 5
visualize_resolution = 256
bound = 3
coeff = 1.5


def scalar():
    return ti.field(dtype=real)


def vec():
    return ti.Vector.field(dim, dtype=real)


def mat():
    return ti.Matrix.field(dim, dim, dtype=real)


def zero_vec():
    return [0.0, 0.0, 0.0]


def zero_matrix():
    return [zero_vec(), zero_vec(), zero_vec()]


class SceneVisuals:
    def __init__(self, params, base_path):
        self.params = params

        self.voxels = {}
        self.faces: np.ndarray = np.empty(0)
        self.masks: np.ndarray = np.empty(0)

        self.graphics_engine = GraphicsEngine3D(params, base_path)

        self.first_frame = True

    def find_faces_for_rendering(self, w, scene):
        dX = w / int(w / dx * 3)
        w = int(round(w * 100))
        # figure out which faces to render
        faces = []
        for voxel, E in self.voxels.items():
            x, y, z = voxel
            if (x - w, y, z) not in self.voxels:  # left
                faces.append((E[0], E[2], E[3], E[1]))
            if (x + w, y, z) not in self.voxels:  # right
                faces.append((E[4], E[5], E[7], E[6]))
            if (x, y - w, z) not in self.voxels:  # down
                faces.append((E[0], E[1], E[5], E[4]))
            if (x, y + w, z) not in self.voxels:  # up
                faces.append((E[2], E[6], E[7], E[3]))
            if (x, y, z - w) not in self.voxels:  # back
                faces.append((E[0], E[4], E[6], E[2]))
            if (x, y, z + w) not in self.voxels:  # forward
                faces.append((E[1], E[3], E[7], E[5]))
        self.faces = np.array(faces)
        self.merge_voxel_corners(dX, copy(scene.x[0]))

    def merge_voxel_corners(self, dX, X):
        X = np.array(X) / dX

        II = self.faces.reshape(-1)
        JJ = np.array(sorted(list(set(II.tolist()))))

        Linf = np.max(np.abs(X[JJ, None] - X[None, JJ]), axis=2)
        Linf = np.rint(Linf).astype(int)
        neighbors_JJ = np.argwhere(Linf == 1)

        merge = {jj: [jj] for jj in JJ}
        for n in neighbors_JJ:
            merge[JJ[n[0]]].append(JJ[n[1]])

        N = max(len(n) for n in merge.values())
        indices = np.array([merge[ii] + [0, ] * (N - len(merge[ii])) for ii in II])
        mask = np.array(
            [[True, ] * len(merge[ii]) + [False, ] * (N - len(merge[ii])) for ii in II])

        shape = (*self.faces.shape, N)
        self.faces = indices.reshape(shape)
        self.masks = mask.reshape(shape)

    def render(self, X, A, x_shift, color=None):
        corners = np.sum(X[self.faces] * self.masks[:, :, :, None], axis=2) / np.sum(self.masks,
                                                                                     axis=2)[:, :,
                                                                              None]
        middles = np.mean(corners, axis=1)
        tri = [np.stack((corners[:, ii], corners[:, (ii + 1) % 4], middles[:]), axis=1) for ii in
               range(4)]
        tri = np.stack(tri, axis=0)
        del corners, middles

        normals = np.cross(tri[:, :, 0] - tri[:, :, 2], tri[:, :, 1] - tri[:, :, 2])
        normals = np.mean(normals, axis=0)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals = np.stack((normals,) * 4, axis=0)

        A = np.mean(A[self.faces[:, :, 0]], axis=1) + .5
        cmap = plt.get_cmap('coolwarm')
        cmap = cm.ScalarMappable(matplotlib.colors.Normalize(vmin=0., vmax=1.), cmap=cmap)

        rgb = cmap.to_rgba(1 - A)[:, :-1]
        hsv = matplotlib.colors.rgb_to_hsv(rgb)
        hsv[:, 1] = hsv[:, 1] * 1
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        if color is not None:
            color = color.astype(float) / 255 if color.dtype == np.uint8 else color
            rgb = np.stack((color,) * rgb.shape[0], axis=0)

        rgb = np.stack((rgb,) * 4, axis=0)

        self.graphics_engine.render(tri.reshape((-1, 3, 3)), normals.reshape((-1, 3)),
                                    rgb.reshape(-1, 3), x_shift)


class Scene:
    def __init__(self, params, base_path=None, use_graphics=False):
        self.n_particles = 0
        self.use_graphics = use_graphics
        self.calculate_balance = params['experiment']['balanced_locomotion']
        if self.use_graphics:
            self.visuals = SceneVisuals(params, base_path)

        self.max_n_particles = 27 * 125

        self.x = np.zeros((2, self.max_n_particles, 3), dtype=np.float32)
        self.v = np.zeros((2, self.max_n_particles, 3), dtype=np.float32)
        self.actuator_id = np.zeros(self.max_n_particles, dtype=np.int32)
        self.C = np.zeros((2, self.max_n_particles, 3, 3), dtype=np.float32)
        self.F = np.zeros((2, self.max_n_particles, 3, 3), dtype=np.float32)
        self.F[:, :, [0, 1, 2], [0, 1, 2]] = 1.

        self.offset_x = 0
        self.offset_y = 0
        self.offset_z = 0
        self.num_actuators = 0
        self.faces = None
        self.ht_x, self.ht_y, self.ht_z = None, None, None
        self.head_pos = None
        self.head_particles_indices = []

    def add_rect(self, x, y, z, w, h, d, actuation, ptype=1):
        density = 3
        w_count = int(w / dx * density)
        h_count = int(h / dx * density)
        d_count = int(d / dx * density)
        real_dx = w / w_count
        real_dy = h / h_count
        real_dz = d / d_count

        if self.calculate_balance:
            if self.head_pos is None:
                self.head_pos = (x, y, z)
            else:
                h_x, h_y, h_z = self.head_pos
                current_dist = abs(h_x - self.ht_x) + abs(h_y - self.ht_y) + abs(h_z - self.ht_z)
                new_dist = abs(x - self.ht_x) + abs(y - self.ht_y) + abs(z - self.ht_z)
                if new_dist < current_dist:
                    self.head_pos = (x, y, z)
                    self.head_particles_indices = []

        key = (int(round(x * 100)), int(round(y * 100)), int(round(z * 100)))

        if self.use_graphics:
            self.visuals.voxels[key] = []

        for i in range(w_count):
            for j in range(h_count):
                for k in range(d_count):
                    if self.calculate_balance and self.head_pos == (
                            x, y, z) and i == w_count // 2 and (
                            j == 0 or j == (h_count - 1)) and k == d_count // 2:
                        self.head_particles_indices.append(len(self.x))
                    if self.use_graphics and (i == 0 or (i == w_count - 1)) and (
                            j == 0 or j == (h_count - 1)) and (
                            k == 0 or k == (d_count - 1)):
                        self.visuals.voxels[key].append(self.n_particles)

                    self.x[0, self.n_particles] = [
                        x + (i + 0.5) * real_dx + self.offset_x,
                        y + (j + 0.5) * real_dy + self.offset_y,
                        z + (k + 0.5) * real_dz + self.offset_z
                    ]
                    self.actuator_id[self.n_particles] = actuation
                    self.n_particles += 1

    def set_offset(self, x, y, z):
        self.offset_x = x
        self.offset_y = y
        self.offset_z = z

    def set_n_actuators(self, n_act):
        self.num_actuators = n_act


@ti.data_oriented
class VSR3D:
    def __init__(self, params: Dict, use_graphics: bool = False,
                 color_t0: np.ndarray = np.array([255, 255, 255])):
        self.params = params
        self.color = color_t0

        self.visuals: SceneVisuals = None

        self.actuation = scalar()
        self.actuator_id = ti.field(ti.i32)
        self.x, self.v = vec(), vec()
        self.pos = vec()
        self.grid_v_in, self.grid_m_in = vec(), scalar()
        self.grid_v_out = vec()
        self.C, self.F = mat(), mat()

        self.global_frequency = scalar()
        self.offsets = scalar()

        self.n_particles = 0
        self.n_actuators = 0
        self.current_step = 0
        self.next_step = 1

        self.agent, self.steps_per_cycle = None, None
        self.origin = np.array([0.5, 0.0, 0.5])

        self.calculate_balance = params['experiment']['balanced_locomotion']
        self.use_graphics = use_graphics or self.params["evaluation"]["visualization"]["render"]
        self.save_every_n_steps = self.params["evaluation"]["visualization"]["save_every_n_steps"]
        if self.use_graphics:
            self.res = [visualize_resolution, visualize_resolution]
            self.pcs = []
        self.faces = None
        self.creature_head_particle_indices = None
        self.rolls_and_pitches = []

        self.max_n_particles = 27 * 125
        self.max_n_actuators = 125
        self.allocate_fields()

    def allocate_fields(self):
        ti.root.dense(ti.i, self.max_n_actuators).place(self.offsets, self.global_frequency)
        ti.root.dense(ti.ij, (2, self.max_n_actuators)).place(self.actuation)

        ti.root.dense(ti.i, self.max_n_particles).place(self.actuator_id)
        ti.root.dense(ti.k, 2).dense(ti.l, self.max_n_particles).place(self.x, self.v, self.C,
                                                                       self.F)
        ti.root.dense(ti.ijk, n_grid).place(self.grid_v_in, self.grid_m_in, self.grid_v_out)
        ti.root.place(self.pos)

    @ti.kernel
    def clear_grid(self):
        for I in ti.grouped(self.grid_m_in):
            self.grid_v_in[I] = ti.zero(self.grid_v_in[I])
            self.grid_m_in[I] = 0

    @ti.kernel
    def p2g(self, current_step: ti.i32, next_step: ti.i32, n_particles: ti.i32):
        for p in range(n_particles):
            base = ti.cast(self.x[current_step, p] * inv_dx - 0.5, ti.i32)
            fx = self.x[current_step, p] * inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            new_F = (ti.Matrix.identity(float, dim) + dt * self.C[current_step, p]) @ self.F[
                current_step, p]
            J = (new_F).determinant()
            self.F[next_step, p] = new_F

            act_id = self.actuator_id[p]
            act = 0.0 if act_id == -1 else self.actuation[current_step, act_id] * act_strength

            A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) * act
            cauchy = mu * (new_F @ new_F.transpose()) + ti.Matrix.identity(float, dim) * (
                    la * ti.log(J) - mu)
            cauchy += new_F @ A @ new_F.transpose()
            stress = -(dt * 4 * inv_dx * inv_dx) * cauchy
            affine = stress + self.C[current_step, p]

            for offset in ti.static(ti.grouped(ti.ndrange(*(3, 3, 3)))):
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                self.grid_v_in[base + offset] += weight * (
                        self.v[current_step, p] + affine @ dpos)
                self.grid_m_in[base + offset] += weight

    @ti.kernel
    def grid_op(self):
        for I in ti.grouped(self.grid_m_in):
            self.grid_v_out[I] = (1 / (self.grid_m_in[I] + 1e-10)) * self.grid_v_in[I]
            self.grid_v_out[I][1] -= dt * gravity
            cond = (I[1] < bound) & (self.grid_v_out[I][1] < 0) | (I[1] > n_grid - bound) & (
                    self.grid_v_out[I][1] > 0)
            self.grid_v_out[I] = 0 if cond else self.grid_v_out[I]

    @ti.kernel
    def g2p(self, current_step: ti.i32, next_step: ti.i32, n_particles: ti.i32):
        for p in range(0, n_particles):
            base = ti.cast(self.x[current_step, p] * inv_dx - 0.5, ti.i32)
            fx = self.x[current_step, p] * inv_dx - ti.cast(base, real)

            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector(zero_vec())
            new_C = ti.Matrix(zero_matrix())

            for offset in ti.static(ti.grouped(ti.ndrange(*(3, 3, 3)))):
                dpos = (offset - fx)
                g_v = self.grid_v_out[base + offset]
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]

                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

            self.v[next_step, p] = new_v
            self.x[next_step, p] = self.x[current_step, p] + dt * self.v[next_step, p]
            self.C[next_step, p] = new_C

    @ti.kernel
    def translate_xz(self, current_step: ti.i32, n_particles: ti.i32, x_shift: real,
                     z_shift: real):
        for p in range(n_particles):
            self.x[current_step, p][0] -= x_shift
            self.x[current_step, p][2] -= z_shift

    @ti.kernel
    def compute_actuation(self, t: ti.i32, current_step: ti.i32, n_actuators: ti.i32):
        for i in range(n_actuators):
            gf = self.global_frequency[i]
            offset = self.offsets[i]
            act = ti.sin(gf * (t * dt - offset))
            self.actuation[current_step, i] = ti.tanh(act)

    @ti.kernel
    def _compute_pos(self, current_step: ti.i32, n_particles: ti.i32):
        for p in range(n_particles):
            contrib = 1.0 / n_particles
            self.pos[None].atomic_add(contrib * self.x[current_step, p])

    def compute_pos(self) -> np.ndarray:
        self.pos.from_numpy(np.zeros(3, dtype=np.float32))
        self._compute_pos(self.current_step, self.n_particles)
        return self.pos.to_numpy()

    def advance(self, s, current_step, next_step, n_particles, n_actuators):
        self.clear_grid()
        self.compute_actuation(s, current_step, n_actuators)
        self.p2g(current_step, next_step, n_particles)
        self.grid_op()
        self.g2p(current_step, next_step, n_particles)

    def calculate_roll_pitch(self):
        if self.calculate_balance:
            i1, i2 = self.creature_head_particle_indices

            a = np.array(self.x[self.current_step, i2] - self.x[self.current_step, i1])
            xy_proj = a[[0, 1]]
            yz_proj = a[[1, 2]]

            xy_proj /= np.linalg.norm(xy_proj)
            yz_proj /= np.linalg.norm(yz_proj)

            x_axis = np.array((1, 0))
            z_axis = np.array((0, 1))

            roll = np.abs(np.arccos(np.dot(xy_proj, x_axis)) - np.pi / 2) / (np.pi / 2)
            pitch = np.abs(np.arccos(np.dot(yz_proj, z_axis)) - np.pi / 2) / (np.pi / 2)

            self.rolls_and_pitches.append(max(roll, pitch))

    def forward(self, total_steps):
        # simulation
        pos = self.compute_pos()
        init_pos = pos - self.origin
        total_x_shift, total_z_shift = 0, 0
        x_shift, _, z_shift = init_pos
        self.translate_xz(self.current_step, self.n_particles, float(x_shift), float(z_shift))
        init_pos = self.compute_pos()

        for s in range(1, total_steps + 1):
            if s % self.steps_per_cycle == 0:
                self.calculate_roll_pitch()
                pos = self.compute_pos()
                x_shift, _, z_shift = pos - self.origin
                total_x_shift += x_shift
                total_z_shift += z_shift
                self.translate_xz(self.current_step, self.n_particles, float(x_shift),
                                  float(z_shift))
            self.render_step(s, total_x_shift, total_z_shift)

            self.advance(s, self.current_step, self.next_step, self.n_particles, self.n_actuators)
            self.current_step = (self.current_step + 1) % 2
            self.next_step = (self.next_step + 1) % 2

        pos = self.compute_pos()
        x_shift, y_shift, z_shift = pos - self.origin
        total_x_shift += x_shift
        total_z_shift += z_shift
        final_pos = np.array(
            [total_x_shift + self.origin[0], y_shift + self.origin[1],
             total_z_shift + self.origin[2]])

        if self.use_graphics:
            self.visuals.graphics_engine.shutdown()

        return np.array(init_pos), np.array(final_pos)

    def render_step(self, s, x_shift, z_shift):
        if self.use_graphics and (s == 1 or s % self.save_every_n_steps == 0):
            x = self.x.to_numpy()[self.current_step]
            aid = self.actuator_id.to_numpy()
            act = self.actuation.to_numpy()[self.current_step][aid] * .5
            act *= (aid >= 0)
            if s == 1:
                self.visuals.render(x, act, x_shift, color=self.color)
            else:
                self.visuals.render(x, act, x_shift)

    def morphology(self, scene, agent):
        # Instantiate morphology
        h, w, d = [self.params["environment"]["voxel_size"]] * 3

        morphology = agent.morphology
        x_voxels = morphology.x_voxels * w * 2
        y_voxels = morphology.y_voxels * h * 2
        z_voxels = morphology.z_voxels * d * 2
        x_voxels -= np.min(x_voxels)
        y_voxels -= np.min(y_voxels)
        z_voxels -= np.min(z_voxels)

        x_voxels += 0.5
        y_voxels += h
        z_voxels += 0.5

        scene.ht_x, scene.ht_y, scene.ht_z = (
            np.max(x_voxels) + w, np.max(y_voxels) + 1.1 * h, np.mean(z_voxels))
        scene.set_offset(w / 2, h / 2, d / 2)
        muscle_id = 0
        for x, y, z, muscle in zip(x_voxels, y_voxels, z_voxels, morphology.muscle_voxels):
            if muscle:
                actuation = muscle_id
                muscle_id += 1
            else:
                actuation = -1
            scene.add_rect(x=x, y=y, z=z, w=w, h=h, d=d, actuation=actuation)

        if self.use_graphics:
            scene.visuals.find_faces_for_rendering(w, scene)

        # print(f"NUMBER OF VOXELS:       {len(x_voxels)}")
        # print(f"NUMBER OF PARTICLES:    {len(scene.x)}")
        # print(f"NUMBER OF PARTICLES:    {len(scene.x) ** (1/3)}")

        scene.set_n_actuators(muscle_id)

    def controller(self, agent):
        # Intantiate controller
        for i, (f_multiplier, offset) in enumerate(
                zip(agent.controller.frequency_multipliers, agent.controller.offsets)):
            self.global_frequency[i] = agent.controller.global_frequency * f_multiplier
            self.offsets[i] = offset

    def run(self, agent):
        self.agent = agent

        scene = Scene(self.params, self.agent.genome_path, self.use_graphics)
        self.morphology(scene, agent)
        self.creature_head_particle_indices = scene.head_particles_indices
        self.n_particles = scene.n_particles
        self.n_actuators = scene.num_actuators
        self.controller(agent)

        if self.use_graphics:
            self.visuals = scene.visuals

        self.x.from_numpy(scene.x)
        self.v.from_numpy(scene.v)
        self.actuator_id.from_numpy(scene.actuator_id)
        self.F.from_numpy(scene.F)
        self.C.from_numpy(scene.C)

        self.steps_per_cycle = int(self.agent.controller.period / dt)
        total_steps = self.params["evaluation"]["num_actuation_cycles"] * self.steps_per_cycle
        init_pos, final_pos = self.forward(total_steps)

        delta = final_pos - init_pos
        x_dist = delta[0]
        balance = 1 - np.max(self.rolls_and_pitches) if self.calculate_balance else 0.0

        self.agent, self.steps_per_cycle, self.visuals, self.faces = None, None, None, None
        self.n_particles, self.n_actuators, self.current_step = 0, 0, 0
        self.next_step = 1
        return {"endpos": delta, "dist": x_dist, "balance": balance}
