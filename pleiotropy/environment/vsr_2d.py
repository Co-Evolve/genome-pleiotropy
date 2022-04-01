import os
from typing import Dict, Optional

import numpy as np
import taichi as ti

from pleiotropy.evolution import agent_handling, ocra
from pleiotropy.evolution.agent_handling.vsr_agent import VSRAgent

real = ti.f32

dim = 2
n_particles = 0
n_solid_particles = 0
n_actuators = 0

n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx

dt = 1e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 2048
steps = 1500
gravity = 3.8

bound = 3
coeff = 0.5

act_strength = 4


def scalar():
    return ti.field(dtype=real)


def vec():
    return ti.Vector.field(dim, dtype=real)


def mat():
    return ti.Matrix.field(dim, dim, dtype=real)


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.particles = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1

        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.particles.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act


@ti.data_oriented
class VSR2D:
    def __init__(self, params: Dict):
        self.use_graphics = params["evaluation"]["visualization"]["render"]
        self.gui = None
        self.actuation = scalar()
        self.actuator_id = ti.field(ti.i32)
        self.particle_type = ti.field(ti.i32)
        self.x, self.v = vec(), vec()
        self.grid_v_in, self.grid_m_in = vec(), scalar()
        self.grid_v_out = vec()
        self.C, self.F = mat(), mat()

        self.x_avg = vec()
        self.global_frequency = scalar()
        self.offsets = scalar()

        self.current_step = 0
        self.next_step = 1

        self.agent: Optional[VSRAgent] = None

    def allocate_fields(self):
        ti.root.dense(ti.i, n_actuators).place(self.offsets, self.global_frequency)

        ti.root.dense(ti.ij, (2, n_actuators)).place(self.actuation)
        ti.root.dense(ti.i, n_particles).place(self.actuator_id, self.particle_type)
        ti.root.dense(ti.k, 2).dense(ti.l, n_particles).place(self.x, self.v, self.C, self.F)
        ti.root.dense(ti.ij, n_grid).place(self.grid_v_in, self.grid_m_in, self.grid_v_out)
        ti.root.place(self.x_avg)

    @ti.kernel
    def clear_grid(self):
        for i, j in self.grid_m_in:
            self.grid_v_in[i, j] = [0, 0]
            self.grid_m_in[i, j] = 0

    @ti.kernel
    def p2g(self, current_step: ti.i32, next_step: ti.i32):
        for p in range(n_particles):
            base = ti.cast(self.x[current_step, p] * inv_dx - 0.5, ti.i32)
            fx = self.x[current_step, p] * inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_F = (ti.Matrix.diag(dim=2, val=1) + dt * self.C[current_step, p]) @ self.F[current_step, p]
            J = new_F.determinant()
            if self.particle_type[p] == 0:  # fluid
                sqrtJ = ti.sqrt(J)
                new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

            self.F[next_step, p] = new_F
            r, s = ti.polar_decompose(new_F)

            act_id = self.actuator_id[p]

            act = self.actuation[current_step, ti.max(0, act_id)] * act_strength
            if act_id == -1:
                act = 0.0

            A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
            cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            mass = 0.0
            if self.particle_type[p] == 0:
                mass = 4
                cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
            else:
                mass = 1
                cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                         ti.Matrix.diag(2, la * (J - 1) * J)
            cauchy += new_F @ A @ new_F.transpose()
            stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
            affine = stress + mass * self.C[current_step, p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                    weight = w[i](0) * w[j](1)
                    self.grid_v_in[base +
                                   offset] += weight * (mass * self.v[current_step, p] + affine @ dpos)
                    self.grid_m_in[base + offset] += weight * mass

    @ti.kernel
    def grid_op(self):
        for i, j in self.grid_m_in:
            inv_m = 1 / (self.grid_m_in[i, j] + 1e-10)
            v_out = inv_m * self.grid_v_in[i, j]
            v_out[1] -= dt * gravity
            if i < bound and v_out[0] < 0:
                v_out[0] = 0
                v_out[1] = 0
            if i > n_grid - bound and v_out[0] > 0:
                v_out[0] = 0
                v_out[1] = 0
            if j < bound and v_out[1] < 0:
                v_out[0] = 0
                v_out[1] = 0
                normal = ti.Vector([0.0, 1.0])
                lsq = (normal ** 2).sum()
                if lsq > 0.5:
                    if ti.static(coeff < 0):
                        v_out[0] = 0
                        v_out[1] = 0
                    else:
                        lin = (v_out.transpose() @ normal)(0)
                        if lin < 0:
                            vit = v_out - lin * normal
                            lit = vit.norm() + 1e-10
                            if lit + coeff * lin <= 0:
                                v_out[0] = 0
                                v_out[1] = 0
                            else:
                                v_out = (1 + coeff * lin / lit) * vit
            if j > n_grid - bound and v_out[1] > 0:
                v_out[0] = 0
                v_out[1] = 0

            self.grid_v_out[i, j] = v_out

    @ti.kernel
    def g2p(self, current_step: ti.i32, next_step: ti.i32):
        for p in range(n_particles):
            base = ti.cast(self.x[current_step, p] * inv_dx - 0.5, ti.i32)
            fx = self.x[current_step, p] * inv_dx - ti.cast(base, real)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector([0.0, 0.0])
            new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j]), real) - fx
                    g_v = self.grid_v_out[base(0) + i, base(1) + j]
                    weight = w[i](0) * w[j](1)
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

            self.v[next_step, p] = new_v
            self.x[next_step, p] = self.x[current_step, p] + dt * self.v[next_step, p]
            self.C[next_step, p] = new_C

    @ti.kernel
    def compute_actuation(self, t: ti.i32, current_step: ti.i32):
        for i in range(n_actuators):
            gf = self.global_frequency[i]
            offset = self.offsets[i]
            act = ti.sin(gf * t * dt - offset)
            self.actuation[current_step, i] = ti.tanh(act)

    @ti.kernel
    def compute_x_avg(self, current_step: ti.i32):
        for i in range(n_particles):
            contrib = 0.0
            if self.particle_type[i] == 1:
                contrib = 1.0 / n_solid_particles
            self.x_avg[None].atomic_add(contrib * self.x[current_step, i])

    @ti.ad.grad_replaced
    def advance(self, s, current_step, next_step):
        self.clear_grid()
        self.compute_actuation(s, current_step)
        self.p2g(current_step, next_step)
        self.grid_op()
        self.g2p(current_step, next_step)

    def forward(self, total_steps=steps):
        # simulation
        self.advance(0, self.current_step, self.next_step)
        self.x_avg[None] = [0, 0]
        self.compute_x_avg(self.current_step)
        init_pos = [x for x in self.x_avg[None]]

        for s in range(1, total_steps - 1):
            self.current_step = (self.current_step + 1) % 2
            self.next_step = (self.next_step + 1) % 2
            self.advance(s, self.current_step, self.next_step)
            self.visualize(s, f'visualizations/{self.agent.idx}/')

        self.x_avg[None] = [0, 0]
        self.compute_x_avg(self.current_step)
        final_pos = [x for x in self.x_avg[None]]

        return np.array(init_pos), np.array(final_pos)

    def morphology(self, scene, agent: VSRAgent):
        # Instantiate morphology
        h, w = 0.05, 0.05

        morphology = agent.morphology
        x_voxels = morphology.x_voxels * w * 2
        y_voxels = morphology.y_voxels * h * 2
        x_voxels -= np.min(x_voxels)
        y_voxels -= np.min(y_voxels)

        scene.set_offset(w / 2, h / 2)

        muscle_id = 0
        for x, y, muscle in zip(x_voxels, y_voxels, morphology.muscle_voxels):
            if muscle:
                actuation = muscle_id
                muscle_id += 1
            else:
                actuation = -1
            scene.add_rect(x=x, y=y, w=w, h=h, actuation=actuation)

        scene.set_n_actuators(muscle_id)

    def controller(self, agent: VSRAgent):
        # Intantiate controller
        for i, offset in enumerate(agent.controller.offsets):
            self.global_frequency[i] = agent.controller.global_frequency
            self.offsets[i] = offset

    def visualize(self, s, folder):
        if self.use_graphics and s % 16 == 0:
            aid = self.actuator_id.to_numpy()
            colors = np.empty(shape=n_particles, dtype=np.uint32)
            particles = self.x.to_numpy()[self.next_step]
            for i in range(n_particles):
                color = 0x111111
                if aid[i] != -1:
                    act = self.actuation[self.current_step, aid[i]]
                    color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
                colors[i] = color
            self.gui.circles(pos=particles, color=colors, radius=1.5)
            self.gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

            os.makedirs(folder, exist_ok=True)
            self.gui.show(f'{folder}/{s:04d}.png')

    def run(self, agent):
        self.gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF, show_gui=self.use_graphics)

        # initialization
        self.agent = agent
        scene = Scene()
        self.morphology(scene, agent)
        scene.finalize()
        self.allocate_fields()
        self.controller(agent)

        for i in range(scene.n_particles):
            self.x[0, i] = scene.particles[i]
            self.F[0, i] = [[1, 0], [0, 1]]
            self.actuator_id[i] = scene.actuator_id[i]
            self.particle_type[i] = scene.particle_type[i]

        init_pos, final_pos = self.forward(steps)

        delta = final_pos - init_pos
        dist = delta[0]

        self.gui.core.should_close = 1
        self.gui.running = False
        self.gui.close()
        self.gui = None

        return {"endpos": delta, "dist": dist}


if __name__ == '__main__':
    import pickle

    import sys

    sys.modules["agent_handling"] = agent_handling
    sys.modules["ocra"] = ocra
    with open("example_agent.pkl", "rb") as f:
        agent = pickle.load(f)

    ti.init(default_fp=real, arch=ti.cpu, flatten_if=True)
    for i in range(5):
        print(f"RUNNING ITERATION : {i}")
        env = VSR2D({"evaluation": {"use_graphics": True}})
        env.run(agent)
