import torch
import typing
from typing import Callable, Dict, List

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Box, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils
from vmas.simulator.sensors import Lidar
# from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic import Holonomic

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom
    
    
from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = 500
    n_agents: int = 2
    n_packages: int = 1
    package_width: float = 0.3
    package_length: float = 0.3
    package_mass: float = 10

class MyTransportRotateScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # == INPUT ==
        self.n_packages = kwargs.pop("n_packages", TaskConfig.n_packages)
        self.package_length = kwargs.pop("package_length", TaskConfig.package_length)
        self.package_width = kwargs.pop("package_width", TaskConfig.package_width)
        self.package_mass = kwargs.pop("package_mass", TaskConfig.package_mass)
        self.n_agents = kwargs.pop("n_agents", TaskConfig.n_agents)

        self.comms_range = kwargs.pop("comms_range", 0)
        ScenarioUtils.check_kwargs_consumed(kwargs) # check if all of kwargs are processed.

        # == VARIABLES ==
        self.shaping_factor = 100
        self.world_semidim = 1
        self.agent_radius = 0.1
        self.goal_radius = 0.1
        
        self.lidar_range = 0.5
        self.n_lidar_rays = 12
        
        # == WORLD ==
        world = World(
            batch_dim,
            device,
            x_semidim=self.world_semidim
            + 2 * self.agent_radius
            + max(self.package_width, self.package_length),
            y_semidim=self.world_semidim
            + 2 * self.agent_radius
            + max(self.package_width, self.package_length),
        )
        
        # == AGENTS ==
        # True-> detectable, False-> undetectable
        # Lidar detects all of objects without goals
        
        self.u_size = 2  # x_force, y_force
        self.c_size = self.n_agents
        agent_obs_size = 3  # pos, vel, action(force)
        self.d_size = agent_obs_size
                
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                # movable=True,
                shape=Sphere(self.agent_radius),
                u_multiplier=0.5,    # force magnitudes
                render_action=False,
                dynamics=Holonomic(),   # (x force, y force)
                action_size=self.u_size + self.c_size + self.d_size,
            )
            
            agent.w = torch.zeros(
                batch_dim, self.c_size,
                device=device,
            )
            agent.z = torch.zeros(
                batch_dim, self.d_size,
                device=device,
            )
            
            agent.comm_usage = torch.zeros(
                batch_dim,
                device=device,
                dtype=torch.float32,
            )
            
            world.add_agent(agent)
            
        # == GOAL ==
        goal = Landmark(
            name="goal",
            collide=False,
            rotatable=True,
            shape=Box(
                length=self.package_length,
                width=self.package_width,
            ),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        
        # == PACKAGES ==
        self.packages = []
        for i in range(self.n_packages):
            package = Landmark(
                name=f"package_{i}",
                collide=True,
                movable=True,
                rotatable=True,
                mass=self.package_mass,
                shape=Box(
                    length=self.package_length,
                    width=self.package_width,
                ),
                color=Color.BLUE,
            )
            package.goal = goal
            self.packages.append(package)
            world.add_landmark(package)

        return world
    
    def reset_world_at(self, env_index: int = None):
        
        # spawn agents
        # ScenarioUtils.spawn_entities_randomly(
        #     self.world.agents,
        #     self.world,
        #     env_index,
        #     min_dist_between_entities=self.agent_radius * 2,
        #     x_bounds=(
        #         -self.world_semidim,
        #         self.world_semidim,
        #     ),
        #     y_bounds=(
        #         -self.world_semidim,
        #         self.world_semidim,
        #     ),
        # )
        # agent_occupied_positions = torch.stack(
        #     [agent.state.pos for agent in self.world.agents], dim=1
        # )
        # if env_index is not None:
        #     agent_occupied_positions = agent_occupied_positions[env_index].unsqueeze(0)
            
        
        goal = self.world.landmarks[0] 
        
        # PackageとGoalをスポーンさせる
        ScenarioUtils.spawn_entities_randomly(
            [goal] + self.packages, 
            self.world,
            env_index,
            min_dist_between_entities=max(
                package.shape.circumscribed_radius() * 1.5 + 0.01
                for package in self.packages),
            x_bounds=(-self.world_semidim, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
            # occupied_positions=agent_occupied_positions,
        )
        
        target_package = self.packages[0]
        
        # パッケージの位置を取得
        # env_indexがNoneの時は全環境(Batch)、ある時は特定環境のみ処理
        if env_index is None:
            batch_size = self.world.batch_dim
            t_pkg_pos = target_package.state.pos # shape: (batch_size, 2)
        else:
            batch_size = 1
            t_pkg_pos = target_package.state.pos[env_index].unsqueeze(0)

        for i, agent in enumerate(self.world.agents):
            # ランダムな角度 (0 ~ 2pi) を生成
            theta = torch.rand(batch_size, device=self.world.device) * 2 * torch.pi
            
            # 距離 = パッケージの半径 + エージェントの半径 + わずかな隙間(epsilon)
            # ぴったりすぎると物理演算で弾け飛ぶことがあるので少し余裕を持たせるのが定石です
            dist = target_package.shape.circumscribed_radius() + agent.shape.radius + 0.05
            
            # 相対位置ベクトル (dx, dy)
            dx = dist * torch.cos(theta)
            dy = dist * torch.sin(theta)
            
            # 新しい位置 = パッケージ位置 + 相対位置
            # env_index の有無で代入先を分岐
            if env_index is None:
                agent.state.pos[:] = t_pkg_pos + torch.stack([dx, dy], dim=1)
            else:
                agent.state.pos[env_index] = (t_pkg_pos + torch.stack([dx, dy], dim=1)).squeeze(0)
        
        # reset w,z
        for agent in self.world.agents:
            if env_index is None:
                agent.w.zero_()
                agent.z.zero_()
                agent.comm_usage.zero_()
            else:
                agent.w[env_index].zero_()
                agent.z[env_index].zero_()
                agent.comm_usage[env_index].zero_()
    
        # initial reward
        for package in self.packages:
            package.on_goal = self.world.is_overlapping(package, package.goal)
            
            if env_index is None:
                package.global_shaping = (
                    torch.linalg.vector_norm(
                        package.state.pos - package.goal.state.pos, dim=1
                    ) * self.shaping_factor
                )
            else:
                package.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        package.state.pos[env_index] - package.goal.state.pos[env_index]
                    ) * self.shaping_factor
                )
            
    def process_action(self, agent: Agent):
        c = agent.action.u[:,self.u_size:self.u_size+self.c_size]
        d = agent.action.u[:,self.u_size+self.c_size:self.u_size+self.c_size+self.d_size]
        agent.action.u = agent.action.u[:,:self.u_size]
    
        agent.w = (c > 0).int() # .int() makes True to 1, False to 0
        agent.z = (d > 0).int()
        
        # 通信量の累積和を計算
        step_usage = (agent.w.sum(dim=-1) * agent.z.sum(dim=-1) * 2).float()
        agent.comm_usage += step_usage
    
    def reward(self, agent: Agent):
        # is_first = agent == self.world.agents[0]
        
        self.rew = torch.zeros(
            self.world.batch_dim,
            device=self.world.device,
            dtype=torch.float32,
        )
        
        # _lambda = 0.2
        _lambda = 0.1
        w_norm = torch.linalg.norm(agent.w.float(), ord=1, dim=-1)
        z_norm = torch.linalg.norm(agent.z.float(), ord=1, dim=-1)
        comm_reward = - _lambda * (w_norm + z_norm)
        
        self.rew += comm_reward
        
        # for package in self.packages:
        #     # vector_norm: L2 norm of differences of x and y
        #     package.dist_to_goal = torch.linalg.vector_norm(
        #         package.state.pos - package.goal.state.pos, dim=1
        #     )
        #     package.on_goal = self.world.is_overlapping(package, package.goal)
            
        #     # == positional reward ==
        #     package_dis_reward = - package.dist_to_goal
        #     # パッケージがゴールに到達していない分だけ加算される
        #     self.rew[~package.on_goal] += package_dis_reward[~package.on_goal]
        for package in self.packages:
                # vector_norm: L2 norm of differences of x and y
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)

                # == progress reward ==
                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew[~package.on_goal] += (
                    package.global_shaping[~package.on_goal]
                    - package_shaping[~package.on_goal]
                )
                package.global_shaping = package_shaping
            
        time_reward = -0.01
        self.rew += time_reward
        
        return self.rew
    
    def observation(self, agent: Agent):
        """
        observation options:
        - package:
            - the distance between the goal and the package
            - the current status of the package
                - angle, velocity?
            - the position of the package
            - the distance and relative angle of the packages from the robots
            - the current progress of transport
                - such as "previous status - current status"
        - robot:
            - the positions of all of robots
            - the current state of all of robots
        
        """
        
        
        package_obs = []
        for package in self.packages:
            package_obs.append(package.state.pos)       # x
            package_obs.append(package.state.vel)       # v
            package_obs.append(package.state.rot)       # thete
            package_obs.append(package.state.ang_vel)   # omega
            package_obs.append(package.goal.state.pos)  # x*
        
        agent_obs = []
        agent_obs.append(agent.state.pos)   # x_i
        agent_obs.append(agent.state.vel)   # v_i
        if agent.action.u is None:
            agent_obs.append(
                torch.zeros(
                    self.world.batch_dim, self.u_size, device=self.world.device
                )
            )
        else:
            agent_obs.append(agent.action.u[:, :self.u_size])    # u_i  sliced for initial observation
        
        # communication
        for j, _agent in enumerate(self.world.agents):
            if _agent == agent:
                continue
            
            # w corresponds to agent index j.
            w_j = agent.w[:, j].unsqueeze(-1)
            
            # z corresponds to feature type.
            # Position (k=0)
            z_pos = agent.z[:, 0].unsqueeze(-1)
            mask_pos = (w_j * z_pos) == 1
            agent_obs.append(torch.where(mask_pos, _agent.state.pos, torch.tensor(-1.0, device=self.world.device)))
            
            # Velocity (k=1)
            z_vel = agent.z[:, 1].unsqueeze(-1)
            mask_vel = (w_j * z_vel) == 1
            agent_obs.append(torch.where(mask_vel, _agent.state.vel, torch.tensor(-1.0, device=self.world.device)))
            
            # Action (k=2)
            z_act = agent.z[:, 2].unsqueeze(-1)
            mask_act = (w_j * z_act) == 1
            if _agent.action.u is None:
                other_u = torch.zeros(
                    self.world.batch_dim, self.u_size, device=self.world.device
                )
            else:
                other_u = _agent.action.u[:, :self.u_size]

            agent_obs.append(torch.where(mask_act, other_u, torch.tensor(-1.0, device=self.world.device)))
        
        observations = torch.cat(
            [
                *package_obs,
                *agent_obs,
            ],
            dim=-1,
        )
        
        # print(observations.size())
        
        return observations

    def info(self, agent: Agent) -> Dict[str, torch.Tensor]:
        return {"comm_usage": agent.comm_usage}

    def done(self):
        """
        TODO:
            - clear
            - collision
        """
        all_package_on_goal = torch.all(
            torch.stack(
                [package.on_goal for package in self.packages],
                dim=1,
            ),
            dim=-1,
        )
        
        # まず座標(dim=2)のどちらかがはみ出ているか -> [Batch, Num_Agents]
        # 次に誰か一人でも(dim=1)はみ出ているか -> [Batch]
        # any_agent_collide_with_wall = torch.any(
        #     torch.any(
        #         torch.stack(
        #             [abs(agent.state.pos) + self.agent_radius >= self.world_semidim for agent in self.world.agents],
        #             dim=1,
        #         ),
        #         dim=-1,
        #     ),
        #     dim=-1,
        # )
        # dones = all_package_on_goal | any_agent_collide_with_wall
        dones = all_package_on_goal
        return dones
        
    # def extra_render(self, env_index: int = 0) -> "List[Geom]":
    #     from vmas.simulator import rendering

    #     geoms: List[Geom] = []

    #     # Communication lines
    #     for i, agent1 in enumerate(self.world.agents):
    #         for j, agent2 in enumerate(self.world.agents):
    #             if j <= i:
    #                 continue
    #             agent_dist = torch.linalg.vector_norm(
    #                 agent1.state.pos - agent2.state.pos, dim=-1
    #             )
    #             if agent_dist[env_index] <= self.comms_range:
    #                 color = Color.BLACK.value
    #                 line = rendering.Line(
    #                     (agent1.state.pos[env_index]),
    #                     (agent2.state.pos[env_index]),
    #                     width=1,
    #                 )
    #                 xform = rendering.Transform()
    #                 line.add_attr(xform)
    #                 line.set_color(*color)
    #                 geoms.append(line)

    #     return geoms