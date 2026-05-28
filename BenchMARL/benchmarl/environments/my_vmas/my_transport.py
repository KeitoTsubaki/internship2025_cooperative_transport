import torch
import typing
from typing import Callable, Dict, List

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Box, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils
from vmas.simulator.sensors import Lidar
from vmas.simulator.dynamics.diff_drive import DiffDrive

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

class MyTransportScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # == INPUT ==
        self.n_packages = kwargs.pop("n_packages", TaskConfig.n_packages)
        self.package_length = kwargs.pop("package_length", TaskConfig.package_length)
        self.package_width = kwargs.pop("package_width", TaskConfig.package_width)
        self.package_mass = kwargs.pop("package_mass", TaskConfig.package_mass)
        n_agents = kwargs.pop("n_agents", TaskConfig.n_agents)

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
        entity_filter = lambda e: "goal" not in e.name
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                # movable=True,
                shape=Sphere(self.agent_radius),
                u_multiplier=0.5,    # force magnitudes
                render_action=False,
            )
            # agent = Agent(
            #     name=f"agent_{i}",
            #     collide=True,
            #     # movable=True,
            #     shape=Sphere(self.agent_radius),
            #     u_multiplier=0.5,    # force magnitudes
            #     render_action=False,
            #     sensors=(
            #         [
            #             Lidar(
            #                 world,
            #                 n_rays=self.n_lidar_rays,
            #                 max_range=self.lidar_range,
            #                 entity_filter=entity_filter,
            #                 render=True
            #             ),
            #         ]
            #     ),
            # )
            world.add_agent(agent)
            
        # == GOAL ==
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=self.goal_radius),
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
        # --- 1. まずパッケージとゴールを先にランダム配置する ---
        goal = self.world.landmarks[0] 
        
        # PackageとGoalをスポーンさせる
        ScenarioUtils.spawn_entities_randomly(
            [goal] + self.packages, 
            self.world,
            env_index,
            min_dist_between_entities=max(
                package.shape.circumscribed_radius() + goal.shape.radius + 0.01
                for package in self.packages),
            x_bounds=(-self.world_semidim, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
        )

        # --- 2. エージェントをパッケージの周囲に配置する ---
        # ※各エージェントをどのパッケージの横に置くか決める必要があります。
        # ここでは例として「全エージェントを最初のパッケージ(self.packages[0])の周囲」に配置します。
        # 複数パッケージある場合は割り振るロジックに変えてください。
        
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

        # --- 3. 初期報酬の計算 (既存コード維持) ---
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
        # # Random position between -1 and 1 if world semidim is 1
        
        # # == SPAWN AGENTS ==
        # ScenarioUtils.spawn_entities_randomly(
        #     self.world.agents,
        #     self.world,
        #     env_index,
        #     min_dist_between_entities=self.agent_radius * 2,
        #     x_bounds=(-self.world_semidim, self.world_semidim),
        #     y_bounds=(-self.world_semidim, self.world_semidim),
        # )
        
        # agent_occupied_positions = torch.stack(
        #     [agent.state.pos for agent in self.world.agents], dim=1
        # )
        
        # if env_index is not None:
        #     agent_occupied_positions = agent_occupied_positions[env_index].unsqueeze(0)
            
        # goal = self.world.landmarks[0]  # choose index 1 because the goal and the packages are treated as same landmarks
        # # == SPAWN LANDMARKS ==
        # ScenarioUtils.spawn_entities_randomly(
        #     [goal] + self.packages, 
        #     # why it doesn't use self.world.landmarks?
        #     # A. self.wprld.landmarks includes goals, packages, and obstacles. it is in order to separate spawning method.
        #     self.world,
        #     env_index,
        #     min_dist_between_entities=max(
        #         package.shape.circumscribed_radius() + goal.shape.radius + 0.01
        #         for package in self.packages),
        #     x_bounds=(-self.world_semidim, self.world_semidim),
        #     y_bounds=(-self.world_semidim, self.world_semidim),
        #     occupied_positions=agent_occupied_positions,
        # )
        
        # for package in self.packages:
        #     package.on_goal = self.world.is_overlapping(package, package.goal)
            
        #     # calculate the reward regarding the deistances between packages and goals on the first state to calcurate progress reward on the first step
        #     if env_index is None:
        #         package.global_shaping = (
        #             torch.linalg.vector_norm(
        #                 package.state.pos - package.goal.state.pos, dim=1
        #             ) * self.shaping_factor
        #         )
        #     else:
        #         package.global_shaping[env_index] = (
        #             torch.linalg.vector_norm(
        #                 package.state.pos[env_index] - package.goal.state.pos[env_index]
        #             ) * self.shaping_factor
        #         )
    
    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        
        # It calculates the reward only if agent idx is 0
        # because the reward is related to only packages,
        # we don't need to consider each agent.
        if is_first:
            # Initialize the reward vector size
            # to the number of parallel environments
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )
            
            for package in self.packages:
                # vector_norm: L2 norm of differences of x and y
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)
                
                # == positional reward ==
                # it depends on the initial state
                # package_dis_reward = - package.dist_to_goal
                # # ~ means NOT
                # self.rew[~package.on_goal] += package_dis_reward[~package.on_goal]
                
                
                # == progress reward ==
                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew[~package.on_goal] += (
                    package.global_shaping[~package.on_goal]
                    - package_shaping[~package.on_goal]
                )
                package.global_shaping = package_shaping
            
            # TODO: collision penalty
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
            
                
            # TODO: ここ間違えてそう. rotateの方は修正した
            minor_reward = -0.01
            self.rew[~package.on_goal] += minor_reward  
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
            # package_obs.append(package.state.pos - package.goal.state.pos)
            package_obs.append(package.goal.state.pos - package.state.pos)
            package_obs.append(package.state.pos - agent.state.pos)
            package_obs.append(package.state.vel)
            # package_obs.append(package.on_goal.unsqueeze(-1))
        
        agent_obs = []
        # agent_obs.append(agent.state.pos)
        agent_obs.append(agent.state.vel)
        
        # sensor_obs = []
        # sensor_obs.append(agent.sensors[0]._max_range - agent.sensors[0].measure())
        
        observations = torch.cat(
            [
                *agent_obs,
                *package_obs,
                # *sensor_obs
            ],
            dim=-1,
        )
        
        # print(observations.size())
        
        return observations

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