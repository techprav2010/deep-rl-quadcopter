import numpy as np
from physics_sim import PhysicsSim
from agents.ounoise import OUNoise
 

class Task():
    def __init__(self,
                 runtime=5.,
                 init_pose= np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]),
                 init_velocities=np.array([0.0, 0.0, 0.0]),
                 init_angle_velocities=np.array([0.0, 0.0, 0.0]),
                 pos_noise=0.25,
                 angle_noise=None,
                 velocity_noise=0.15,
                 velocity_angle_noise=None,
                 target_pos=np.array([0.0, 0.0, 10.0])
                 ):

        self.target_pos = target_pos
        self.pos_noise = pos_noise
        self.angle_noise = angle_noise
        self.velocity_noise = velocity_noise
        self.velocity_angle_noise = velocity_angle_noise
        self.action_size = 1
        self.action_repeat = 1
        self.action_high = 1.2 * 400
        self.action_low = 0.99 * 400
        self.noise = OUNoise(self.action_size, mu=0.0, theta=0.2, sigma=0.1)
        self.action_b = (self.action_high + self.action_low) / 2.0
        self.action_m = (self.action_high - self.action_low) / 2.0

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.state_size = len(self.get_state())



    def get_reward(self):

        """Uses current pose of sim to return reward."""

        # reward = np.tanh(1 - 0.7 * (abs(self.sim.pose[:3] - self.target_pos))).sum()
        # print("reward ", reward)

        # reward = np.square(self.sim.pose[:3] - self.target_pos).sum()
        # reward = np.sqrt(reward)
        # reward /=3
        # print("\n")
        # print("self.sim.pose ", self.sim.pose)
        # print("self.target_pos ", self.target_pos)
        # np.clip(reward, 10, -10)
        # reward /= 10
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        # reward = np.tanh(1 - 0.003 * (abs(self.sim.pose[:3] - self.target_pos))).sum()

        # reward = np.tanh(1 - 0.3 * (abs(self.sim.pose[:3] - self.target_pos))).sum()

        # reward = (1.5 - np.sum(np.square(( self.sim.pose[:3] -  self.target_pos) / 300.0))) * 2

        # reward = np.tanh( 1.-0.3*(abs(self.sim.pose[:3] - self.target_pos)).sum())
        # reward = (0.5 - np.mean(np.square((self.sim.pose[:3] - self.target_pos) / 200.0))) * 2
        # reward = (0.5 - np.mean(np.square((self.sim.pose[:3] - self.target_pos) / 300.0))) * 2

        # reward = 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum()
        # if(self.sim.pose[2] >0) :
        #     reward +=  10 #abs(self.sim.pose[2] - self.target_pos[2])
        # else:
        #     reward -=  10 #abs(self.sim.pose[2] - self.target_pos[2])
        # reward = np.tanh(reward)
        # reward = np.tanh(1 - np.mean(np.square(self.sim.pose[:3] - self.target_pos)))
        # reward=0
        # if (self.sim.pose[2] > 0):
        #     reward +=1
        # if self.sim.pose[2] >= self.target_pos[2]:
        #     reward += 5
        # return reward
        # reward = self.sim.v[2] / 10.0
        # reward += (self.sim.pose[2] - self.target_pos[2]) / 10.0
        # reward -= np.linalg.norm(self.sim.pose[:2]) / 10.0
        # return reward
        #
        # p1 = self.sim.pose[:3]
        # p2 = self.target_pos
        # env_bounds = 300.0
        # bound = np.array([env_bounds, env_bounds, env_bounds])
        # reward = (0.5 - np.mean(np.square((p1 - p2) / bound))) * 2

        reward = np.tanh(1. - 0.76 * (abs(self.sim.pose[:3] - self.target_pos)).sum())
        return reward

    def normalize_angles(self, angles):
        normalized_angles = np.copy(angles)
        for i in range(len(normalized_angles)):
            while normalized_angles[i] > np.pi:
                normalized_angles[i] -= 2 * np.pi
        return normalized_angles

    def get_state(self):
        position_error = (self.sim.pose[:3] - self.target_pos)
        return np.array([position_error[2], self.sim.v[2], self.sim.linear_accel[2]])

    def step(self, actionInput):
        reward = 0
        # pose_all = []
        for _ in range(self.action_repeat):
            action=actionInput
            action += self.noise.sample()
            action = np.clip(action, -1, 1)
            speed_of_rotors = (action * self.action_m) + self.action_b
            done = self.sim.next_timestep(speed_of_rotors* np.ones(4))  # update the sim pose and velocities
            reward += self.get_reward()
            next_state = self.get_state()
            if reward <= 0:
                done = True
            # pose_all.append(self.sim.pose)
            if self.sim.pose[2] >= self.target_pos[2]:
                reward += 1
            # if self.sim.pose[2] >= self.target_pos[2]:
            #     reward += 10
            # else:
            #     reward += -20
            # if done:
            #     if self.sim.time < self.sim.runtime:
            #         reward += -1
            #     else:
            #         reward += 5
            #     break
        # next_state = np.concatenate(pose_all)
        return next_state, reward, done


    def reset_noise(self):
        self.noise.reset()

    def reset(self):
        self.sim.reset()
        self.noise.reset()
        rnd_pos = np.copy(self.sim.init_pose)
        rnd_pos[2] += np.random.normal(0.0, self.pos_noise, 1)
        self.sim.pose = np.copy(rnd_pos)
        rnd_velocity = np.copy(self.sim.init_velocities)
        rnd_velocity[2] += np.random.normal(0.0, self.velocity_noise, 1)
        self.sim.v = np.copy(rnd_velocity)
    
        return self.get_state()
