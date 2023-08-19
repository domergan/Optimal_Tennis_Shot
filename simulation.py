import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class SpinningBall:
    def __init__(self, mass, radius, pos_0, vel_0, ang_vel):
        self.mass = mass
        self.radius = radius
        self.pos_0 = pos_0
        self.vel_0 = vel_0
        self.ang_vel = ang_vel
        
        # Constants
        self.area = np.pi * (radius ** 2)
        self.air_density = 0.09
        self.drag_coeff = 0.1
        self.gravity = np.array([0, 0, -9.81])
        self.max_bounce = 1
        
    def derivative(self, t, state):
        x, y, z, vx, vy, vz = state
        vel = np.array([vx, vy, vz])
        vel_mag = np.linalg.norm(vel)
        Cm = 1 / (2 + 1.98 * (vel_mag / (self.ang_vel * self.radius)))
        
        Magnus = Cm * self.air_density * np.pi * ((self.radius ** 2) / 8) * np.cross(np.array([0, 0, self.ang_vel]), vel)
        drag = -0.5 * self.air_density * (vel_mag ** 2) * self.drag_coeff * self.area * (vel / vel_mag)
        gravity = self.mass * self.gravity

        acc = (drag + gravity + Magnus) / self.mass
            
        return [vx, vy, vz, acc[0], acc[1], acc[2]]
    
    def hit_ground(self, t, state):
        return state[2]
    
    hit_ground.terminal  = True
    hit_ground.direction = -1
    
    def solve(self, end_time):
        init_cond = [*self.pos_0, *self.vel_0]
        
        start_time = 0
        eval_times = []
        
        traj_x = []
        traj_y = []
        traj_z = []

        while start_time < end_time:
        
            solution = solve_ivp(self.derivative, 
                                 [0, end_time], 
                                 init_cond, 
                                 events=self.hit_ground, 
                                 dense_output=True)
            
            init_cond = solution.y[:, -1]
            
            if init_cond[2] < 0.05:
                init_cond[-1] = -init_cond[-1] * 0.9
                
            eval_times = np.linspace(0, solution.t[-1], 1000)
            start_time = solution.t[-1]
            
            solution_at_eval_times = solution.sol(eval_times)
            
            traj_x = np.append(traj_x, solution_at_eval_times[0])
            traj_y = np.append(traj_y, solution_at_eval_times[1])
            traj_z = np.append(traj_z, solution_at_eval_times[2])
            
            if self.max_bounce == 0:
                break
            
            self.max_bounce -= 1
        
        return [traj_x, traj_y, traj_z]
            
    def plot_trajectory(self, trajectory):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = trajectory[0]
        ys = trajectory[1]
        zs = trajectory[2]
        ax.plot(xs, ys, zs, label='Trajectory')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Ball Trajectory')
        ax.legend()
        plt.show()

# Define the ball with its properties
ball = SpinningBall(mass=0.1, radius=0.035, pos_0=[0, 0, 2.2], vel_0=[0, 24, 2], ang_vel=15)

# Simulate and plot the trajectory
trajectory = ball.solve(end_time=10)
ball.plot_trajectory(trajectory)
