import numpy as np
import matplotlib.pyplot as plt

class SpinningBall:
    def __init__(self, mass, radius, initial_position, initial_velocity, angular_velocity):
        self.mass = mass
        self.radius = radius
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.angular_velocity = angular_velocity
        
        self.A = np.pi * ((self.radius ** 2))
        self.rho = 0.09
        self.C = 0.1
        self.g = np.array([0, 0, -9.81])
        
    def dSdt(self, t, S):
        x, y, z, vx, vy, vz = S
        v = np.array([vx, vy, vz])
        
        v_magnitude = np.linalg.norm(v)
        Cm = 1 / (2 + 1.98 * (v_magnitude / (self.angular_velocity * self.radius)))
        F_magnus = Cm * self.rho * np.pi * ((self.radius ** 2) / 8) * np.cross(np.array([0, 0, self.angular_velocity]), v)
        F_drag = -0.5 * self.rho * (v_magnitude ** 2) * self.C * self.A * (v / v_magnitude)
        F_gravity = self.mass * self.g

        a = (F_drag + F_gravity + F_magnus) / self.mass
        
        # Bounce 
        if z < self.radius and vz < 0:
            vz = -vz * 0.7
            
        return [vx, vy, vz, a[0], a[1], a[2]]
    
    def eulerMethod(self, fun, y0, h, ti, tf):
        n = int((tf - ti) / h)
        t = np.linspace(ti, tf, n+1)
        y = np.zeros((n+1, len(y0)))
        y[0] = y0
        for i in range(n):
            y[i+1] = np.add(y[i], np.multiply(fun(t[i], y[i]), h))
        return y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5], t

    def simulateEuler(self, t_end, h):
        initial_conditions = [*self.initial_position, *self.initial_velocity]
        sol = self.eulerMethod(self.dSdt, initial_conditions, h, 0, t_end)
        return sol
            
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
ball = SpinningBall(mass=0.1, radius=0.035, initial_position=[0, 0, 2.5], initial_velocity=[0, 24, 2], angular_velocity=100)

# Simulate and plot the trajectory
trajectoryEuler = ball.simulateEuler(t_end=10, h=0.01)
ball.plot_trajectory(trajectoryEuler)
