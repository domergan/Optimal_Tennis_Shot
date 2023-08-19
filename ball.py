import numpy as np
import matplotlib.pyplot as plt

G = np.array([0, 0, -9.81])

class Ball:
    def __init__(self, initPosition, initVelocity, mass, radius, dragCoef, airDensity):
        self.mass = mass
        self.radius = radius
        self.position = initPosition
        self.velocity = initVelocity
        self.dragCoef = dragCoef
        self.airDensity = airDensity

    def __str__(self):
        return f"Ball (Mass: {self.mass} g, Radius: {self.radius} cm)"

class Simulator:
    def __init__(self, initVelocity, initPosition, ball):
        self.velocity = initVelocity
        self.position = initPosition
        self.momentum = (ball.mass * initVelocity)
        self.ball = ball
        self.dt = 0.001
        self.t = 0

    def simulate(self):
        
        gravityForce = self.ball.mass * G
        area = np.pi * (self.ball.radius ** 2)
        
        trajectoryX = []
        trajectoryY = []
        trajectoryZ = []
        
        while self.position[2] >= 0.1:
            dragForce = 0.5 * self.ball.airDensity * np.square(self.velocity) * self.ball.dragCoef * area
            netForce = dragForce + gravityForce
            self.momentum = self.momentum + netForce * self.dt
            self.velocity = self.momentum / self.ball.mass
            self.position = self.position + self.velocity * self.dt
            self.t = self.t + self.dt
            
            #print(f"Time: {self.t:.3f} s, Position: {self.position}")
            
            trajectoryX.append(self.position[0])
            trajectoryY.append(self.position[1])
            trajectoryZ.append(self.position[2])
            
        return [trajectoryX, trajectoryY, trajectoryZ]
    
class Visualizer:
    def __init__(self, trajectory):
        self.trajectory = trajectory

    def plot_trajectory(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        xs = self.trajectory[0]
        ys = self.trajectory[1]
        zs = self.trajectory[2]
        
        ax.plot(xs, ys, zs, label='Trajectory')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        ax.set_zlim(zmin=0)
        
        ax.set_title('Ball Trajectory')
        ax.legend()

        plt.show()
        
        
if __name__ == "__main__":
    initPosition = np.array([0, 0, 10])
    initVelocity = np.array([1, 1, 0])
    mass = 0.2
    radius = 0.05
    dragCoef = 0.47
    airDensity = 1.225

    ball = Ball(initPosition, initVelocity, mass, radius, dragCoef, airDensity)
    simulator = Simulator(initVelocity, initPosition, ball)
    trajectory = simulator.simulate()
    
    visualizer = Visualizer(trajectory)
    visualizer.plot_trajectory()

    