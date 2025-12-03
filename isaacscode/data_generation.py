import numpy as np
import matplotlib.pyplot as plt



def generate_flower_data(dist=6, sigma=(1.4, 2.5)):

    # make random draws reproducible
    np.random.seed(42)
    
    # randomly sample data around a point with gaussian noise of given sigma
    points = []
    sigmas = []
    known_landmarks = []

    for i in range(5):
        mean = np.array([0.0, dist * (i + 1)])
        sigma_sample = np.random.uniform(*sigma)
        samples = np.random.normal(loc=mean, scale=sigma_sample/2.54, size=(10, 2))
        for s in samples:
            points.append(s)
            sigmas.append(sigma_sample)
            known_landmarks.append(i+1)


    return points, sigmas, known_landmarks


if __name__ == "__main__":

    dist = 6
    sigma = (1.4, 2.5)

    points, sigmas, known_landmarks = generate_flower_data(dist, sigma)

    colors = {1: 'red', 2: 'orange', 3: 'green', 4: 'blue', 5: 'purple'}

    plt.figure(figsize=(8, 8))
    for i, point in enumerate(points):
        plt.scatter(point[0], point[1], s=20, color=colors[known_landmarks[i]], alpha=0.6)
    for i in range(5):
        plt.scatter(0, (i+1)*dist, s=20, color=colors[i+1], marker='X', label=f'Landmark {i+1}')
    plt.title('Simulated Flower Data with Gaussian Noise')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.grid()
    # plt.legend()
    plt.axis('equal')
    plt.show()
