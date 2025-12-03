import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_flower_data(rows = 2, per_row = 4, density = 0.9, dist=0.1524, sigma=(0.014, 0.025)): # 6in <--> 0.1524 m   9in <--> 0.2286 m  12in <--> 0.3048 m
    # np.random.seed(42)  # make random draws reproducible
    map = []  # placeholder for flower map
    for i in range(per_row):
        for j in range(rows):
            if np.random.rand() < density:
                x = np.random.normal(loc=j * -dist, scale=np.random.uniform(*sigma))
                y = np.random.normal(loc=i * dist, scale=np.random.uniform(*sigma))
                map.append([x, y])
                # plt.plot(map[i,j][0], map[i,j][1], 'kx')  # plot flower location
    # plt.show()
    return map


def generate_measurement_data(map, sigma=(0.014, 0.025)):

    # make random draws reproducible
    np.random.seed(42)
    
    # randomly sample data around a point with gaussian noise of given sigma
    points = []
    sigmas = []
    known_landmarks = []

    #    Iterate through each landmark in the map
    for i, landmark in enumerate(map):
        mean = landmark
        sigma_sample = np.random.uniform(*sigma)
        samples = np.random.normal(loc=mean, scale=sigma_sample, size=(10, 2))
        for s in samples:
            points.append(s)
            sigmas.append(sigma_sample)
            known_landmarks.append(i+1)

        
    return points, sigmas, known_landmarks

def write_csv(points, sigmas, filename='simulated_flower_data.csv'):

    data = {
        'GPSparsed': points,
        'horizontal_accuracy': sigmas
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

if __name__ == "__main__":

    dist = 0.1524  # distance between flowers in meters (6 inches)
    sigma = (0.014, 0.025) # range of variances

    map = generate_flower_data(dist=dist, sigma=sigma)
    print(map)
    points, sigmas, known_landmarks = generate_measurement_data(map, sigma)
    # colors = {1: 'red', 2: 'orange', 3: 'green', 4: 'blue', 5: 'purple'}
    N = len(map) + 1
    cmap = plt.cm.tab20.colors  # gives a list of 20 colors
    colors = [cmap[i % 20] for i in range(N)]
    # plt.figure(figsize=(8, 8))
    plt.axis('equal')
    for i, point in enumerate(points):
        plt.scatter(point[0], point[1], s=10, color=colors[known_landmarks[i]], alpha=0.7, edgecolors=None)
    for i, landmark in enumerate(map):
        plt.scatter(landmark[0], landmark[1], s=10, color=colors[i+1], marker='X', edgecolors='k', label=f'Landmark {i+1}')
    plt.title('Simulated Flower Data with Gaussian Noise')
    plt.xlabel('N/S [m]')
    plt.ylabel('E/W [m]]')
    # plt.grid()
    # plt.legend()
    plt.axis('equal')
    write_csv(points, sigmas)
    plt.show()
