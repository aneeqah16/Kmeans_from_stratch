"""K-Means Clustering ALgorithm Implementation
This module contains core K-Means Clustering Algorithm and helper Functions.

Author: Aneeqah Ashraf
Date : Februray 2026
"""

import random
import numpy as np
class Kmeans():
    def __init__(self, K,max_iters = 100):
        self.K = K
        self.max_iters = max_iters 
    def distance(self,p1, p2):
        return np.linalg.norm(p1 - p2)

    def distance_squared(self,p1, p2):
        diff = np.array(p1) - np.array(p2)
        return np.dot(diff, diff)
    def init_centroids(self, points, K, seed=None):
        if K > len(points):
            raise ValueError(f"K ({K} cannot be greater than the number of points {len(points)}")

        if seed is not None:
            random.seed(seed)
        points_list = list(points)
        return [np.array(p, dtype=float) for p in random.sample(points_list, K)]
    
    def init_centroids_plusplus(self, points, seed=None):
        K = self.K
        points_array = np.array(points)

        if K > len(points_array):
            raise ValueError(f"K ({K}) cannot be greater than number of points ({len(points_array)})")

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        centroids = []

        # Choose first centroid randomly
        first_idx = random.randint(0, len(points_array) - 1)
        centroids.append(points_array[first_idx])

        # Choose remaining centroids
        for _ in range(K - 1):
            distances = []

            for point in points_array:
                min_dist = min(self.distance_squared(point, centroid) for centroid in centroids)
                distances.append(min_dist)

            distances = np.array(distances)
            total = distances.sum()

            if total == 0:
                next_centroid = points_array[random.randint(0, len(points_array) - 1)]
            else:
                probs = distances / total
                next_idx = np.random.choice(len(points_array), p=probs)
                next_centroid = points_array[next_idx]

            centroids.append(next_centroid)

        return centroids
    def assign_clusters(self, points, centroids):
        assignments = []
        
        for point in points:
            min_distance = float("inf")
            closest_centroid = 0

            for i, centroid in enumerate(centroids):
                dist = self.distance_squared(point, centroid)

                if dist < min_distance:
                    min_distance = dist
                    closest_centroid = i

            assignments.append(closest_centroid)

        return assignments

    def update_clusters(self, points, assignments):
        new_centroids = []

        for k in range(self.K):
            cluster_points = [points[i] for i in range(len(points)) if assignments[i] == k]

            if len(cluster_points) == 0:
                new_centroid = random.choice(points)
            else:
                cluster_points = np.array(cluster_points)
                new_centroid = cluster_points.mean(axis=0)
            new_centroids.append(new_centroid)
        return new_centroids

    def compute_sse(self, points, assignments, centroids):
        sse = 0.0

        for i, point in enumerate(points):
            clusters_id = assignments[i]
            centroid = centroids[clusters_id]
            sse += self.distance_squared(point, centroid)

        return sse
    def max_centroid_movement(self, old_centroids, new_centroids):
        max_movement = 0.0

        for old, new in zip(old_centroids, new_centroids):
            movement = self.distance(np.array(old), np.array(new))
            if movement > max_movement:
                max_movement = movement

        return max_movement
    def kmeans(self, points, tol=1e-4, init="random", verbose=False, seed=None):
        points = np.array(points)
        K = self.K

        if K < 1:
            raise ValueError("K must be at least 1")
        if K > len(points):
            raise ValueError(f"K ({K}) cannot be greater than number of points ({len(points)})")

        # Initialization
        if init == "kmeans++":
            centroids = self.init_centroids_plusplus(points, seed)
        elif init == "random":
            centroids = self.init_centroids(points, K, seed)
        else:
            raise ValueError("init must be 'random' or 'kmeans++'")

        previous_assignment = None
        iterations = 0

        for iteration in range(self.max_iters):
            iterations = iteration + 1

            assignments = self.assign_clusters(points, centroids)
            new_centroids = self.update_clusters(points, assignments)

            movement = self.max_centroid_movement(centroids, new_centroids)

            if verbose:
                print(f"Iteration {iteration + 1}, movement = {movement}")

            if previous_assignment is not None and previous_assignment == assignments:
                break

            if movement < tol:
                break

            centroids = new_centroids
            previous_assignment = assignments[:]

        sse = self.compute_sse(points, assignments, centroids)

        # Store results
        self.centroids = centroids
        self.assignments = assignments
        self.sse = sse
        self.iterations = iterations

        return centroids, assignments, sse, iterations
    
    def cluster_sizes(self):
        sizes = [0] * self.K
        for cluster_id in self.assignments:
            sizes[cluster_id] += 1
        return sizes
