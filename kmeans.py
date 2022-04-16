import numpy as np

def cosine_distance(x1, x2):
    distance_arr = (x1 @ x2.T) / (np.linalg.norm(x1, axis=1, keepdims=True) @ np.linalg.norm(x2, axis=1, keepdims=True).T)
    distance_arr = 1 - distance_arr + 1e-8
    return distance_arr

def euclidean_distance(x1, x2):
    distance_arr = ((x1[:,np.newaxis,:] - x2)**2).sum(2)**0.5
    return distance_arr

class KMeansBase:
    def __init__(self, data, k, metric, seed):
        self.data = data
        self.k = k
        self.metric = metric
        np.random.seed(seed)

    def cluster(self):
        return self._lloyds_iterations()

    def _initial_centroids(self):
        # get the initial set of centroids
        centroid_indexes = np.random.choice(range(self.data.shape[0]), self.k, replace=False)
        # get the corresponding data points
        return self.data[centroid_indexes, :]

    def _lloyds_iterations(self):
        centroids = self._initial_centroids()
        stabilized = False

        j_values = []
        iterations = 0
        while (not stabilized) and (iterations < 1000):
            print ('iteration counter: ', iterations)
            try:
                # data array shape = n x m
                # centroids array shape = k x m
                if self.metric == 'cosine':
                    distance_arr = cosine_distance(self.data, centroids)
                elif self.metric == 'euclidean':
                    distance_arr = euclidean_distance(self.data, centroids)

                # Use a matrix of n x k where [i,j] = 1 if the ith data point belongs to cluster j.
                min_location = np.zeros(distance_arr.shape)
                min_location[range(distance_arr.shape[0]), np.argmin(distance_arr, axis=1)] = 1

                # calculate J
                j_val = np.sum(distance_arr[min_location == True])
                j_values.append(j_val)
                
                # calculates the new centroids
                new_centroids = np.empty(centroids.shape)
                for col in range(0, self.k):
                    if self.data[min_location[:, col] == True,:].shape[0] == 0:
                        new_centroids[col] = centroids[col]
                    else:
                        new_centroids[col] = np.mean(self.data[min_location[:, col] == True, :], axis=0)

                # compare centroids to see if they are equal or not
                if self._compare_centroids(centroids, new_centroids):
                    # it has resulted in the same centroids.
                    stabilized = True
                else:
                    centroids = new_centroids
            except:
                print ('exception!')
                continue
            else:
                iterations += 1

        print ('Required ', iterations, ' iterations to stabilize.')
        cls = np.argmax(min_location, axis=1)
        return iterations, j_values, centroids, cls

    def _compare_centroids(self, old_centroids, new_centroids, precision=-1):
        if precision == -1:
            return np.array_equal(old_centroids, new_centroids)
        else:
            diff = np.sum(new_centroids - old_centroids, axis=1)
            if np.max(diff) <= precision:
                return True
            else:
                return False

class KMeansPP(KMeansBase):
    def __init__(self, data, k, metric, seed):
        KMeansBase.__init__(self, data, k, metric, seed)

    def _initial_centroids(self):
        # pick the initial centroid randomly
        centroids = self.data[np.random.choice(range(self.data.shape[0]),1), :]

        # run k - 1 passes through the data set to select the initial centroids
        while centroids.shape[0] < self.k :
            if self.metric == 'cosine':
                distance_arr = cosine_distance(self.data, centroids)
            elif self.metric == 'euclidean':
                distance_arr = euclidean_distance(self.data, centroids)

            min_location = np.zeros(distance_arr.shape)
            min_location[range(distance_arr.shape[0]), np.argmin(distance_arr, axis=1)] = 1
            # calculate J
            j_val = np.sum(distance_arr[min_location == True])
            # calculate the probability distribution
            prob_dist = np.min(distance_arr, axis=1)/j_val
            # select the next centroid using the probability distribution
            centroids = np.vstack([centroids, self.data[np.random.choice(range(self.data.shape[0]),1, p = prob_dist), :]])
        return centroids