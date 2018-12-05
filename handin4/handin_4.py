import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
# Load the Iris data set
import sklearn.datasets
import imageio
import matplotlib.pyplot as plt
import os

from sklearn.metrics.pairwise import distance_metrics
from sklearn.mixture import GaussianMixture as EM
from tensorflow.examples.tutorials.mnist import input_data

iris = sklearn.datasets.load_iris()
X = iris['data'][:, 0:2]  # reduce to 2d so you can plot if you want
true_Y = iris['target']
np.set_printoptions(precision=10)


######################################## part 1 ####################################################

def lloyds_algorithm(X, k, T):
    """ Clusters the data of X into k clusters using T iterations of Lloyd's algorithm.

        Parameters
        ----------
        X : Data matrix of shape (n, d)
        k : Number of clusters.
        T : Maximum number of iterations to run Lloyd's algorithm.

        Returns
        -------
        clustering: A vector of shape (n, ) where the i'th entry holds the cluster of X[i].
        centroids:  The centroids/average points of each cluster.
        cost:       The cost of the clustering
    """
    n, d = X.shape

    # Initialize clusters random.
    Y = np.random.randint(0, k, (n,))
    centroids = np.zeros((k, d))

    # Used to stop if cost isn't improving (decreasing)
    cost = 0
    oldcost = 0

    # Column names
    for i in range(T):
        # Update centroid
        # YOUR CODE HERE
        # END CODE
        # for every centroid and its centers
        for id, centroid in enumerate(centroids):
            cluster_points = []
            # find points corresponding to clusters
            for nr, point in enumerate(X):
                if Y[nr] == id:  # if clusters match
                    cluster_points.append(point)

            sums = np.zeros_like(X)
            cnt = 0
            for x in cluster_points:
                for dim in range(d):
                    sums[cnt, dim] = x[dim]
                cnt += 1
            for dim in range(d):
                if len(cluster_points) != 0:
                    centroids[id, dim] = np.array(sums[:, dim]).sum() / len(cluster_points)
                else:
                    centroids[id, dim] = 0

        if np.isnan(np.min(centroids)) == True:
            print(centroids)
            break

        # Update clustering
        # YOUR CODE HERE
        # for every point (row, (x,y))
        for id, value in enumerate(X):
            cluster_distance = []
            # for every centroid, calculate distance to a point
            for id_cluster, center in enumerate(centroids):
                cluster_distance.append(np.linalg.norm(value - center))
            # cluster a point based on min distance to a center
            Y[id] = np.argmin(cluster_distance)
        # END CODE

        # Compute and print cost
        cost = 0
        for j in range(n):
            cost += np.linalg.norm(X[j] - centroids[Y[j]]) ** 2
        # print(i + 1, "\t\t", cost)
        # Stop if cost didn't improve more than epislon (decrease)
        if np.isclose(cost, oldcost):
            break
        oldcost = cost

    return Y, centroids, cost


def compute_lloyd_clustering(k=3):
    Y, centroids, cost = lloyds_algorithm(X, k, 100)
    # fig = plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c=Y)
    # for i, j in centroids:
    #     plt.scatter(i, j, s=150, c='red', marker='x')
    # ax = fig.add_subplot(111)
    # fig.subplots_adjust(top=0.85)
    # ax.set_title("The result of clustering")
    # plt.show()
    return Y


def compute_probs_cx(points, means, covs, probs_c):
    '''
    Input
      - points: (n times d) array containing the dataset
      - means:  (k times d) array containing the k means
      - covs:   (k times d times d) array such that cov[j,:,:] is the covariance matrix of the j-th Gaussian.
      - priors: (k) array containing priors
    Output
      - probs:  (k times n) array such that the entry (i,j) represents Pr(C_i|x_j)
    '''
    # Convert to numpy arrays.
    points, means, covs, probs_c = np.asarray(points), np.asarray(means), np.asarray(covs), np.asarray(probs_c)

    # Get sizes
    n, d = points.shape
    k = means.shape[0]

    # Compute probabilities
    # This will be a (k, n) matrix where the (i,j)'th entry is Pr(C_i)*Pr(x_j|C_i).
    probs_cx = np.zeros((k, n))
    for i in range(k):
        try:
            probs_cx[i] = probs_c[i] * multivariate_normal.pdf(mean=means[i], cov=covs[i], x=points)
        except Exception as e:
            print("Cov matrix got singular: ", e)

    # The sum of the j'th column of this matrix is P(x_j); why?
    probs_x = np.sum(probs_cx, axis=0, keepdims=True)
    assert probs_x.shape == (1, n)

    # Divide the j'th column by P(x_j). The the (i,j)'th then
    # becomes Pr(C_i)*Pr(x_j)|C_i)/Pr(x_j) = Pr(C_i|x_j)
    probs_cx = probs_cx / probs_x

    return probs_cx, probs_x


def em_algorithm(X, k, T, epsilon=0.001, means=None):
    """ Clusters the data X into k clusters using the Expectation Maximization algorithm.

        Parameters
        ----------
        X : Data matrix of shape (n, d)
        k : Number of clusters.
        T : Maximum number of iterations
        epsilon :  Stopping criteria for the EM algorithm. Stops if the means of
                   two consequtive iterations are less than epsilon.
        means : (k times d) array containing the k initial means (optional)

        Returns
        -------
        means:     (k, d) array containing the k means
        covs:      (k, d, d) array such that cov[j,:,:] is the covariance matrix of
                   the Gaussian of the j-th cluster
        probs_c:   (k, ) containing the probability Pr[C_i] for i=0,...,k.
        llh:       The log-likelihood of the clustering (this is the objective we want to maximize)
    """
    n, d = X.shape

    # Initialize and validate mean
    if means is None:
        means = np.random.rand(k, d)

    # Initialize cov, prior
    probs_x = np.zeros(n)
    probs_cx = np.zeros((k, n))
    probs_c = np.zeros(k) + np.random.rand(k)

    covs = np.zeros((k, d, d))
    for i in range(k): covs[i] = np.identity(d)
    probs_c = np.ones(k) / k

    # Column names
    # print("Iterations\tLLH")

    close = False
    old_means = np.zeros_like(means)
    iterations = 0
    while not (close) and iterations < T:
        old_means[:] = means

        new_means = np.zeros_like(means)
        new_probs = np.zeros(k) + np.random.rand(k)
        new_covs = np.zeros((k, d, d))

        # Expectation step
        probs_cx, probs_x = compute_probs_cx(X, means, covs, probs_c)
        assert probs_cx.shape == (k, n)

        # Maximization step
        # YOUR CODE HERE
        for k_index in range(k):  # for all clusters(gaussian dists.)
            for n_index in range(n):  # for all observations(points)
                # calculating means of distributions
                new_means[k_index] += probs_cx[k_index, n_index] * X[n_index]
            new_means[k_index] = new_means[k_index] / probs_cx[k_index, :].sum()

        for k_index in range(k):  # for all clusters(gaussian dists.)
            for n_index in range(n):  # for all observations(points)
                # calculating covariance between latest prediced dist. and observations
                ys = np.reshape(X[n_index] - new_means[k_index], (2, 1))
                new_covs[k_index] += probs_cx[k_index, n_index] * np.dot(ys, ys.T)
            new_covs[k_index] = new_covs[k_index] / probs_cx[k_index, :].sum()

        for means_index in range(len(means)):
            for n_index in range(n):
                # update probability for each cluster
                new_probs[means_index] = new_probs[means_index] + probs_cx[means_index, n_index]
        new_probs = 1 / n * new_probs  # Pc_i, prob of being in cluster i

        # END CODE

        # Compute per-sample average log likelihood (llh) of this iteration
        llh = 1 / n * np.sum(np.log(probs_x))
        # print(iterations + 1, "\t\t", llh)

        if np.isnan(np.min(new_covs)) == False:
            covs = new_covs
            probs_c = new_probs
            means = new_means
        elif np.isnan(np.min(new_covs)) == True:
            break

        # Stop condition
        dist = np.sqrt(((means - old_means) ** 2).sum(axis=1))
        close = np.all(dist < epsilon)
        iterations += 1

    # Validate output
    assert means.shape == (k, d)
    assert covs.shape == (k, d, d)
    assert probs_c.shape == (k,)

    return means, covs, probs_c, llh


def compute_em_clusters(means, covs, probs_c, data):
    probs_cx = compute_probs_cx(data, means, covs, probs_c)[0]
    clustering = probs_cx.argmax(axis=0)

    return clustering


def cluster_with_em(k=3):
    means, covs, probs, lik = em_algorithm(X, k, 100)
    clusters = compute_em_clusters(means, covs, probs, X)

    # fig = plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c=clusters)
    # ax = fig.add_subplot(111)
    # fig.subplots_adjust(top=0.85)
    # ax.set_title("The result of clustering with the EM algorithm")
    # plt.show()
    return clusters


def best_lloyd_centers(limit=100, k=3):
    costs = []
    centers = []

    for i in range(limit):
        costs.append(lloyds_algorithm(X, k, 100)[2])
        centers.append(lloyds_algorithm(X, k, 100)[1])

    the_best_index = np.argmin(costs)
    return centers[the_best_index]


def use_lloyd_on_em(iter=100, k=3):
    centers = best_lloyd_centers(iter, k=k)

    means, covs, probs, lik = em_algorithm(X, k, 100, means=centers)
    clusters = compute_em_clusters(means, covs, probs, X)
    return (clusters)


######################################## part 2 ####################################################

def intra_cluster_distance(X, labels):
    k = len(np.unique(labels))

    mean_dist = []

    for label in range(k):
        collect = []
        for x_1 in range(len(X)):
            for x_2 in range(len(X)):
                if labels[x_1] == label and x_1 != x_2 and labels[x_2] == label:
                    euclid_norm = np.linalg.norm(X[x_1] - X[x_2])
                    collect.append(euclid_norm)
        mean_dist.append(np.mean(collect))
    return np.array(mean_dist)


def _nearest_cluster_distance(X, labels, i):
    label = labels[i]
    b = np.min(
        [np.mean(
            [np.linalg.norm(X[i] - X[j]) for j in np.where(labels == cur_label)[0]]
        ) for cur_label in set(labels) if not cur_label == label])
    return b

def silhouette(X, labels):
    A = intra_cluster_distance(X, labels)
    #print(A)
    B = np.array(np.mean([_nearest_cluster_distance(X, labels, i) for i in range(len(X))]))
    #print(B)
    silh_samples = (B - A) / np.maximum(A, B)

    silh = np.mean(np.nan_to_num(silh_samples))
    # END CODE
    print(silh)
    return silh


def f1(predicted, labels):
    n, = predicted.shape
    assert labels.shape == (n,)
    r = np.max(predicted) + 1
    k = np.max(labels) + 1

    # Implement the F1 score here
    # YOUR CODE HERE
    contingency = np.zeros((r, k))
    for x in range(len(predicted)):
        contingency[predicted[x], labels[x]] += 1

    pres = np.divide(contingency.max(axis=1), contingency.sum(axis=1))
    recall = ((contingency[:, contingency.argmax(axis=1)]).max(axis=1) / (
        contingency[:, contingency.argmax(axis=1)]).sum(axis=0))

    F_individual = (2 * pres * recall) / (pres + recall)
    # print("F_individual", F_individual)
    F_overall = sum(F_individual) / r

    # END CODE

    assert contingency.shape == (r, k)
    return F_individual, F_overall, contingency


def test_k_with_silhouette():
    results_sill = np.zeros((10, 3))

    results_f1 = np.zeros((10, 3))

    for k in range(2, 7, 1):
        print(k)
        predictions_em = cluster_with_em(k=k)
        predictions_lloyd_em = use_lloyd_on_em(100, k=k)
        predictions_lloyd = compute_lloyd_clustering(k=k)
        results_sill[k, 2] = silhouette(X, predictions_lloyd_em)
        results_sill[k, 0] = silhouette(X, predictions_lloyd)
        results_sill[k, 1] = silhouette(X, predictions_em)

        results_f1[k, 2] = f1(predictions_lloyd_em, true_Y)[1]
        results_f1[k, 0] = f1(predictions_lloyd, true_Y)[1]
        results_f1[k, 1] = f1(predictions_em, true_Y)[1]

    print(np.matrix(results_sill))
    print(np.matrix(results_f1))


# producing tables for f1 and silhouette
test_k_with_silhouette()

######################################## part 3 ####################################################

def download_image(url):
    filename = url[url.rindex('/') + 1:]
    try:
        with open(filename, 'rb') as fp:
            return imageio.imread(fp) / 255
    except FileNotFoundError:
        import urllib.request
        with open(filename, 'w+b') as fp, urllib.request.urlopen(url) as r:
            fp.write(r.read())
            return imageio.imread(fp) / 255


def compress_kmeans(im, k, T, name):
    height, width, depth = im.shape
    data = im.reshape((height * width, depth))
    clustering, centroids, score = lloyds_algorithm(data, k, 5)

    # make each entry of data to the value of it's cluster
    data_compressed = data

    for i in range(k): data_compressed[clustering == i] = centroids[i]

    im_compressed = data_compressed.reshape((height, width, depth))

    # The following code should not be changed.
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(im_compressed)
    plt.savefig("compressed.jpg")
    plt.show()

    original_size = os.stat(name).st_size
    compressed_size = os.stat('compressed.jpg').st_size
    print("Original Size: \t\t", original_size)
    print("Compressed Size: \t", compressed_size)
    print("Compression Ratio: \t", round(original_size / compressed_size, 5))


def compress_facade(img, name, k=5, T=20):
    img_facade = download_image(img)
    compress_kmeans(img_facade, k, T, name)


def compressing_image():
    img = 'https://uploads.toptal.io/blog/image/443/toptal-blog-image-1407508081138.png'
    img_facade = download_image(img)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img_facade)
    plt.show()

    size = os.stat('toptal-blog-image-1407508081138.png').st_size

    print("The image consumes a total of %i bytes. \n" % size)
    print("You should compress your image as much as possible! ")

    compress_facade(img, "toptal-blog-image-1407508081138.png")

    img_facade = download_image(img)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img_facade)
    plt.show()

    size = os.stat('toptal-blog-image-1407508081138.png').st_size

    print("The image consumes a total of %i bytes. \n" % size)
    print("You should compress your image as much as possible! ")


# compressing_image()


def part4_mnist():
    mnist = input_data.read_data_sets("data/")

    X = mnist.train.images
    y = mnist.train.labels

    # One cluster for each digit
    k = 10

    # Run EM algorithm on 1000 images from the MNIST dataset.
    expectation_maximization = EM(n_components=k, max_iter=10, init_params='kmeans', covariance_type='diag', verbose=1,
                                  verbose_interval=1).fit(X)

    means = expectation_maximization.means_
    covs = expectation_maximization.covariances_

    fig, ax = plt.subplots(1, k, figsize=(8, 1))

    for i in range(k):
        ax[i].imshow(means[i].reshape(28, 28), cmap='gray')

    plt.show()

    sample(means, covs, 0)


def sample(means, covs, num):
    mean = means[num]
    cov = covs[num]

    fig, ax = plt.subplots(1, 10, figsize=(8, 1))

    for i in range(10):
        img = multivariate_normal.rvs(mean=mean, cov=cov)  # draw random sample
        ax[i].imshow(img.reshape(28, 28), cmap='gray')  # draw the random sample
    plt.show()

# part4_mnist()
