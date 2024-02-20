import jax
import numpy as np
import jax.numpy as jnp
from copy import deepcopy

from jax.config import config
# from jax.scipy.special import distance

from sklearn.metrics import adjusted_rand_score, \
    normalized_mutual_info_score, accuracy_score

config.update("jax_enable_x64", True)

N_BOOTSTRAPS = 100


class BGDCMfn:
    """Computes various Bootstrapped Gradient Descent Clustering Methods for Feature-rich Networks (BGDCMfn).
    Parameters
    ----------

    n_clusters : int, default=10
        The number of clusters to form as well as the number of
        centroids to generate.

    p : float, default=2
        P value in Minkowski distance (metric), otherwise it is ignored.

    init : {'k-means++', 'random', 'user'}, string, default='random'
        Method for initialization:
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence.
        'random': choose `n_clusters` observations (rows) at random from
        data for the initial centroids.
        'user': user-defined centroid indices for initial centroids.

    n_init : int, default=10
        Number of time the GDC algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=100
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tau : float, default=5e-2
        Tolerance threshold of the ||gradients||, mu ± tau*sigma of bootstrap means.

    batch_size = int, default=1,
        the size of the batch to apply the update rule.
        Fixing the batch size equal to one should be considered one of the main
         contribution of the proposed methods.

    mu_1 : float, default=9e-1,
          Exponential decay rate for the first moment estimates
          in Adam or decay rate for Nestrov GDC.

    mu_2 : float, default=999e-3,
          Exponential decay rate for the second moment estimates
           (i.e. squared gradients) in Adam.

    step_size : float, default=1e-3
        Gradient Descent step size, also known as learning rate to
         update the cluster centers.

    centroids_idx :  array of shape (n_clusters, 1), default = None
              A 1-D array containing user-defined seed indices to
               initiate centroids.

    verbose : int, default=0
        Verbosity mode.

    update_rule = str, default="agdc",
        one of the three following  cases are supported:
            1)  vgdc: applies vanilla gdc algorithm (VGDC);
            2) ngdc : applies gdc with Nestrov momentum algorithm (NGDC);
            3) agdc : applies gdc with Adam algorithm (Adam GDC).

    Attributes
    ----------

    """

    def __init__(
            self,
            n_clusters=10,
            *,
            p=2.,
            tau=5e-2,
            rho=1,
            xi=1,
            n_init=1,
            verbose=0,
            mu_1_x=45e-2,
            mu_2_x=95e-2,
            mu_1_a=45e-2,
            mu_2_a=95e-2,
            max_iter=100,
            init="random",
            batch_size=1,
            step_size=1e-2,
            centroids_idx=None,
            update_rule="vanilla",

    ):
        self.p = p
        self.tau = tau
        self.rho = rho
        self.xi = xi
        self.mu_1_x = mu_1_x
        self.mu_1_a = mu_1_a
        self.mu_2_x = mu_2_x
        self.mu_2_a = mu_2_a
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.max_iter = max_iter
        self.step_size = step_size
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.update_rule = update_rule
        self.centroids_idx = centroids_idx

        # Variable and attribute initialization
        self.idx = None
        self.ari = -jnp.inf
        self.nmi = -jnp.inf
        self.inertia = jnp.inf
        self.accuracy = -jnp.inf
        self.best_ari = -jnp.inf
        self.best_nmi = -jnp.inf
        self.moment_1_x = list()
        self.moment_2_x = list()
        self.moment_1_a = list()
        self.moment_2_a = list()
        self.data_scatter = jnp.inf
        self.best_inertia = jnp.inf
        self.batches = None
        self.clusters = None
        self.gdc_adam = False
        self.stop_type = None
        self.centroids_x = None
        self.centroids_a = None
        self._centroids_x = None
        self._centroids_a = None
        self.best_batches = None
        self.gdc_vanilla = False
        self.gdc_nestrov = False
        self.best_clusters = None
        self.best_centroids_idx = None
        self.best_history_ari = None
        self.best_history_nmi = None
        self.best_history_inertia = None
        self.best_history_grads_x = None
        self.best_history_grads_a = None
        self.best_history_centroid_x = None
        self.best_history_centroid_a = None
        self.best_history_grads_x_magnitude = None
        self.best_history_grads_a_magnitude = None

        # List of arrays s.t each inner list in converted to np.arr at the end of the iteration (bootstrapping).
        self.bootstrapping_grads_x = list()  # history of gradients of x, i.e. features, for iterations.
        self.bootstrapping_grads_a = list()  # history of gradients of a, i.e. networks, for iterations.
        self.bootstrapping_centroid_x = list()
        self.bootstrapping_centroid_a = list()
        self.bootstrapping_grads_x_magnitude = list()  # history of ||grads_x||, i.e., features.
        self.bootstrapping_grads_a_magnitude = list()
        self.bootstrapping_grads_batch_magnitude = list()  # history of ||grads_x|| + ||grads_a||
        self.mean_bootstrapping_grads_magnitude  = None  # mean of bootstrapping_resamples (first final statistics )
        self.std_bootstrapping_grads_magnitude  = None  # std of bootstrapping_resamples (first final statistics )
        self.mean_bootstrapping_centroids_x = None
        self.mean_bootstrapping_centroids_a = None

    @staticmethod
    def is_positive_semidefinite(m):
        return jax.tree_leaves(jnp.all(jnp.linalg.eigvals(m) >= 0))

    @staticmethod
    def is_approximately_zero(v):
        return None

    @staticmethod
    def _get_subkey():
        seed = np.random.randint(low=0, high=1e10, size=1)[0]
        key = jax.random.PRNGKey(seed)
        _, subkey = jax.random.split(key)
        return subkey

    @staticmethod
    def compute_data_scatter(m):
        return jnp.sum(jnp.power(m, 2))

    @staticmethod
    def get_batches(n_samples, batch_size):
        """ returns a list of lists, containing the shuffled indexes of the entire data points,
            split into approximately equal chunks."""

        samples_indexes = np.arange(0, n_samples)
        np.random.shuffle(samples_indexes)

        remainder = n_samples % batch_size
        if remainder != 0:
            n_splits = (n_samples // batch_size) + 1
        else:
            n_splits = n_samples // batch_size

        batches = [
            samples_indexes[batch_size * i:batch_size * (i + 1)] for i in range(n_splits)
        ]
        return batches

    @staticmethod
    def get_bootstrapping_batches(n_samples, n_resamples):
        """ returns a resampled list of data points' indexes for applying bootstrapping"""
        samples_indexes = np.arange(0, n_samples)
        return np.random.choice(samples_indexes, size=(n_resamples,), replace=True)

    def compute_inertia(self, m, centroids):
        return sum(
            [jnp.sum(
                jnp.power(m[jnp.where(self.clusters == k)] - centroids[k], 2),
            ) for k in range(self.n_clusters)
            ]
        )

    def compute_distances(self, data_point_x, data_point_a, distance_fn):
        distances_x = jnp.asarray([
            distance_fn(
                data_point=data_point_x, centroid=centroid, p=self.p) for centroid in self.centroids_x]
        )
        distances_a = jnp.asarray(
            [distance_fn(data_point=data_point_a, centroid=centroid, p=self.p) for centroid in self.centroids_a]
        )
        return distances_x + distances_a

    def compute_grads(self, data_point_x, data_point_a, distance_fn, k):

        grads_x = jax.jacfwd(distance_fn, argnums=(1,))(data_point_x, self.centroids_x[k, :], self.p)
        grads_x = jnp.nan_to_num(jax.tree_leaves(grads_x)[0])

        grads_a = jax.jacfwd(distance_fn, argnums=(1,))(data_point_a, self.centroids_a[k, :], self.p)
        grads_a = jnp.nan_to_num(jax.tree_leaves(grads_a)[0])

        return grads_x, grads_a

    def compute_hessians(self, data_point_x, data_point_a, distance_fn, k):
        hessian_mat_x = jax.hessian(distance_fn, argnums=(1,))(data_point_x, self.centroids_x[k, :], self.p)[0]
        hessian_mat_a = jax.hessian(distance_fn, argnums=(1,))(data_point_a, self.centroids_a[k, :], self.p)[0]

        return hessian_mat_x, hessian_mat_a

    def kmeans_plus_plus(self, x, a, distance_fn,):
        n_samples = x.shape[0]
        subkey = self._get_subkey()
        self.centroids_idx = jax.random.randint(
            subkey, minval=0, maxval=n_samples,
            shape=(1,)
        )
        for k in range(self.n_clusters-1):
            d_x = jnp.zeros((n_samples,), dtype=float)  # distances matrix
            d_a = jnp.zeros((n_samples,), dtype=float)  # distances matrix
            for i in range(n_samples):
                if i not in self.centroids_idx:
                    d_x = d_x.at[i].set(
                        distance_fn(data_point=x[i], centroid=x[self.centroids_idx[k], :], p=self.p)
                    )
                    d_x = d_x.at[i].set(
                        distance_fn(data_point=a[i], centroid=a[self.centroids_idx[k], :], p=self.p)
                    )
            d_t = d_x + d_a
            next_center = jnp.argmax(d_t)
            self.centroids_idx = jnp.append(self.centroids_idx, next_center)

        return self.centroids_idx

    def _initiate_centroids_idx(self, x, a, distance_fn):
        """Generates array of indices to initiate centroids."""

        if self.init.lower() == "k-means++":
            print(
                "\n Centroids are initialized using K-Means++. \n"
            )
            self.centroids_idx = self.kmeans_plus_plus(x=x, a=a, distance_fn=distance_fn,)

        elif self.init.lower() == "random":
            print(
                "\n Centroids are initialized randomly. \n"
            )
            subkey = self._get_subkey()
            self.centroids_idx = jax.random.randint(
                subkey, minval=0, maxval=x.shape[0],
                shape=(self.n_clusters,)
            )
        elif self.init.lower() == "user":
            print(
                "\n User defined centroids indices are being applied.\n"
            )
            if isinstance(self.centroids_idx, (jax.numpy.ndarray,)):
                self.centroids_idx = jax.numpy.asarray(self.centroids_idx)

        else:
            assert False, "\n Ill-defined centroids initialization method."

        return self.centroids_idx

    def _initiate_clusters(self, n_samples):
        """ Generates array of shape (n_samples * 1) to store cluster recovery results.
        ---
        parameter
        n_samples : int , default = None
            Indicates the number of samples, i.e., number of rows of a feature data matrix or
             number of rows of an adjacency matrix.
        """

        subkey = self._get_subkey()
        self.clusters = jax.random.randint(
            subkey, minval=self.n_clusters+1, maxval=self.n_clusters+1,  # 2*self.n_clusters,
            shape=(n_samples,)
        )

        return self.clusters

    def apply_gdc_rules(self, grads_x, grads_a, k, t):

        if self.gdc_vanilla:
            updated_centroid_x = self.centroids_x[k, :] - self.step_size * grads_x
            updated_centroid_a = self.centroids_a[k, :] - self.step_size * grads_a
            self.centroids_x = self.centroids_x.at[k].set(updated_centroid_x)
            self.centroids_a = self.centroids_a.at[k].set(updated_centroid_a)

        elif self.gdc_nestrov:
            previous_moment_1_x = deepcopy(self.moment_1_x)  # momentum
            self.moment_1_x = (self.mu_1_x * previous_moment_1_x) - (self.step_size * grads_x)
            updated_centroid_x = self.centroids_x[k, :] + self.moment_1_x
            self.centroids_x = self.centroids_x.at[k].set(updated_centroid_x)

            # networks
            previous_moment_1_a = deepcopy(self.moment_1_a)  # momentum
            self.moment_1_a = (self.mu_1_a * previous_moment_1_a) - (self.step_size * grads_a)
            updated_centroid_a = self.centroids_a[k, :] + self.moment_1_a
            self.centroids_a = self.centroids_a.at[k].set(updated_centroid_a)

        elif self.gdc_adam:
            previous_moment_1_x = deepcopy(self.moment_1_x)  # first moment (estimate) of gradients
            previous_moment_2_x = deepcopy(self.moment_2_x)  # second moment (square) of gradients
            tmp_moment_1_x = (self.mu_1_x * previous_moment_1_x) + \
                             (1 - self.mu_1_x) * grads_x
            tmp_moment_2_x = (self.mu_2_x * previous_moment_2_x) + \
                             (1 - self.mu_2_x) * (jnp.power(grads_x, 2))
            self.moment_1_x = tmp_moment_1_x / (1 - self.mu_1_x ** t)  # m in my notation
            self.moment_2_x = tmp_moment_2_x / (1 - self.mu_2_x ** t)  #

            updated_centroid_x = self.centroids_x[k, :] - (
                    (self.step_size * self.moment_1_x) / (jnp.sqrt(self.moment_2_x) + 1e-8)
            )
            self.centroids_x = self.centroids_x.at[k].set(updated_centroid_x)

            # networks
            previous_moment_1_a = deepcopy(self.moment_1_a)  # first moment (estimate) of gradients
            previous_moment_2_a = deepcopy(self.moment_2_a)  # second moment (square) of gradients
            tmp_moment_1_a = (self.mu_1_a * previous_moment_1_a) + \
                             (1 - self.mu_1_a) * grads_a
            tmp_moment_2_a = (self.mu_2_a * previous_moment_2_a) + \
                             (1 - self.mu_2_a) * (jnp.power(grads_a, 2))
            self.moment_1_a = tmp_moment_1_a / (1 - self.mu_1_a ** t)  # m in my notation
            self.moment_2_a = tmp_moment_2_a / (1 - self.mu_2_a ** t)

            updated_centroid_a = self.centroids_a[k, :] - (
                    (self.step_size * self.moment_1_a) / (jnp.sqrt(self.moment_2_a) + 1e-8)
            )
            self.centroids_a = self.centroids_a.at[k].set(updated_centroid_a)
        else:
            print("Unsupported update rule!")

        return updated_centroid_x, updated_centroid_a

    def apply_bootstrapping(self, x, a, distance_fn, y=None,):

        """Apply Bootstrapping on Various Gradient Descent Update rules to compute the statistics,
            i.e., the average and standard deviation of the magnitude of the gradients.
            Supported updates rules and algorithms:
                1) vanilla GDC (VGDC);
                2) Nestrov GDC (NGDC);
                3) Adam GDC (AGDC).

            Note:
             Only batch_size=1, which implements online gradient descent, is supported.

        Parameters
        ----------
        x : features array of shape (n_samples, n_features) or
             similarity data of shape (n_samples, n_samples) instances to cluster.
        a : network adjacency array of shape (n_samples, n_samples) instances to cluster.
        distance_fn : callable to compute the distance
             between data points and cluster centroids.
        y : array of shape (n_samples, 1), default = None
            Represents the ground truth.
        """

        # Compute the data scatter
        self.data_scatter = self.compute_data_scatter(m=x) + self.compute_data_scatter(m=a)

        # Clusters and centroids:
        n_samples = x.shape[0]
        n_features = x.shape[1]

        # Apply Bootstrapping for the same seeds to compute the statistics of gradients magnitude.
        for itr in range(N_BOOTSTRAPS):

            t = 0  # timestamp
            self.clusters = self._initiate_clusters(n_samples=n_samples)
            self.centroids_x = x[self.idx, :]
            self.centroids_a = a[self.idx, :]

            # Create batches
            self.batches = self.get_bootstrapping_batches(n_samples=n_samples, n_resamples=200)
            # Zero initialization of the moments vectors
            self.moment_1_x = jnp.zeros(n_features)
            self.moment_2_x = jnp.zeros(n_features)
            self.moment_1_a = jnp.zeros(n_samples)
            self.moment_2_a = jnp.zeros(n_samples)

            per_iter_grads_x = list()
            per_iter_grads_a = list()
            per_iter_centroid_x = list(list() for _ in range(self.n_clusters))
            per_iter_centroid_a = list(list() for _ in range(self.n_clusters))
            per_iter_grads_x_magnitude = list()
            per_iter_grads_a_magnitude = list()
            per_iter_grads_batch_magnitude = list()  # total magnitude of grads, i.e., ||grads_x|| + ||grads_a||

            for batch in self.batches:  # online
                t += 1
                batch_data_x = x[batch, :]
                batch_data_a = a[batch, :]
                # distances of size (K,)
                distances_x = jnp.asarray([
                    distance_fn(
                        data_point=batch_data_x, centroid=centroid, p=self.p) for centroid in self.centroids_x
                ])
                distances_a = jnp.asarray([
                    distance_fn(
                        data_point=batch_data_a, centroid=centroid, p=self.p) for centroid in self.centroids_a
                ])
                distances_t = self.rho * distances_x + self.xi * distances_a
                k = distances_t.argmin(axis=0)  # batch_clusters
                self.clusters = self.clusters.at[batch].set(k)

                # Compute gradients of the distance function w.r.t all centroids of the batch to update centroids
                batch_clusters = self.clusters[batch]
                # Compute grads of k-th cluster, i.e., the closest.
                # Features
                grads_x = jax.jacfwd(distance_fn, argnums=(1,))(
                    batch_data_x, self.centroids_x[k, :], self.p
                )
                # Links
                grads_x = jax.tree_leaves(grads_x)[0]
                grads_a = jax.jacfwd(distance_fn, argnums=(1,))(
                    batch_data_a, self.centroids_a[k, :], self.p
                )
                grads_a = jax.tree_leaves(grads_a)[0]
                magnitude_of_grads_x = jnp.sum(jnp.abs(grads_x))
                magnitude_of_grads_a = jnp.sum(jnp.abs(grads_a))
                batch_grads_magnitude = magnitude_of_grads_x + magnitude_of_grads_a

                # print(
                #     f"batch: {batch} with ||grads||={batch_grads_magnitude:.2f}, "
                # )

                updated_centroid_x, updated_centroid_a = self.apply_gdc_rules(
                    grads_x=grads_x, grads_a=grads_a, k=k, t=t
                )

                # track changes per each Bootstrapping resample, called iter here.
                per_iter_grads_x.append(grads_x)
                per_iter_grads_a.append(grads_a)
                per_iter_centroid_x[k].append(updated_centroid_x)
                per_iter_centroid_a[k].append(updated_centroid_a)
                per_iter_grads_x_magnitude.append(magnitude_of_grads_x)
                per_iter_grads_a_magnitude.append(magnitude_of_grads_a)
                per_iter_grads_batch_magnitude.append(batch_grads_magnitude)

                # Per data points
                if self.verbose >= 6:
                    print("printing Bootstrapping details:")
                    self.inertia = self.compute_inertia(m=x, centroids=self.centroids_x) + \
                                   self.compute_inertia(m=a, centroids=self.centroids_a)
                    if y is not None:
                        self.ari = adjusted_rand_score(y, self.clusters)
                        self.nmi = normalized_mutual_info_score(y, self.clusters)
                        self.accuracy = accuracy_score(y, self.clusters)
                    print(
                        f"data point: {batch}, K:{y[batch]}"
                        f" inertia: {self.inertia:.3f},"
                        f" ari:{self.ari:.3f},"
                        f" acc:{self.accuracy:.3f},"
                        f" ||grad_x||:{magnitude_of_grads_x:.3f},"
                        f" ||grad_a||:{magnitude_of_grads_a:.3f},"
                    )
                    if self.verbose >= 7:
                        print(
                            f"Gradient of data point in feature space: \n {grads_x} \n"
                            f"Gradient of data point in networks space: \n {grads_a} \n"
                        )

            # Save the results and histories before next iteration starts
            per_iter_grads_x = np.asarray(per_iter_grads_x)
            per_iter_grads_a = np.asarray(per_iter_grads_a)
            per_iter_grads_x_magnitude = np.asarray(per_iter_grads_x_magnitude)
            per_iter_grads_a_magnitude = np.asarray(per_iter_grads_a_magnitude)
            per_iter_grads_batch_magnitude = np.asarray(per_iter_grads_batch_magnitude)

            self.bootstrapping_grads_x.append(per_iter_grads_x)
            self.bootstrapping_grads_a.append(per_iter_grads_a)
            self.bootstrapping_grads_x_magnitude.append(per_iter_grads_x_magnitude)
            self.bootstrapping_grads_a_magnitude.append(per_iter_grads_a_magnitude)
            self.bootstrapping_grads_batch_magnitude.append(per_iter_grads_batch_magnitude)

            # keep track of updated centroids without step size
            # per_iter_centroid_x = list(np.asarray(cx).mean(axis=0) for cx in per_iter_centroid_x)
            # per_iter_centroid_a = list(np.asarray(ca).mean(axis=0) for ca in per_iter_centroid_a)
            # self.bootstrapping_centroid_x.append(per_iter_centroid_x)
            # self.bootstrapping_centroid_a.append(per_iter_centroid_a)

            self.bootstrapping_centroid_x.append(self.centroids_x)  # tracking the updated centroids with step size
            self.bootstrapping_centroid_a.append(self.centroids_a)  # tracking the updated centroids with step size


        self.mean_bootstrapping_centroids_x = np.asarray(self.bootstrapping_centroid_x).mean(axis=0)
        self.mean_bootstrapping_centroids_a = np.asarray(self.bootstrapping_centroid_a).mean(axis=0)

        self.mean_bootstrapping_grads_magnitude  = np.asarray(self.bootstrapping_grads_batch_magnitude).mean()
        self.std_bootstrapping_grads_magnitude  = np.asarray(self.bootstrapping_grads_batch_magnitude).std()

        print(
            f"Bootstrapping of {itr} resamples was computed for seeds: \n {self.idx}. \n"
            f" The final statistics: "
            f" Mean: {self.mean_bootstrapping_grads_magnitude :.2f} ± "
            f" Std: {self.std_bootstrapping_grads_magnitude :.2f} \n"
            f" Flatten centroids_x mean: {self.mean_bootstrapping_centroids_x.mean():.2f} &"
            f" Flatten centroids_a mean: {self.mean_bootstrapping_centroids_a.mean():.2f} \n"
        )

        return (self.mean_bootstrapping_grads_magnitude , self.std_bootstrapping_grads_magnitude ,
                self.mean_bootstrapping_centroids_x, self.mean_bootstrapping_centroids_a)

    def fit(self, x, a, distance_fn, y=None,):

        """Compute Various Gradient Descent Clustering Algorithm --with various update rules.
            Supported updates rules and algorithms:
                1) vanilla GDC (VGDC);
                2) Nestrov GDC (NGDC);
                3) Adam GDC (AGDC).

            Note:
             batch_size=1 implements stochastic gradient descent;
             batch_size=n_samples  implements batch gradient descent;
             1 < batch_size < n_samples  implements mini-batch gradient descent.

        Parameters
        ----------
        x : features array of shape (n_samples, n_features) or
             similarity data of shape (n_samples, n_samples) instances to cluster.
        a : network adjacency array of shape (n_samples, n_samples) instances to cluster.
        distance_fn : callable to compute the distance
             between data points and cluster centroids.
        y : array of shape (n_samples, 1), default = None
            Represents the ground truth.
        """

        if self.update_rule == "vgdc":
            self.gdc_vanilla = True
        elif self.update_rule == "ngdc":
            self.gdc_nestrov = True
        elif self.update_rule == "agdc":
            self.gdc_adam = True
        print(
            f"update rule: {self.update_rule} \n",
            f"{self.gdc_vanilla, self.gdc_nestrov, self.gdc_adam},"
        )
        if not isinstance(x, (jax.numpy.ndarray,)):
            x = jnp.asarray(x)
        if not isinstance(a, (jax.numpy.ndarray,)):
            a = jnp.asarray(a)
        if y is not None:
            if not isinstance(y, (jax.numpy.ndarray,)):
                y = jnp.asarray(y)
            if len(y.shape) > 1:
                y = y.reval()

        search_for_best_nmi_or_ari = True
        n_samples = x.shape[0]
        n_features = x.shape[1]

        # Repeat the computations with different seeds initialization to select the best results.
        for _ in range(self.n_init):

            # fixing seed idx
            self.idx = self._initiate_centroids_idx(x=x, a=a, distance_fn=distance_fn)
            print(
                f"init no. {_} seeds: {self.idx}"
            )

            # Apply Bootstrapping to obtain mean and std of ||grads|| using case resampling bootstrapping
            # with Monte Carlo algorithm per each random seed:
            mu, sigma, centroids_x, centroids_a = self.apply_bootstrapping(x=x, a=a, distance_fn=distance_fn, y=y)

            # Initialization:
            t = 0  # timestamp of boostraps computations
            # Clusters and centroids: using the centroids from Bootstrapping
            self.clusters = self._initiate_clusters(n_samples=n_samples)
            self.centroids_x = jnp.asarray(centroids_x) # x[self.idx, :]
            self.centroids_a = jnp.asarray(centroids_a)  # a[self.idx, :]
            self._centroids_x = deepcopy(centroids_x)  # An attribute to save init centroids for further analysis
            self._centroids_a = deepcopy(centroids_a)

            # Create batches
            self.batches = self.get_batches(n_samples=n_samples, batch_size=self.batch_size)

            # Zero initialization of the moments vectors
            self.moment_1_x = jnp.zeros(n_features)
            self.moment_2_x = jnp.zeros(n_features)
            self.moment_1_a = jnp.zeros(n_samples)
            self.moment_2_a = jnp.zeros(n_samples)

            # Compute the data scatter
            # self.data_scatter = self.compute_data_scatter(m=x) + self.compute_data_scatter(m=a)

            # list of arrays s.t each inner list in converted to np.arr at the end of the iteration.
            history_ari = list()
            history_nmi = list()
            history_inertia = list()
            history_grads_x = list()  # history of gradients of x, i.e. features, for iterations.
            history_grads_a = list()  # history of gradients of a, i.e. networks, for iterations.
            history_centroid_x = list()
            history_centroid_a = list()
            history_grads_x_magnitude = list()  # history of ||grads_x||, i.e., features.
            history_grads_a_magnitude = list()
            history_total_grads_magnitude = list()  # history of ||grads_x|| + ||grads_a||
            tracked_total_grads_magnitude = list()  # List of ( ||grad_x||+||grad_a|| ) of all batches used to update
            # the centroids up timestamp "t".

            self.stop_type = None

            print(
                f" Centroids_idx: {self.idx}, n_init: {_}"
                f"\n Best ARI: {self.best_ari:.3f}"
                f" Best NMI: {self.best_nmi:.3f}"
                f" Data Scatter: {self.data_scatter:.3f}"
                f" Best Inertia: {self.best_inertia: .3f}"
                f"\n Centroids: {self.centroids_idx}"
            )

            for itr in range(1, self.max_iter+1):

                per_iter_ari = list()
                per_iter_nmi = list()
                per_iter_inertia = list()
                per_iter_grads_x = list()
                per_iter_grads_a = list()
                per_iter_centroid_x = list()
                per_iter_centroid_a = list()
                per_iter_grads_x_magnitude = list()
                per_iter_grads_a_magnitude = list()
                per_iter_grads_t_magnitude = list()  # total magnitude of grads, i.e., ||grads_x|| + ||grads_a||

                for batch in self.batches: # online
                    assert 2 > len(batch), "Currently, only on-line GDC is supported!"
                    batch = jnp.squeeze(batch)
                    batch_data_x = x[batch, :]
                    batch_data_a = a[batch, :]

                    # distances of size (K,)
                    distances_x = jnp.asarray([
                        distance_fn(
                            data_point=batch_data_x, centroid=centroid, p=self.p) for centroid in self.centroids_x
                    ])

                    distances_a = jnp.asarray([
                        distance_fn(
                            data_point=batch_data_a, centroid=centroid, p=self.p) for centroid in self.centroids_a
                    ])

                    distances_t = self.rho * distances_x + self.xi * distances_a

                    k = distances_t.argmin(axis=0)  # batch_clusters
                    # print(f"k = {k}, {k[0]}")
                    # update clusters but not the weights, i.e. the centroids
                    self.clusters = self.clusters.at[batch].set(k)
                    previous_clusters = deepcopy(self.clusters)

                    # Compute gradients of the distance function w.r.t all centroids of the batch to update centroids
                    # batch_data_x = jnp.squeez(x[batch, :])
                    # batch_data_a = a[batch, :]
                    batch_clusters = self.clusters[batch]

                    # Compute grads of k-th cluster, i.e., the closest.
                    grads_x = jax.jacfwd(distance_fn, argnums=(1,))(
                        batch_data_x, self.centroids_x[k, :], self.p
                    )  # mean_batch_x >> batch_data_x
                    grads_x = jax.tree_leaves(grads_x)[0]

                    grads_a = jax.jacfwd(distance_fn, argnums=(1,))(
                        batch_data_a, self.centroids_a[k, :], self.p
                    )  # mean_batch_a  >> batch_data_a
                    grads_a = jax.tree_leaves(grads_a)[0]
                    magnitude_of_grads_x = jnp.sum(jnp.abs(grads_x))
                    magnitude_of_grads_a = jnp.sum(jnp.abs(grads_a))
                    batch_grads_magnitude = magnitude_of_grads_x + magnitude_of_grads_a

                    # Condition to decide which batch is appropriate to update centroids:
                    # If the magnitude  of sum the two gradients, i.e., ||grads_x|| + ||grads_a||,
                    # of the current data point, i.e., x[batch] and a[batch], is within [mu-sigma, mu+sigma],
                    # where, mu and sigma are the mean and standard deviation of the magnitude of gradients of all
                    # previously considered data points respectively, then the current batch is appropriate and the
                    # centroids will be updated.

                    low_thr = mu - self.tau * sigma
                    high_thr = mu + self.tau * sigma

                    if low_thr <= batch_grads_magnitude <= high_thr:

                        to_update = True
                        t += 1
                        # print(
                        #     f"Going to update the batch No. {batch} which is the {t} time using {self.update_rule}.\n"
                        #     f"clusters:{self.clusters}\n"
                        # )
                    else:
                        to_update = False

                    if to_update:
                        updated_centroid_x, updated_centroid_a = self.apply_gdc_rules(
                            grads_x=grads_x, grads_a=grads_a, k=k, t=t
                        )
                        # Per data points
                        if self.verbose >= 3:
                            self.inertia = self.compute_inertia(m=x, centroids=self.centroids_x) + \
                                           self.compute_inertia(m=a, centroids=self.centroids_a)
                            if y is not None:
                                self.ari = adjusted_rand_score(y, self.clusters)
                                self.nmi = normalized_mutual_info_score(y, self.clusters)
                                self.accuracy = accuracy_score(y, self.clusters)
                            print(
                                f"Data point: {batch},"
                                f" K:{y[batch]}"
                                f" inertia: {self.inertia:.3f},"
                                f" ari:{self.ari:.3f},"
                                f" acc:{self.accuracy:.3f}, \n"
                                f" BTS: mu {mu:.2f}, sigma {sigma:.2f},"
                                f" mu-sigma = {low_thr:.2f},"
                                f" mu+sigma = {high_thr:.2f}, \n"
                                f" ||grad_x||:{magnitude_of_grads_x:.2f},"
                                f" ||grad_a||:{magnitude_of_grads_a:.2f},"
                                f" batch_grads_magnitude: {batch_grads_magnitude:.2f} \n"
                            )
                            if self.verbose >= 5:
                                print(
                                    f"Gradient of data point in feature space: \n {grads_x} \n"
                                    f"Gradient of data point in networks space: \n {grads_a} \n"
                                )

                        # Per data point, i.e. per batch, inside one iteration
                        per_iter_grads_x.append(grads_x)
                        per_iter_grads_a.append(grads_a)
                        per_iter_centroid_x.append(updated_centroid_x)
                        per_iter_centroid_a.append(updated_centroid_a)
                        per_iter_grads_x_magnitude.append(magnitude_of_grads_x)
                        per_iter_grads_a_magnitude.append(magnitude_of_grads_a)
                        self.inertia = self.compute_inertia(m=x, centroids=self.centroids_x) + \
                                       self.compute_inertia(m=a, centroids=self.centroids_a)
                        per_iter_inertia.append(self.inertia)
                        if y is not None:
                            self.ari = adjusted_rand_score(y, self.clusters)
                            self.nmi = normalized_mutual_info_score(y, self.clusters)
                            per_iter_ari.append(self.ari)
                            per_iter_nmi.append(self.nmi)
                        else:
                            self.ari = -jnp.inf
                            self.nmi = -jnp.inf
                            per_iter_ari.append(self.ari)
                            per_iter_nmi.append(self.nmi)

                # Save the results and histories before next iteration starts
                per_iter_ari = np.asarray(per_iter_ari)
                per_iter_nmi = np.asarray(per_iter_nmi)
                per_iter_inertia = np.asarray(per_iter_inertia)
                per_iter_grads_x = np.asarray(per_iter_grads_x)
                per_iter_grads_a = np.asarray(per_iter_grads_a)
                per_iter_centroid_x = np.asarray(per_iter_centroid_x)
                per_iter_centroid_a = np.asarray(per_iter_centroid_a)
                per_iter_grads_x_magnitude = np.asarray(per_iter_grads_x_magnitude)
                per_iter_grads_a_magnitude = np.asarray(per_iter_grads_a_magnitude)

                history_ari.append(per_iter_ari)
                history_nmi.append(per_iter_nmi)
                history_inertia.append(per_iter_inertia)
                history_grads_x.append(per_iter_grads_x)
                history_grads_a.append(per_iter_grads_a)
                history_centroid_x.append(per_iter_centroid_x)
                history_centroid_a.append(per_iter_centroid_a)
                history_grads_x_magnitude.append(per_iter_grads_x_magnitude)
                history_grads_a_magnitude.append(per_iter_grads_a_magnitude)

                if self.verbose >= 4:
                    if y is not None and self.verbose != 0:
                        print(
                            f" N_itr = {itr} ARI = {self.ari:.3f} NMI = {self.nmi:.3f}"
                            f" Inertia = {self.inertia:.3f}"
                            f" Magnitude of gradient of X = {per_iter_grads_x_magnitude.mean():.3f}"
                            f" Gradients of X = {per_iter_grads_x.mean(axis=0)}"
                            f" Magnitude of gradient of A = {per_iter_grads_a_magnitude.mean():.3f}"
                            f" Gradients of A = {per_iter_grads_a.mean(axis=0)}"
                        )
                    elif y is None and self.verbose != 0:
                        print(
                            f" Inertia = {self.inertia:.3f}"
                            f" Magnitude of gradient of X = {per_iter_grads_x_magnitude.mean():.3f}"
                            f" Gradients of X = {per_iter_grads_x.mean(axis=0)}"
                            f" Magnitude of gradient of A = {per_iter_grads_a_magnitude.mean():.3f}"
                            f" Gradients of A = {per_iter_grads_a.mean(axis=0)}"
                        )

                    # Per iteration clusters' inertia convergence condition:
                    self.inertia = self.compute_inertia(
                        m=x, centroids=self.centroids_x) + self.compute_inertia(
                        m=a, centroids=self.centroids_a)

                if y is not None and self.verbose != 0:
                    print(
                        "\n"
                        f"N_iter = {itr}"
                        f" ARI = {self.ari:.3f}"
                        f" NMI = {self.nmi:.3f} "
                        f" Inertia = {self.inertia:.3f}\n"
                        f" Magnitude of gradients of X = {per_iter_grads_x_magnitude.mean():.3f} ±"
                        f" {per_iter_grads_x_magnitude.std():.3f} \n"
                        f" Magnitude of gradients of A = {per_iter_grads_a_magnitude.mean():.3f} ±"
                        f" {per_iter_grads_a_magnitude.std():.3f} \n"
                    )
                elif y is None and self.verbose != 0:
                    print(
                        "\n"
                        f"N_iter = {itr}"
                        f" Inertia = {self.inertia:.3f} \n"
                        f" Magnitude of gradients of X = {per_iter_grads_x_magnitude.mean():.3f} ±"
                        f" {per_iter_grads_x_magnitude.std():.3f} \n"
                        f" Magnitude of gradients of A = {per_iter_grads_a_magnitude.mean():.3f} ±"
                        f" {per_iter_grads_a_magnitude.std():.3f} \n"
                    )
                if self.verbose > 5:
                    print(
                        f"N_iter = {itr}: \n"
                        "Inertia History for all previous iterations: \n",
                        history_inertia, "\n",
                        "ARI History for all previous iterations: \n",
                        history_ari, "\n",
                        "NMI History for all previous iterations: \n",
                        history_nmi, "\n",
                        f" Mean of gradients of X = {per_iter_grads_x.mean(axis=0)} \n"
                        f" Std. of gradients of X = {per_iter_grads_x.std(axis=0)} \n"
                        f" Mean of gradients of A = {per_iter_grads_a.mean(axis=0)} \n"
                        f" Std. of gradients of A = {per_iter_grads_a.std(axis=0)} \n \n"
                    )

            # Comparing the obtained results with the previous results and save the best one.
            if y is not None and search_for_best_nmi_or_ari is True:
                self.ari = adjusted_rand_score(y, self.clusters)
                self.nmi = normalized_mutual_info_score(y, self.clusters)
                self.inertia = self.compute_inertia(
                    m=x, centroids=self.centroids_x) + self.compute_inertia(
                    m=a, centroids=self.centroids_a)
                if self.best_ari < self.ari:
                    self.best_ari = deepcopy(self.ari)
                    self.best_nmi = deepcopy(self.nmi)
                    self.best_centroids_idx = deepcopy(self.idx)
                    self.best_inertia = deepcopy(self.inertia)
                    self.best_batches = deepcopy(self.batches)
                    self.best_clusters = deepcopy(self.clusters)  # data with label >= n_clusters+1 are not clustered.
                    self.best_history_ari = deepcopy(history_ari)
                    self.best_history_nmi = deepcopy(history_nmi)
                    self.best_history_inertia = deepcopy(history_inertia)
                    self.best_history_grads_x = deepcopy(history_grads_x)
                    self.best_history_grads_a = deepcopy(history_grads_a)
                    self.best_history_centroid_x = deepcopy(history_centroid_x)
                    self.best_history_centroid_a = deepcopy(history_centroid_a)
                    self.best_history_grads_x_magnitude = deepcopy(history_grads_x_magnitude)
                    self.best_history_grads_a_magnitude = deepcopy(history_grads_a_magnitude)
            else:
                self.inertia = self.compute_inertia(
                    m=x, centroids=self.centroids_x) + self.compute_inertia(
                    m=a, centroids=self.centroids_a)
                if self.inertia < self.best_inertia:
                    self.best_centroids_idx = deepcopy(self.idx)
                    self.best_inertia = deepcopy(self.inertia)
                    self.best_batches = deepcopy(self.batches)
                    self.best_clusters = deepcopy(self.clusters)  # data with label >= n_clusters+1 are not clustered.
                    self.best_history_ari = deepcopy(history_ari)
                    self.best_history_nmi = deepcopy(history_nmi)
                    self.best_history_inertia = deepcopy(history_inertia)
                    self.best_history_grads_x = deepcopy(history_grads_x)
                    self.best_history_grads_a = deepcopy(history_grads_a)
                    self.best_history_centroid_x = deepcopy(history_centroid_x)
                    self.best_history_centroid_a = deepcopy(history_centroid_a)
                    self.best_history_grads_x_magnitude = deepcopy(history_grads_x_magnitude)
                    self.best_history_grads_a_magnitude = deepcopy(history_grads_a_magnitude)

        return self.best_clusters  # data points with label greater than n_clusters+1 are not clustered
