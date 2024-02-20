import jax
import numpy as np
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)


class Distances:
    """ Implementation of a comprehensive set of distance metrics."""

    @staticmethod
    def minkowski_fn(data_point, centroid, p):
        """Computes p-valued Minkowski distance between two 1-D vectors.
        datapoint : array of shape (1, n_features) instance to cluster.
        centers: array of shape (1, n_features) clusters centers.
        """

        if centroid.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        if p < 1:
            raise ValueError("p must be at least 1")

        if data_point.ndim > 1:
            return (jnp.sum((jnp.abs(data_point - centroid)) ** p, axis=1)) ** 1 / p
        else:
            return (jnp.sum((jnp.abs(data_point - centroid)) ** p)) ** 1 / p

    @staticmethod
    def cosine_fn(data_point, centroid, p=None):
        """Computes cosine distance between two 1-D vectors.
        datapoint : array of shape (1, n_features) instance to cluster.
        centers : array of shape (1, n_features) clusters centers.
        P value is added due to consistencies issues.
        """
        if data_point.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        if centroid.ndim > 1:
            raise ValueError("Input vector should be 1-D.")

        return 1 - jnp.divide(
            (jnp.inner(data_point, centroid) + 1e-10),
            (jnp.sqrt(sum(data_point ** 2) + 1e-10) * jnp.sqrt(sum(centroid ** 2) + 1e-10))
        )

    @staticmethod
    def canberra_fn(data_point, centroid, p=None):
        if data_point.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        if centroid.ndim > 1:
            raise ValueError("Input vector should be 1-D.")

        return jnp.sum(
            jnp.divide(jnp.abs(data_point - centroid), (jnp.abs(data_point) + jnp.abs(centroid)))
        )

    @staticmethod
    def sigmoid_of_minkowski_fn(data_point, centroid, p=2):
        if data_point.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        if centroid.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        if data_point.ndim > 1:
            return jnp.divide(
                1, (1 + (jnp.sum((jnp.abs(data_point - centroid)) ** p, axis=1)) ** (1 / p))
            )
        else:
            return jnp.divide(
                1, (1 + ((jnp.sum((jnp.abs(data_point - centroid)) ** p)) ** (1 / p)))
            )

    @staticmethod
    def tanh_of_minkowski_fn(data_point, centroid, p=2):
        if data_point.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        if centroid.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        if data_point.ndim > 1:
            raise ValueError("Not implemented yet.")
            # return jnp.divide(
            #     1, (1 + (jnp.sum((jnp.abs(data_point - centroid)) ** p, axis=1)) ** (1 / p))
            # )
        else:
            return 1 - jnp.divide(
                (((jnp.sum((jnp.abs(data_point - centroid)) ** p)) ** (1 / p)) -
                 ((jnp.sum((jnp.abs(data_point - centroid)) ** p)) ** (1 / p))),
                (((jnp.sum((jnp.abs(data_point - centroid)) ** p)) ** (1 / p)) +
                 ((jnp.sum((jnp.abs(data_point - centroid)) ** p)) ** (1 / p)))
            )

    @staticmethod
    def sigmoid_of_normalized_minkowski_fn(data_point, centroid, p=2):
        if data_point.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        if centroid.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        if data_point.ndim > 1:
            return 1 - jnp.divide(
                1, (1 + (jnp.sum((jnp.abs(data_point - centroid)) ** p, axis=1) / len(centroid)) ** (1 / p))
            )
        else:
            return 1 - jnp.divide(
                1, (1 + ((jnp.sum((jnp.abs(data_point - centroid)) ** p) / len(centroid)) ** (1 / p)))
            )
