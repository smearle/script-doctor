import jax
from typing import Optional
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class Categorical:
    """Minimal JAX-only Categorical with logits API similar to replace distrax.Categorical"""
    logits: jnp.ndarray  # shape (..., K)
    dtype: jnp.dtype = jnp.int32  # dtype of sampled classes

    @property
    def num_classes(self) -> int:
        return self.logits.shape[-1]

    def sample(self, seed: jax.Array, shape: Optional[tuple] = None) -> jnp.ndarray:
        """
        Samples integers in [0, K) using logits.
        - If `shape` is None, returns shape (...,).
        - If `shape` is given, returns shape `shape + logits.shape[:-1]`.
        """
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        if shape is None:
            return jax.random.categorical(seed, log_probs, axis=-1).astype(self.dtype)
        # Tile logits to the requested leading shape
        tiled = jnp.broadcast_to(log_probs, tuple(shape) + log_probs.shape)
        flat = tiled.reshape((-1, tiled.shape[-1]))
        keys = jax.random.split(seed, flat.shape[0])
        samples = jax.vmap(lambda k, lp: jax.random.categorical(k, lp, axis=-1))(keys, flat)
        return samples.reshape(tuple(shape) + log_probs.shape[:-1]).astype(self.dtype)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        value: integer classes, shape must broadcast to logits.shape[:-1]
        returns: log p(value), shape broadcast of value/logits without class dim
        """
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        value = value.astype(jnp.int32)
        # Expand value to (..., 1) and gather along last axis.
        gathered = jnp.take_along_axis(
            log_probs, jnp.expand_dims(value, axis=-1), axis=-1
        )
        return jnp.squeeze(gathered, axis=-1)
    
    def entropy(self) -> jnp.ndarray:
        """Shannon entropy of the categorical distribution."""
        log_probs = jax.nn.log_softmax(self.logits)
        p = jnp.exp(log_probs)
        x = jnp.where(p == 0, 0.0, log_probs)
        return -jnp.sum(x*p, axis=-1)