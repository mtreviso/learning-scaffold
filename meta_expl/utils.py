from typing import Union, Tuple, Iterator
import json
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import common_utils

from optax import (
    GradientTransformation,
    chain,
    clip_by_global_norm,
    scale_by_adam,
    additive_weight_decay,
    scale,
)


def cross_entropy_loss(logits, targets):
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets"
            % (str(logits.shape), str(targets.shape))
        )
    onehot_targets = common_utils.onehot(targets, logits.shape[-1])
    loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
    return loss.mean()


def adamw_with_clip(
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 1e-4,
    max_norm: float = 1.0,
) -> GradientTransformation:
    return chain(
        clip_by_global_norm(max_norm),
        scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        additive_weight_decay(weight_decay),
        scale(-learning_rate),
    )


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


class PRNGSequence(Iterator[jax.random.PRNGKey]):
    """
    Iterator of JAX random keys.

    Inspired by the equivalent Haiku class
    """

    def __init__(self, key_or_seed: Union[jax.random.PRNGKey, int]):
        """Creates a new :class:`PRNGSequence`."""
        if isinstance(key_or_seed, int):
            key_or_seed = jax.random.PRNGKey(key_or_seed)

        self._key = key_or_seed

    def __next__(self) -> jax.random.PRNGKey:
        self._key, subkey = jax.random.split(self._key)
        return subkey

    next = __next__

    def take(self, num: int) -> Tuple[jax.random.PRNGKey, ...]:
        keys = jax.random.split(self._key, num + 1)
        self._key = keys[0]
        return keys[1:]


def multiply_no_nan(x, y):
    return jnp.where(x == 0, 0, x * y)


LOWER_CONST = 1e-7
UPPER_CONST = 1 - LOWER_CONST


def logprobs(probs):
    probs = jnp.maximum(probs, LOWER_CONST)
    probs = jnp.minimum(probs, UPPER_CONST)
    return jnp.log(probs)


def accumulate_gradient(loss_and_grad_fn, params, images, labels, accum_steps):
    """Accumulate gradient over multiple steps to save on memory."""
    if accum_steps and accum_steps > 1:
        assert (
            images.shape[0] % accum_steps == 0
        ), f"Bad accum_steps {accum_steps} for batch size {images.shape[0]}"
        step_size = images.shape[0] // accum_steps
        l, g = loss_and_grad_fn(params, images[:step_size], labels[:step_size])

        def acc_grad_and_loss(i, l_and_g):
            imgs = jax.lax.dynamic_slice(
                images, (i * step_size, 0, 0, 0), (step_size,) + images.shape[1:]
            )
            lbls = jax.lax.dynamic_slice(
                labels, (i * step_size, 0), (step_size, labels.shape[1])
            )
            li, gi = loss_and_grad_fn(params, imgs, lbls)
            l, g = l_and_g
            return (l + li, jax.tree_multimap(lambda x, y: x + y, g, gi))

        l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
        return jax.tree_map(lambda x: x / accum_steps, (l, g))
    else:
        return loss_and_grad_fn(params, images, labels)
