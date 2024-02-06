import jax.numpy as jnp

from transformers import ElectraConfig, FlaxElectraForSequenceClassification
from transformers.models.electra.modeling_flax_electra import (
    FlaxElectraForSequenceClassificationModule,
)

import jax
import flax

from . import register_model, WrappedModel
from .scalar_mix import ScalarMix
from .kuma import HardKuma


@register_model("electra", config_cls=ElectraConfig)
class HardElectrarModel(WrappedModel):
    """A Electra-based classification module with a HardKuma attention layer on top"""

    num_labels: int
    config: ElectraConfig

    def setup(self):
        self.electra_module = FlaxElectraForSequenceClassificationModule(
            config=self.config
        )
        self.scalarmix = ScalarMix()
        self.layer_a = nn.Dense(
            self.config.hidden_size // self.config.num_attention_heads,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.layer_b = nn.Dense(
            self.config.hidden_size // self.config.num_attention_heads,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        deterministic: bool = True,
    ):

        if token_type_ids is None:
            token_type_ids = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        _, hidden_states, attentions = self.electra_module(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=True,
            output_attentions=True,
            unnorm_attention=True,
            deterministic=deterministic,
            return_dict=False,
        )

        embeddings = hidden_states[0]
        # outputs = self.scalarmix(hidden_states, attention_mask)


        
        
        a = jax.nn.softplus(self.layer_a(attn_logits))
        b = jax.nn.softplus(self.layer_b(attn_logits))
        a = jax.lax.clamp(1e-6, a, 100.0)  # extreme values could result in NaNs
        b = jax.lax.clamp(1e-6, b, 100.0)  # extreme values could result in NaNs
        support = jnp.array([-0.1, 1.1])
        dist = HardKuma([a, b], support=support)
        if self.training:
            z = z_dist.sample()  # [B, M, 1]
        else:
            mean = dist.mean()
            p0 = dist.prob(jnp.zeros(1))
            p1 = dist.prob(jnp.ones(1))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value
            attn_weights = jnp.where(pc < 0.5, jnp.where(p0 > p1, 0, 1), mean)


        outputs = self.electra_module.classifier(
            outputs[:, None, :], deterministic=deterministic
        )

        state = {
            "outputs": outputs,
            "hidden_states": hidden_states,
            "attentions": attentions,
        }
        return outputs, state

    # define gradient over embeddings
    def embeddings_grad_fn(
        self,
        inputs,
    ):
        def model_fn(word_embeddings, y):
            _, hidden_states, _ = self.electra_module.electra.encoder(
                word_embeddings,
                inputs["attention_mask"],
                head_mask=None,
                output_hidden_states=True,
                output_attentions=True,
                unnorm_attention=True,
                deterministic=True,
                return_dict=False,
            )
            outputs = self.scalarmix(hidden_states, inputs["attention_mask"])
            outputs = self.electra_module.classifier(
                outputs[:, None, :], deterministic=True
            )
            # we use sum over batch dimension since
            # we need batched gradient and because embeddings
            # on each sample are independent
            # summing will just retrieve the batched gradient
            return jnp.sum(outputs[jnp.arange(outputs.shape[0]), y], axis=0)

        return jax.grad(model_fn)

    def extract_embeddings(self, params):
        return (
            params["params"]["FlaxElectraForSequenceClassificationModule_0"]["electra"][
                "embeddings"
            ]["word_embeddings"]["embedding"],
            params["params"]["FlaxElectraForSequenceClassificationModule_0"]["electra"][
                "embeddings"
            ]["position_embeddings"]["embedding"],
        )

    @staticmethod
    def convert_to_new_checkpoint(old_params):
        keymap = {
            "FlaxElectraForSequenceClassificationModule_0": "electra_module",
            "ScalarMix_0": "scalarmix",
        }
        new_params = {"params": {}}
        for key, value in old_params["params"].items():
            if key in keymap:
                new_params["params"][keymap[key]] = value
            else:
                new_params["params"][key] = value

        return flax.core.freeze(new_params)

    @staticmethod
    def convert_to_old_checkpoint(new_params):
        keymap = {
            "electra_module": "FlaxElectraForSequenceClassificationModule_0",
            "scalarmix": "ScalarMix_0",
        }
        old_params = {"params": {}}
        for key, value in new_params["params"].items():
            if key in keymap:
                old_params["params"][keymap[key]] = value
            else:
                old_params["params"][key] = value

        return flax.core.freeze(old_params)

    @classmethod
    def initialize_new_model(
        cls,
        key,
        inputs,
        num_classes,
        identifier="google/electra-small-discriminator",
        **kwargs,
    ):
        model = FlaxElectraForSequenceClassification.from_pretrained(
            identifier,
            num_labels=num_classes,
        )
        classifier = cls(
            num_labels=num_classes,
            config=model.config,
        )
        params = classifier.init(key, **inputs)

        # replace original parameters with pretrained parameters
        params = params.unfreeze()
        assert "electra_module" in params["params"]
        params["params"]["electra_module"] = model.params
        params = flax.core.freeze(params)

        return classifier, params
