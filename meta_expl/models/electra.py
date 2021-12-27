import flax.linen as nn
import jax.numpy as jnp
from transformers import ElectraConfig
from transformers.models.electra.modeling_flax_electra import (
    FlaxElectraForSequenceClassificationModule,
)


class ElectraModel(nn.Module):
    """A Electra-based classification module"""

    num_labels: int
    config: ElectraConfig

    @nn.compact
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

        electra_module = FlaxElectraForSequenceClassificationModule(config=self.config)
        outputs, hidden_states, attentions = electra_module(
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
        state = {"hidden_states": hidden_states, "attentions": attentions}
        return outputs, state
