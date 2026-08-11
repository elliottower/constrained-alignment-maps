"""Expose `head_dim` on Qwen2Config for pyvene.

pyvene maps Qwen2's per-head components to `config.head_dim`
(`pyvene/models/qwen2/modelings_intervenable_qwen2.py`), but no released
transformers version defines that attribute on `Qwen2Config` — the attention
module computes it locally instead, as `hidden_size // num_attention_heads`
(`transformers/models/qwen2/modeling_qwen2.py:258`). Any pyvene intervention on
a Qwen attention head therefore raises `AttributeError` before it runs.

This supplies the same value the model itself uses, so it adds an attribute
rather than choosing one. Residual-stream interventions map to `hidden_size` and
never reach this path; MIB's IOI coefficient fitter patches attention heads and
does.
"""

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

if not hasattr(Qwen2Config, "head_dim"):
    Qwen2Config.head_dim = property(
        lambda self: self.hidden_size // self.num_attention_heads
    )
