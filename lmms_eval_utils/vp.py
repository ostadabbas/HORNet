import torch
import jax
import numpy as np
import jax.numpy as jnp
from lmms_eval.videoprism.videoprism import models as vp


class FrozenVideoPrismEncoder:
    """
    A non-trainable wrapper around the JAX/Flax VideoPrism encoder.
    Produces video features with no gradients and no parameter updates.
    """

    def __init__(self, model_name="videoprism_public_v1_base"):
        # Load model + pretrained weights
        self.model_name = model_name
        self.flax_model = vp.get_model(model_name)
        self.params = vp.load_pretrained_weights(model_name)

        # JIT-compiled forward function
        @jax.jit
        def _forward(params, inputs):
            return self.flax_model.apply(params, inputs, train=False)

        self._forward = _forward

    def __call__(self, video_inputs):
        """
        video_inputs: jnp.ndarray of shape [B, T, H, W, 3]
        returns: features of shape [B, num_tokens, feature_dim]
        """
        outputs, _ = self._forward(self.params, video_inputs)
        return outputs

class TorchVideoPrism(torch.nn.Module):
    def __init__(self, vp_encoder):
        super().__init__()
        self.vp = vp_encoder  # instance of FrozenVideoPrismEncoder

    def forward(self, video_tensor):
        # Convert PyTorch → JAX
        video_np = jnp.array(video_tensor.cpu().numpy())

        # Run frozen encoder
        feats = self.vp(video_np)

        # Convert JAX → PyTorch
        return torch.from_numpy(np.array(feats)).to(video_tensor.device)