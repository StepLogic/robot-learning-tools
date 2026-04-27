from functools import partial
import jax
import jax.numpy as jnp
from jax import random as jax_random
from transformers import AutoImageProcessor, FlaxDinov2Model
import flax.linen as nn
from typing import Any
import numpy as np


class TemporalEncoder(nn.Module):
    embed_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Args:
            x: [batch, frames, 16, 16, feature_dim]
        """
        batch, _, _, _ = x.shape
        # x = x.reshape(batch, frames, 16, 16, feature_dim)
        # breakpoint()
        # 2. 3D Convolutional Layers
        # Layer 1: [batch, 4, 16, 16, 64]
        # print(x.shape)
        h = nn.Conv(features=8, kernel_size=(3, 3, 3), padding='SAME')(x)
        # print(h.shape)
        h = nn.LayerNorm()(h)
        h = nn.gelu(h)
        
        # [batch, 2, 8, 8, 128]
        # print(h.shape)
        h = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(h)
        # print(h.shape)
        h = nn.LayerNorm()(h)
        h = nn.gelu(h)

        # Layer 3: Final reduction
        # [batch, 1, 4, 4, 256]
        # print(h.shape)
        h = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(h)
        # print(h.shape)
        h = nn.LayerNorm()(h)
        h = nn.gelu(h)

        # 3. Flatten and Project
        # Size: batch * (1 * 4 * 4 * 256) = batch * 4096
        # print("Before Flatten",h.shape,x.shape)
        h = h.reshape((batch, -1))
        # print("After Flatten",h.shape)
        # h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h)
        # print("Dropout",h.shape)
        h = nn.Dense(64)(h)
        # print("NN Shape",h.shape)
        h = nn.gelu(h)
        h = nn.Dense(self.embed_dim)(h)
        # print(h.shape)
        return h
    
# class TemporalEncoder(nn.Module): 
#     embed_dim: int
#     dropout_rate: float = 0.1
    
#     @nn.compact
#     def __call__(self, x, training: bool = False):
#         """
#         Args:
#             x: [batch, num_frames, num_tokens, feature_dim]
#         Returns:
#             [batch, embed_dim]
#         """
#         # batch, num_frames, num_tokens, feature_dim = x.shape
        
#         # Reshape: [batch, frames, 16, 16, feature_dim]
#         x = x[:, :, 1:, :]  # Remove CLS
#         # x = x.reshape(batch, num_frames, 16, 16, feature_dim)
        
#         # Single 3D conv with larger kernel
#         h = nn.Conv(
#             features=128,
#             kernel_size=(3, 5, 5),  # Larger spatial receptive field
#             strides=(1, 2, 2),
#             padding='SAME'
#         )(x)
#         h = nn.LayerNorm()(h)
#         h = nn.gelu(h)
        
#         # Another 3D conv
#         h = nn.Conv(
#             features=256,
#             kernel_size=(3, 3, 3),
#             strides=(1, 2, 2),
#             padding='SAME'
#         )(h)
#         h = nn.LayerNorm()(h)
#         h = nn.gelu(h)
        
#         # Global pool and project
#         h = jnp.mean(h, axis=(1, 2, 3))
#         # h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h)
#         h = nn.Dense(self.embed_dim)(h)
#         return h

# class DINOv2Encoder(nn.Module):
#     """
#     DINOv2 encoder with optional temporal transformer.
#     Processes either single images or sequences of images.
#     """
#     model_name: str = "facebook/dinov2-small"
#     use_temporal: bool = True
#     num_frames: int = 4
#     num_heads: int = 6
#     num_layers: int = 4
#     dropout_rate: float = 0.1
#     freeze_dinov2: bool = True  # Freeze DINOv2 weights
    
#     def setup(self):
#         # Create temporal transformer if needed
#         if self.use_temporal:
#             self.temporal_transformer = TemporalEncoder(
#                 embed_dim=387,
#                 dropout_rate=self.dropout_rate
#             )


    
#     @nn.compact
#     def __call__(self, x, training: bool = False):

#         # is_sequence =   # [batch, frames, C, H, W]
#         if x.ndim == 5:
#             batch_size, num_frames, channels, height, width = x.shape
#             x = x.reshape(batch_size * num_frames, channels, height, width)
#             # features = self.extract_dinov2_features(x)
#             x = x.reshape(batch_size, num_frames, 257 ,self.feature_dim)
#             # print(features.shape)
#             x = self.temporal_transformer(x, training=training)
#         else:
#             # print(x.shape)
#             # num_frames, channels, latent = x.shape
#             # features = self.extract_dinov2_features(x)
#             x = x[None,...]
#             x = self.temporal_transformer(x, training=training)
#             x = x.squeeze(0)  # 
#         # output = jnp.mean(temporal_features, axis=1)  # [batch, feature_dim]
#         # breakpoint()
        
#         return x

# class DINOv2Encoder(nn.Module):
#     """
#     Optimized DINOv2 encoder with temporal transformer.
#     """
#     model_name: str = "facebook/dinov2-small"
#     use_temporal: bool = True
#     num_frames: int = 4
#     num_heads: int = 6
#     num_layers: int = 4
#     dropout_rate: float = 0.1
#     freeze_dinov2: bool = True
    
#     def setup(self):
#         self.dinov2_model = FlaxDinov2Model.from_pretrained(self.model_name)
#         self.feature_dim = self.dinov2_model.config.hidden_size
        
#         # Precompute normalization constants
#         self.register_buffer('mean', jnp.array([0.485, 0.456, 0.406]))
#         self.register_buffer('std', jnp.array([0.229, 0.224, 0.225]))
        
#         if self.use_temporal:
#             self.temporal_transformer = TemporalEncoder(
#                 embed_dim=self.feature_dim,
#                 dropout_rate=self.dropout_rate
#             )
    
#     def register_buffer(self, name, value):
#         """Helper to store constants (will be JIT-compiled as constants)"""
#         setattr(self, f'_{name}', value)
    
#     # @partial(jax.jit, static_argnums=(0,))
#     def _preprocess(self, pixel_values):
#         """
#         Optimized preprocessing pipeline.
#         Combines resize + normalize + transpose in one pass.
#         """
#         # Resize to 224x224 (vectorized)
#         if pixel_values.shape[-2:] != (224, 224):
#             pixel_values = jax.image.resize(
#                 pixel_values,
#                 (pixel_values.shape[0], 224, 224, 3),
#                 method="bilinear"
#             )
        
#         # Normalize (vectorized)
#         mean = jnp.array([0.485, 0.456, 0.406])
#         std = jnp.array([0.229, 0.224, 0.225])
#         pixel_values = (pixel_values - mean) / std
        
#         # Transpose to (N, C, H, W)
#         pixel_values = jnp.transpose(pixel_values, (0, 3, 1, 2))
        
#         return pixel_values
    
#     def extract_dinov2_features(self, pixel_values):
#         """
#         Extract features from DINOv2 model.
#         Optimized to avoid redundant operations.
#         """
#         # Preprocess
#         pixel_values = self._preprocess(pixel_values)
        
#         # Forward pass through DINOv2
#         outputs = self.dinov2_model(
#             pixel_values,
#             params=self.dinov2_model.params
#         )
        
#         # Extract features and stop gradients if frozen
#         features = outputs.last_hidden_state  # (batch, 257, feature_dim)
        
#         if self.freeze_dinov2:
#             features = jax.lax.stop_gradient(features)
        
#         return features
    
#     @nn.compact
#     def __call__(self, x, training: bool = False):
#         """
#         Forward pass with optimized batch handling.
        
#         Args:
#             x: Input images
#                - 5D: (batch, num_frames, C, H, W)
#                - 4D: (num_frames, C, H, W)
        
#         Returns:
#             Temporal features (batch, num_frames, num_tokens, feature_dim)
#             or (num_frames, num_tokens, feature_dim) for 4D input
#         """
#         # Normalize input to 5D (batch dimension)
#         is_batched = x.ndim == 5
        
#         if is_batched:
#             batch_size, num_frames, channels, height, width = x.shape
#             # Merge batch and frames for efficient processing
#             x_flat = x.reshape(batch_size * num_frames, channels, height, width)
#         else:
#             num_frames, channels, height, width = x.shape
#             x_flat = x[None,...]
#             batch_size = 1
        
#         # Extract DINOv2 features (single forward pass)
#         features = self.extract_dinov2_features(x_flat)
        
#         # Reshape to (batch, num_frames, num_tokens, feature_dim)
#         features = features.reshape(batch_size, num_frames, 257, self.feature_dim)
        
#         # Apply temporal transformer
#         if self.use_temporal:
#             temporal_features = self.temporal_transformer(features, training=training)
#         else:
#             temporal_features = features
        
#         # Return in original format
#         # print(temporal_features.shape)
#         if not is_batched:
#             temporal_features = temporal_features.squeeze(0)
        
#         return temporal_features

class DINOv2Encoder(nn.Module):
    """
    DINOv2 encoder with optional temporal transformer.
    Processes either single images or sequences of images.
    """
    model_name: str = "facebook/dinov2-small"
    use_temporal: bool = True
    num_frames: int = 4
    num_heads: int = 6
    num_layers: int = 4
    dropout_rate: float = 0.1
    freeze_dinov2: bool = True  # Freeze DINOv2 weights
    
    def setup(self):
        # self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.dinov2_model = FlaxDinov2Model.from_pretrained(self.model_name)
        self.feature_dim = 64
        
        # Create temporal transformer if needed
        if self.use_temporal:
            self.temporal_transformer = TemporalEncoder(
                embed_dim=self.feature_dim,
                dropout_rate=self.dropout_rate
            )

    
    def extract_dinov2_features(self, pixel_values, params=None):
        # Use provided params or model's params
        dinov2_params = params if params is not None else self.dinov2_model.params
        # breakpoint()
        # pixel_values = self.processor(images=pixel_values, return_tensors="np")
        pixel_values = jax.image.resize(pixel_values, (pixel_values.shape[0], 224, 224, 3), method="bilinear")
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])

        pixel_values = (pixel_values - mean) / std
        if pixel_values.ndim == 4:
            pixel_values = jnp.transpose(pixel_values, (0, 3, 1, 2))
        elif pixel_values.ndim == 3:
            pixel_values = jnp.transpose(pixel_values, (2, 0, 1))

        
        outputs = self.dinov2_model(pixel_values, params=dinov2_params)
        # breakpoint()
        cls_features = outputs.last_hidden_state  # [batch, feature_dim]

        # Stop gradients if freezing DINOv2
        # if self.freeze_dinov2:
        cls_features = jax.lax.stop_gradient(cls_features)
        
        return cls_features
    
    @nn.compact
    def __call__(self, x, training: bool = False):

        # is_sequence =   # [batch, frames, C, H, W]
        if x.ndim == 5:
            batch_size, num_frames, channels, height, width = x.shape
            x = x.reshape(batch_size * num_frames, channels, height, width)
            features = self.extract_dinov2_features(x)
            features = features.reshape(batch_size, num_frames, 257 ,self.dinov2_model.config.hidden_size)
            # print(features.shape)
            temporal_features = self.temporal_transformer(features, training=training)
        else:
            num_frames, channels, height, width = x.shape
            features = self.extract_dinov2_features(x)
            features = features[None,...]
            temporal_features = self.temporal_transformer(features, training=training)
            temporal_features = temporal_features.squeeze(0)  # 
        return temporal_features


class DINOv2TemporalProcessor:
   
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        use_temporal: bool = True,
        num_frames: int = 4,
        num_heads: int = 6,
        num_layers: int = 4,
        freeze_dinov2: bool = True,
        seed: int = 0
    ):
        self.model_name = model_name
        self.use_temporal = use_temporal
        self.num_frames = num_frames
        
        # Load image processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Create model
        self.model = DINOv2TemporalProcessor(
            model_name=model_name,
            use_temporal=use_temporal,
            num_frames=num_frames,
            num_heads=num_heads,
            num_layers=num_layers,
            freeze_dinov2=freeze_dinov2
        )
        
        # Initialize parameters
        self.rng = jax_random.PRNGKey(seed)
        self._initialize_params()
        
        self._jit_forward = jax.jit(self._forward_fn, static_argnums=(2,))
        
        # Warm up
        dummy_input = jnp.ones((1, num_frames, 3, 224, 224))
        _ = self._jit_forward(self.params, dummy_input, False)
        
    def _initialize_params(self):
        """Initialize model parameters."""
        # Dummy input for initialization
        if self.use_temporal:
            dummy_input = jnp.ones((1, self.num_frames, 3, 224, 224))
        else:
            dummy_input = jnp.ones((1, 3, 224, 224))
        
        # Initialize
        self.rng, init_rng, dropout_rng = jax_random.split(self.rng, 3)
        self.params = self.model.init(
            {'params': init_rng, 'dropout': dropout_rng},
            dummy_input,
            training=False
        )
        
        print(f"Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(self.params))} parameters")
    
    def _forward_fn(self, params, x, training):
        """JIT-compiled forward function."""
        self.rng, dropout_rng = jax_random.split(self.rng)
        return self.model.apply(
            params,
            x,
            training=training,
            rngs={'dropout': dropout_rng}
        )
    
    def preprocess_images(self, images):
        # Use HuggingFace processor
        processed = self.processor(images=images, return_tensors="np")
        pixel_values = jnp.array(processed['pixel_values'])
        
        return pixel_values
    
    def encode(self, images, training=False):
        return self._jit_forward(self.params, images, training)
    
    def encode_sequence(self, image_sequence, training=False):
        pixel_values = self.preprocess_images(image_sequence)
        if pixel_values.ndim == 4:  # [frames, C, H, W]
            pixel_values = pixel_values[None, ...]  # [1, frames, C, H, W]
        return self.encode(pixel_values, training)
