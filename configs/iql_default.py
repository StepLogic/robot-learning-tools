import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Optimizer learning rates
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4          # IQL has a separate value (V) network

    # Network architecture
    config.hidden_dims = (256, 256)
    config.cnn_features = (32, 64)
    config.cnn_padding = "VALID"
    config.latent_dim = 50

    # Encoder: "none" passes embeddings straight through (PlaceholderEncoder),
    # matching your MobileNetFeatureWrapper pipeline.
    config.encoder = "none"

    # RL hyperparameters
    config.discount = 0.99
    config.tau = 0.005              # Soft target-critic update rate
    # config.num_qs = 10
    # IQL-specific hyperparameters
    config.expectile = 0.9          # How optimistic the value function is (0.5 = mean, 1.0 = max)
    config.A_scaling = 10.0         # Temperature scaling on advantage weights for actor update
    config.critic_reduction = "min" # "min" is more conservative; use "mean" if training is unstable

    return config