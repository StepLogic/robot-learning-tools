import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 8e-4
    config.hidden_dims = (256, 256)
    config.cnn_features = (32, 64)
    config.cnn_padding = "VALID"
    config.latent_dim = 50
    config.encoder = "none"
    config.discount = 0.99
    config.tau = 0.005
    config.num_qs = 2
    config.init_temperature = 0.1
    config.target_entropy = None
    config.backup_entropy = True
    config.critic_reduction = "mean"
    return config
