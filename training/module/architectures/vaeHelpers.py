import tensorflow as tf


def sampling(inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def vae_loss(z_log_var, z_mean, reco_loss):
    def loss(x, x_decoded_mean):
        reconstruction_loss = getattr(tf.keras.losses, reco_loss)(x, x_decoded_mean)
        kl_loss = - 0.5 * tf.keras.backend.mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        return reconstruction_loss + kl_loss
    
    return loss