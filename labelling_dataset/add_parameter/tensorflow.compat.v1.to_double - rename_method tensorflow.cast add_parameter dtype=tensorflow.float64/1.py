import numpy as np
import tensorflow
import tensorflow_probability as tfp
tf = tensorflow.compat.v1 
tf.disable_v2_behavior()
tfd = tfp.distributions
tfb = tfp.bijectors


class VAE:
    def __init__( self, x_dim, z_dim, encoder_layers, decoder_layers, dtype=np.float64, sess=tf.Session()):
        self.x_op = tf.placeholder( dtype=dtype, shape=[None, x_dim])

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.batch_mlp = self.__batch_mlp

        couple1 = tfb.RealNVP(num_masked=int(z_dim / 2),
                        shift_and_log_scale_fn=tfb.real_nvp_default_template(
                            hidden_layers=[z_dim, z_dim, z_dim]))
        remask1 = tfb.Permute(permutation=list(range(0, z_dim))[::-1])
        couple2 = tfb.RealNVP(num_masked=int(z_dim / 2),
                              shift_and_log_scale_fn=tfb.real_nvp_default_template(
                                  hidden_layers=[z_dim, z_dim, z_dim]))
        self.flow_bijector = tfb.Chain( [couple2, remask1, couple1])

        self.Loss, self.ELBO, self.pri_dist =\
            self.__graph_create( self.x_op, z_dim, encoder_layers, decoder_layers, dtype)

        self.optimizer = tf.train.AdamOptimizer( 0.0005).minimize( self.Loss)

        self.sess = sess
        self.saver = tf.train.Saver()

        self.sess.run( tf.global_variables_initializer())

    def __batch_mlp( self, input, output_sizes, variable_scope):
        output = input
        with tf.variable_scope( variable_scope, reuse=tf.AUTO_REUSE):
            for i, size in enumerate( output_sizes[:-1]):
                output = tf.nn.relu(
                    tf.layers.dense( output, size, name="layer_{}".format(i)))

            output = tf.layers.dense( output, output_sizes[-1], name="layer_{}".format(i + 1))

        return output

    def __graph_create( self, x, z_dim, encoder_layers, decoder_layers, dtype):
        with tf.variable_scope( 'vae_graph', reuse=tf.AUTO_REUSE):
            mean, sigma = tf.split(  self.__batch_mlp( x, encoder_layers, 'encoder'), num_or_size_splits=2, axis=-1)

            self.RealNVP = tfd.TransformedDistribution(
                       distribution = tfd.MultivariateNormalDiag(loc=mean, scale_diag=sigma),
                       bijector=self.flow_bijector)
                       
            pri_dist = tfd.MultivariateNormalDiag( 
                        loc=tf.constant( 0., shape=[z_dim], dtype=dtype), 
                        scale_diag=tf.constant( 1., shape=[z_dim], dtype=dtype))

            noise = tf.to_double( pri_dist.sample( sample_shape=tf.shape(x)[0]))
            z = mean + tf.multiply( sigma, noise)
            z = self.RealNVP.bijector.forward(z)
            y_mn, y_sd = tf.split(  self.__batch_mlp( z, decoder_layers, 'decoder'), num_or_size_splits=2, axis=-1)
            y_sd = tf.sigmoid( y_sd)

            posteriori = tfd.MultivariateNormalDiag( mean, sigma)
            KL_pri_posteriori = tfp.distributions.kl_divergence( posteriori, pri_dist)

            p_z_x = tfd.MultivariateNormalDiag( y_mn, y_sd)
            log_p_z_x = p_z_x.log_prob( x)

            ELBO = -KL_pri_posteriori + log_p_z_x - tf.reduce_mean( self.RealNVP.bijector.inverse_log_det_jacobian( z, event_ndims=1))
            Loss = -tf.reduce_mean( ELBO)

        return Loss, ELBO, pri_dist

    def train( self, x):
        self.sess.run( self.optimizer, feed_dict = { self.x_op : x })

    def get_ELBO( self, x):
        return self.sess.run( self.ELBO, feed_dict = { self.x_op : x })

    def KDE_log_prob( self, x, n_samples):
        z_samples = self.pri_dist.sample( n_samples)
        y_out = self.batch_mlp( z_samples, self.decoder_layers, variable_scope='vae_graph/decoder')
        mn, sd = tf.split( y_out, num_or_size_splits=2, axis=-1)
        sd = tf.sigmoid( sd)
        p_z_x = tfd.MultivariateNormalDiag( mn, sd)
        x = tf.expand_dims( x, axis=1)
        x = tf.tile( x, [1, n_samples, 1])
        log_prob_z_x = tf.reduce_mean( p_z_x.log_prob( x), axis=-1)

        return self.sess.run( log_prob_z_x)

    def save( self, path):
        self.saver.save( self.sess, path)

    def load( self, path):
        self.saver.restore( self.sess, path)

    


            


