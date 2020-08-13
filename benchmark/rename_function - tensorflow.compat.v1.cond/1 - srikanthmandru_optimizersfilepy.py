

import tensorflow as tf

### Learning_rate decay optimizer
def generator_optimizer(generator_lr , step_change = 2000):
  gstep = tf.compat.v1.train.get_or_create_global_step()
  base_lr = generator_lr
  # Halve the learning rate at 1000 steps.
  lr = tf.compat.v1.cond(gstep < step_change, lambda: base_lr, lambda: base_lr / 2.0)
  return tf.compat.v1.train.AdamOptimizer(lr, 0.9)
  
 
def discriminator_optimizer(discriminator_lr, step_change = 2000):
  gstep = tf.compat.v1.train.get_or_create_global_step()
  base_lr = discriminator_lr
  # Halve the learning rate at 1000 steps.
  lr = tf.compat.v1.cond(gstep < step_change, lambda: base_lr, lambda: base_lr / 2.0)
  return tf.compat.v1.train.AdamOptimizer(lr, 0.9)

