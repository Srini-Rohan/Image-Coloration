import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class Pix2PixGAN():

    def __init__(self,input_size):
          self.input_size = input_size
          self.generator = self.Generator()
          self.discriminator = self.Discriminator()
          self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
          self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
          self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

          self.LAMBDA = 100

    def downsample(self,filters, size, batchnorm=True):

          model = tf.keras.Sequential()
          model.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer='he_normal', use_bias=False))

          if batchnorm:
            model.add(tf.keras.layers.BatchNormalization())

          model.add(tf.keras.layers.LeakyReLU())

          return model

    def upsample(self,filters, size, dropout=False):

          model = tf.keras.Sequential()
          model.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer='he_normal',use_bias=False))

          model.add(tf.keras.layers.BatchNormalization())

          if dropout:
              model.add(tf.keras.layers.Dropout(0.5))

          model.add(tf.keras.layers.ReLU())

          return model

    def Generator(self):

          inputs = tf.keras.layers.Input(shape=self.input_size)

          down_stack = [
            self.downsample(64, 4, batchnorm=False),
            self.downsample(128, 4),
            self.downsample(256, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
          ]

          up_stack = [
            self.upsample(512, 4, dropout=True),
            self.upsample(512, 4, dropout=True),
            self.upsample(512, 4, dropout=True),
            self.upsample(512, 4),
            self.upsample(256, 4),
            self.upsample(128, 4),
            self.upsample(64, 4),
          ]

          initializer = tf.random_normal_initializer(0., 0.02)
          last_layer = tf.keras.layers.Conv2DTranspose(3, 4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh')

          x = inputs

          skips = []
          for down in down_stack:
            x = down(x)
            skips.append(x)

          skips = reversed(skips[:-1])

          for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

          x = last_layer(x)

          return tf.keras.Model(inputs=inputs, outputs=x)

    def Discriminator(self):
          initializer = tf.random_normal_initializer(0., 0.02)

          inp = tf.keras.layers.Input(shape=self.input_size, name='input_image')
          tar = tf.keras.layers.Input(shape=self.input_size, name='target_image')

          x = tf.keras.layers.concatenate([inp, tar])

          down1 = self.downsample(64, 4, False)(x)
          down2 = self.downsample(128, 4)(down1)
          down3 = self.downsample(256, 4)(down2)

          zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
          conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1)

          batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

          leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

          zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

          last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                        kernel_initializer=initializer)(zero_pad2)

          return tf.keras.Model(inputs=[inp, tar], outputs=last)
    def generator_loss(self,disc_generated_output, gen_output, target):
          gan_loss = self.loss(tf.ones_like(disc_generated_output), disc_generated_output)

          l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

          total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

          return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self,disc_real_output, disc_generated_output):
          real_loss = self.loss(tf.ones_like(disc_real_output), disc_real_output)

          generated_loss = self.loss(tf.zeros_like(disc_generated_output), disc_generated_output)

          total_disc_loss = real_loss + generated_loss

          return total_disc_loss
    def train_step(self,input_image, target, epoch):
          with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = self.generator(input_image, training=True)

                disc_real_output = self.discriminator([input_image, target], training=True)
                disc_generated_output = self.discriminator([input_image, gen_output], training=True)

                gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
                disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

          generator_gradients = gen_tape.gradient(gen_total_loss,
                                                  self.generator.trainable_variables)
          discriminator_gradients = disc_tape.gradient(disc_loss,
                                                       self.discriminator.trainable_variables)

          self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                  self.generator.trainable_variables))
          self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                      self.discriminator.trainable_variables))
    def fit(self,train_ds, epochs):

          for epoch in range(epochs):
            
            print("Epoch: ", epoch+1)
            for n, (input_image, target) in train_ds.enumerate():
              
              self.train_step(input_image, target, epoch)

