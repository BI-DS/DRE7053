import os
# supress all tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf 
from VAE_prob import VAE
import numpy as np

import tensorflow as tf 
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import argparse
import os

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

def display_imgs(x, name, dset, y=None):
  if not isinstance(x, (np.ndarray, np.generic)):
    x = np.array(x)

  plt.ioff()
  n = x.shape[0]
  fig, axs = plt.subplots(1, n, figsize=(n, 1))
  if y is not None:
    fig.suptitle(np.argmax(y, axis=1))
  for i in range(n):
    axs.flat[i].imshow(x[i].squeeze(), interpolation='none', cmap='gray')
    axs.flat[i].axis('off')
  #plt.show()
  plt.savefig(os.path.join('../output',name+'_'+dset+'.png'))
  plt.close()
  #plt.ion()

class Encoder(layers.Layer):
    def __init__(self,
                 input_shape = (28,28,1),
                 kernel_size = 5,
                 activation  = 'leaky_relu',
                 latent_dim  = 15,
                 base_depth = 32,
                 name        = 'encoderCNN',
                 **kwargs):
        super(Encoder,self).__init__(name=name, **kwargs)
        self.hidden_layers = Sequential(
            [
            layers.InputLayer(input_shape=input_shape),
            tfkl.Conv2D(base_depth, kernel_size, strides=1,
                padding='same', activation=activation),
            tfkl.Conv2D(base_depth, kernel_size, strides=2,
                padding='same', activation=activation),
            tfkl.Conv2D(2 * base_depth, kernel_size, strides=1,
                padding='same', activation=activation),
            tfkl.Conv2D(2 * base_depth, kernel_size, strides=2,
                padding='same', activation=activation),
            tfkl.Conv2D(4 * latent_dim, kernel_size+2, strides=1,
                padding='valid', activation=activation),
            tfkl.Flatten(),
            tfkl.Dense(2 * latent_dim),
            tfpl.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(
                    loc=t[..., :latent_dim], scale_diag=tf.math.exp(t[..., latent_dim:]))),
            ]
            )
        self._latent_dim = latent_dim

    @property
    def latent_dim(self):
        return self._latent_dim

    def call(self, inputs):
        pz_x = self.hidden_layers(inputs)
    
        # output is a MVN pdf! :) 
        return pz_x

class Decoder(layers.Layer):
    def __init__(self,
                 latent_dim=15,
                 target_shape=(28,28,1),
                 kernel_size = 5,
                 base_depth = 32,
                 activation = 'leaky_relu',
                 name = 'decoderCNN',
                 **kwargs):
        super(Decoder,self).__init__(name=name, **kwargs)
        self.hidden_layers = Sequential(
                [
                layers.InputLayer(input_shape=(latent_dim,)),
                tfkl.Reshape([1, 1, latent_dim]),
                tfkl.Conv2DTranspose(2 * base_depth, kernel_size+2, strides=1,
                         padding='valid', activation=activation,name='decoder_layers'),
                tfkl.Conv2DTranspose(2 * base_depth, kernel_size, strides=1,
                         padding='same', activation=activation),
                tfkl.Conv2DTranspose(2 * base_depth, kernel_size, strides=2,
                         padding='same', activation=activation),
                tfkl.Conv2DTranspose(base_depth, kernel_size, strides=1,
                         padding='same', activation=activation),
                tfkl.Conv2DTranspose(base_depth, kernel_size, strides=2,
                         padding='same', activation=activation),
                tfkl.Conv2DTranspose(base_depth, kernel_size, strides=1,
                         padding='same', activation=activation),
                tfkl.Conv2D(filters=1, kernel_size=kernel_size, strides=1, 
                        padding='same', activation=None),
                tfkl.Flatten(),
                tfpl.IndependentBernoulli(target_shape)
                ]
                )
    
    def call(self, inputs):
        qz_x = self.hidden_layers(inputs)
        
        # output is a Bernoulli pdf! :) 
        return qz_x

def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", default= 15, help="Dimensionality of z", type=int)
    parser.add_argument("--epochs", default= 20, help="No of epochs", type=int)
    parser.add_argument("--dset", default='mnist', choices=['mnist','fashion_mnist'], help="dataset to be used")
    args = parser.parse_args()

    
    latent_dim = args.latent_dim
    vae = VAE(#tfd.Laplace,
              tfd.MultivariateNormalDiag, 
              Encoder(latent_dim = latent_dim),
              Decoder(latent_dim = latent_dim)
            )

    if args.dset == 'mnist':
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    elif args.dset == 'fashion_mnist':
        (train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()

    train_images = preprocess_images(train_images)
    test_images  = preprocess_images(test_images)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(500)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    epochs = args.epochs 
    elbo_metric = tf.keras.metrics.Mean()
    all_elbo=[]

    for epoch in range(epochs):
        print('training epoch {}'.format(epoch+1))
        for i, x_batch in enumerate(train_dataset):
            enc_dec_elbo = vae.train(x_batch,optimizer,L=5)
            elbo = enc_dec_elbo[1]
            elbo_metric(elbo) 
        all_elbo.append(elbo_metric.result())
        print('ELBO {:.4f}'.format(elbo_metric.result()))

    plt.plot(all_elbo)
    plt.xlabel('epoch')
    plt.ylabel('ELBO')
    plt.savefig('../output/elbo_{}.pdf'.format(args.dset))
    plt.close()

   # reconstruct 10 images from test
    x = test_images[:10]
    px_z = vae.reconstruct(x) 
    
    print('Original  Samples:')
    display_imgs(x,'original',args.dset)

    print('Randomly Generated Samples...')
    display_imgs(px_z.sample(),'qz_sample',args.dset)

    print('Randomly Generated Modes...')
    display_imgs(px_z.mode(),'qz_mode',args.dset)

    print('Randomly Generated Means...')
    display_imgs(px_z.mean(),'qz_mean',args.dset)
   
    # Now sample from prior and generate
    px_z = vae.generate() 
    
    print('Randomly Generated Samples...')
    display_imgs(px_z.sample(),'pz_sample',args.dset)

    print('Randomly Generated Modes...')
    display_imgs(px_z.mode(),'pz_mode',args.dset)

    print('Randomly Generated Means...')
    display_imgs(px_z.mean(),'pz_mean',args.dset)

if __name__ == "__main__":
    main()
