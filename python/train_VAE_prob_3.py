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
  plt.ion()

class Encoder(layers.Layer):
    def __init__(self,
                 input_shape=(28, 28, 1),
                 kernel_size=5,
                 activation='leaky_relu',
                 latent_dim=15,
                 base_depth=32,
                 name='encoderCNN',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self._latent_dim = latent_dim

        # 神经网络，用于提取特征并输出分布参数
        self.feature_extractor = Sequential(
            [
                layers.InputLayer(shape=input_shape),  # 修改：使用 shape 而不是 input_shape
                tfkl.Conv2D(base_depth, kernel_size, strides=1,
                            padding='same', activation=activation),
                tfkl.Conv2D(base_depth, kernel_size, strides=2,
                            padding='same', activation=activation),
                tfkl.Conv2D(2 * base_depth, kernel_size, strides=1,
                            padding='same', activation=activation),
                tfkl.Conv2D(2 * base_depth, kernel_size, strides=2,
                            padding='same', activation=activation),
                tfkl.Conv2D(4 * latent_dim, kernel_size + 2, strides=1, # 注意：原始代码使用了 kernel_size+2
                            padding='valid', activation=activation),
                tfkl.Flatten(),
                tfkl.Dense(2 * latent_dim),  # 输出潜变量分布的参数 (例如，均值和对数方差)
            ],
            name="feature_extractor_network"
        )

        # TFP 分布层，单独定义
        self.distribution_layer = tfpl.DistributionLambda(
            lambda t: tfd.MultivariateNormalDiag(
                loc=t[..., :self._latent_dim],
                scale_diag=tf.math.exp(t[..., self._latent_dim:])
            ),
            name="distribution_output_layer"
        )

    @property
    def latent_dim(self):
        return self._latent_dim

    def call(self, inputs):
        # 1. 通过特征提取网络获取分布的原始参数
        raw_params = self.feature_extractor(inputs)
        # 2. 将参数传递给TFP层以创建分布对象
        pz_x = self.distribution_layer(raw_params)
        return pz_x

class Decoder(layers.Layer):
    def __init__(self,
                 latent_dim=15,
                 target_shape=(28, 28, 1), # 这是伯努利分布的事件形状
                 kernel_size=5,
                 base_depth=32,
                 activation='leaky_relu',
                 name='decoderCNN',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        
        # 神经网络，用于从潜变量生成图像的logits
        self.logit_generator = Sequential(
            [
                layers.InputLayer(shape=(latent_dim,)),  # 修改：使用 shape
                tfkl.Reshape([1, 1, latent_dim]),
                tfkl.Conv2DTranspose(2 * base_depth, kernel_size + 2, strides=1,
                                     padding='valid', activation=activation),
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
                            padding='same', activation=None), # 输出原始logits，形状通常为 (H, W, 1)
                tfkl.Flatten(),  # 将logits展平，形状为 (H*W*1)
            ],
            name="logit_generator_network"
        )

        # TFP 伯努利分布层，单独定义
        # IndependentBernoulli 会根据 target_shape (event_shape) 来解释输入的 logits
        self.distribution_layer = tfpl.IndependentBernoulli(
            event_shape=target_shape
            # convert_to_tensor_fn=tfd.Bernoulli.logits # 可以明确指定输入是logits，但通常默认行为也适用
        )

    def call(self, inputs):
        # 1. 通过logits生成网络获取logits
        logits = self.logit_generator(inputs)
        # 2. 将logits传递给TFP层以创建伯努利分布对象
        px_z = self.distribution_layer(logits) # 在VAE中，解码器输出通常表示 P(X|z)
        return px_z


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
    loss_metric = tf.keras.metrics.Mean()

    for epoch in range(epochs):
        print('training epoch {}'.format(epoch+1))
        for i, x_batch in enumerate(train_dataset):
            enc_dec_loss = vae.train(x_batch,optimizer,L=5)
            loss = enc_dec_loss[1]
            loss_metric(loss) 
        print('loss {:.4f}'.format(loss_metric.result()))

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
