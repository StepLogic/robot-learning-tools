import functools
from typing import Sequence
import warnings

import flax.linen as nn
import h5py
import jax
import jax.numpy as jnp

from jaxrl2.networks.constants import default_init
import numpy as np
import flaxmodels as fm
from flaxmodels.resnet.resnet import BasicBlock,LAYERS,URLS
from flaxmodels import utils
from flaxmodels.resnet.ops import BatchNorm
from flax.core.frozen_dict import freeze
def batch_norm(x, train, epsilon=1e-05, momentum=0.99, params=None, dtype='float32',name=None):
    # print(params)
    # breakpoint()
    if params is None:
        x = BatchNorm(epsilon=epsilon,
                      momentum=momentum,
                      use_running_average=not train,
                      name=name,
                      dtype=dtype)(x)
    else:
        params= {k:jnp.array(v) for k,v in params.items()}
        # breakpoint()
        x = BatchNorm(epsilon=epsilon,
                      momentum=momentum,
                      bias_init=lambda *_ : jnp.array(params['bias']),
                      scale_init=lambda *_ : jnp.array(params['scale']),
                      mean_init=lambda *_ : jnp.array(params['mean']),
                      var_init=lambda *_ : jnp.array(params['var']),
                      use_running_average=not train,
                      dtype=dtype,
                      name=name
                      )(x)
    return x

def preprocess_image_resnet18(img):
    # target_size=(224, 224)
    # resize_shape=np.array(img.shape)
    # resize_shape[0:2]=np.array(target_size)
    # # print(resize_shape)
    # img=jax.image.resize(img,shape=resize_shape, method="bilinear")
    
    # img = jnp.transpose(img,(2, 0, 1 ,3))
    # mean = jnp.array([0.485, 0.456, 0.406]).reshape(3,1,1,1)
    # std = jnp.array([0.229, 0.224, 0.225]).reshape(3,1,1,1)
    # img = (img - mean) / std
    # img = jnp.array(img)
    # # breakpoint()
    # print(img.shape)
    img = img.reshape(1, *img.shape)
    img = jnp.squeeze(img)
    return img

def batch_process_image_resnet18(img):
    return jax.vmap(preprocess_image_resnet18, (0))(img)

# class MLP(nn.Module):
#   def setup(self):
#     # Submodule names are derived by the attributes you assign to. In this
#     # case, "dense1" and "dense2". This follows the logic in PyTorch.
#     self.dense1 = nn.Dense(32)
#     self.dense2 = nn.Dense(32)

#   def __call__(self, x):
#     x = self.dense1(x)
#     x = nn.relu(x)
#     x = self.dense2(x)
#     return x
# URLS = {'resnet18': 'https://www.dropbox.com/s/wx3vt76s5gpdcw5/resnet18_weights.h5?dl=1',
#         'resnet34': 'https://www.dropbox.com/s/rnqn2x6trnztg4c/resnet34_weights.h5?dl=1',
#         'resnet50': 'https://www.dropbox.com/s/fcc8iii38ezvqog/resnet50_weights.h5?dl=1',
#         'resnet101': 'https://www.dropbox.com/s/hgtnk586pnz0xug/resnet101_weights.h5?dl=1',
#         'resnet152': 'https://www.dropbox.com/s/tvi28uwiy54mcfr/resnet152_weights.h5?dl=1'}

# LAYERS = {'resnet18': [2, 2, 2, 2],
#           'resnet34': [3, 4, 6, 3],
#           'resnet50': [3, 4, 6, 3],
#           'resnet101': [3, 4, 23, 3],
#           'resnet152': [3, 8, 36, 3]}
class PretrainedResNet(nn.Module):
    output: str='softmax'
    pretrained: str='imagenet'
    normalize: bool=True
    architecture: str='resnet18'
    num_classes: int=1000
    block: nn.Module=BasicBlock
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    ckpt_dir: str=None
    dtype: str='float32'

    def setup(self):
        self.param_dict = None
        if self.pretrained == 'imagenet':
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.architecture])
            self.param_dict = h5py.File(ckpt_file, 'r')
    
    @nn.compact
    def __call__(self, x,train=True):
        """
        Args:
            x (tensor): Input tensor of shape [N, H, W, 3]. Images must be in range [0, 1].
            train (bool): Training mode.

        Returns:
            (tensor): Out
            If output == 'logits' or output == 'softmax':
                (tensor): Output tensor of shape [N, num_classes].
            If output == 'activations':
                (dict): Dictionary of activations.
        """
        # train=False
        x=batch_process_image_resnet18(x)
        # observations = self.encoder(key, observations)
        # observations = encoder.apply(params, observations, train=True)
        # observations=encoder(observations)
        # jax.lax.stop_gradient(observations)
        param_dict=self.param_dict
        if self.normalize:
            # x=x/255
            mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1).astype(x.dtype)
            std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1).astype(x.dtype)
            x = (x - mean) / std

        if self.pretrained == 'imagenet':
            if self.num_classes != 1000:
                warnings.warn(f'The user specified parameter \'num_classes\' was set to {self.num_classes} '
                                'but will be overwritten with 1000 to match the specified pretrained checkpoint \'imagenet\', if ', UserWarning)

        act = {}

        x = nn.Conv(features=64, 
                    kernel_size=(7, 7),
                    kernel_init=self.kernel_init if param_dict is None else lambda *_ : jnp.array(param_dict['conv1']['weight']),
                    strides=(2, 2), 
                    padding=((3, 3), (3, 3)),
                    use_bias=False,
                    dtype=self.dtype)(x)
        act['conv1'] = x
        # print(train)
        x = batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if param_dict is None else param_dict['bn1'],
                        #    name="bn1",
                           dtype=self.dtype)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # Layer 1
        down = self.block.__name__ == 'Bottleneck'
        for i in range(LAYERS[self.architecture][0]):
            params = None if param_dict is None else param_dict['layer1'][f'block{i}']
            x = self.block(features=64,
                           kernel_size=(3, 3),
                           downsample=i == 0 and down,
                           stride=i != 0,
                           param_dict=params,
                           block_name=f'block1_{i}',
                           dtype=self.dtype)(x, act, train)
        
        # Layer 2
        for i in range(LAYERS[self.architecture][1]):
            params = None if param_dict is None else param_dict['layer2'][f'block{i}']
            x = self.block(features=128,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block2_{i}',
                           dtype=self.dtype)(x, act, train)
        # jax.lax.stop_gradient(x)
        # Layer 3
        for i in range(LAYERS[self.architecture][2]):
            params = None if param_dict is None else param_dict['layer3'][f'block{i}']
            x = self.block(features=256,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block3_{i}',
                           dtype=self.dtype)(x, act, train)
        # if not train:
        # x = jax.lax.stop_gradient(x)
        # Layer 4
        for i in range(LAYERS[self.architecture][3]):
            params = None if param_dict is None else param_dict['layer4'][f'block{i}']
            x = self.block(features=512,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block4_{i}',
                           dtype=self.dtype)(x, act, train)
        # breakpoint()
        x = jnp.mean(x, axis=(1, 2))
        x=jnp.squeeze(x)
        x = jax.lax.stop_gradient(x)
        return x

# class PretrainedResNet(nn.Module):
#     # def setup(self):
#     #    self.encoder=fm.ResNet18()
#     #    resnet18 = fm.ResNet18(output='logits', pretrained='imagenet')
#     #    params = resnet18.init(key, x)
#     # def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
#     #     key = jax.random.PRNGKey(0)
#     #     o_shape=observations.shape
#     #     observations=batch_process_image_resnet18(observations)
#     #     observations = self.encoder(key, observations)
#     #     # observations = encoder.apply(params, observations, train=True)
#     #     # observations=encoder(observations)
#     #     # jax.lax.stop_gradient(observations)
#     #     observations=jnp.squeeze(observations)
#     #     return observations
