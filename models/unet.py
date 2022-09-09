from typing import Tuple, Union

import torch
from torch import nn


class UNetBase(nn.Module):
    """Base class for U-Net that contains helper functions.
    """
    def __init__(self):
        super().__init__()

    def get_activation(self, activation: str) -> Union[nn.ReLU, nn.LeakyReLU, nn.ELU]:
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky':
            return nn.LeakyReLU(negative_slope=0.1)
        elif activation == 'elu':
            return nn.ELU()

    def get_normalization(
        self, normalization: str, num_channels: int
    ) -> Union[nn.BatchNorm2d, nn.InstanceNorm2d]:
        if normalization == 'batch':
            return nn.BatchNorm2d(num_channels)
        if normalization == 'instance':
            return nn.InstanceNorm2d(num_channels)

    def get_up_layer(
        self, in_channels: int, out_channels: int, kernel_size: int = 2,
        stride: int = 2, up_mode: str = 'transposed',
    ) -> Union[nn.ConvTranspose2d, nn.Upsample]:
        if up_mode == 'transposed':
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                      stride=stride)
        else:
            return nn.Upsample(scale_factor=2.0, mode=up_mode)

    def autocrop(self, encoder_layer: torch.Tensor, decoder_layer: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Center-crops the encoder_layer to the size of the decoder_layer,
        so that merging (concatenation) between levels/blocks is possible.
        This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
        """
        if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
            ds = encoder_layer.shape[2:]
            es = decoder_layer.shape[2:]
            assert ds[0] >= es[0]
            assert ds[1] >= es[1]
            if encoder_layer.dim() == 4:  # 2D
                encoder_layer = encoder_layer[
                    :, :, ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                    ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                ]
        return encoder_layer, decoder_layer


class DownBlock(UNetBase):
    """A down-sampling that performs 3 Convolutions and 1 MaxPool.
    Activation and normalization layera follow each convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, pooling: bool = True,
                 activation: str='relu', normalization: str=None, padding: int=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        self.padding = padding
        self.activation = activation

        # convolution layers
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1,
                               padding=self.padding, bias=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1,
                               padding=self.padding, bias=True)
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1,
                               padding=self.padding, bias=True)
        # pooling layers
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # non-linear activations
        self.act1 = self.get_activation(self.activation)
        self.act2 = self.get_activation(self.activation)
        self.act3 = self.get_activation(self.activation)
        # normalization layers
        self.norm1 = self.get_normalization(normalization=self.normalization,
                                            num_channels=self.out_channels)
        self.norm2 = self.get_normalization(normalization=self.normalization,
                                            num_channels=self.out_channels)
        self.norm3 = self.get_normalization(normalization=self.normalization,
                                            num_channels=self.out_channels)

    def forward(self, x: torch.tensor):
        y = self.conv1(x)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1

        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # activation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2

        y = self.conv3(y)  # convolution 2
        y = self.act3(y)  # activation 2
        if self.normalization:
            y = self.norm3(y)  # normalization 2

        before_pooling = y  # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)  # pooling

        return y, before_pooling


class UpBlock(UNetBase):
    """An up-sampling vlock that performs 2 Convolutions and 1 UpConvolution/Upsample.
    Activation and normalization layera follow each convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, activation: str='relu',
                 normalization: str=None, padding: int=1, up_mode: str = 'transposed'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.padding = padding
        self.activation = activation
        self.up_mode = up_mode

        # upconvolution/upsample layer
        self.up = self.get_up_layer(self.in_channels, self.out_channels, kernel_size=2,
                                     stride=2, up_mode=self.up_mode)
        # conv layers
        self.conv0 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1,
                               padding=0, bias=True)
        self.conv1 = nn.Conv2d(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1,
                               padding=self.padding, bias=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1,
                               padding=self.padding, bias=True)

        # activation layers
        self.act0 = self.get_activation(self.activation)
        self.act1 = self.get_activation(self.activation)
        self.act2 = self.get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm0 = self.get_normalization(normalization=self.normalization,
                                                num_channels=self.out_channels)
            self.norm1 = self.get_normalization(normalization=self.normalization,
                                                num_channels=self.out_channels)
            self.norm2 = self.get_normalization(normalization=self.normalization,
                                                num_channels=self.out_channels)

    def forward(self, encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
        """Forward pass
        Args:
            encoder_layer (torch.Tensor): Tensor from the encoder pathway
            decoder_layer (torch.Tensor): Tensor from the decoder pathway (to be up'd)
        """
        up_layer = self.up(decoder_layer)  # up-convolution/up-sampling
        cropped_encoder_layer, _ = self.autocrop(encoder_layer, up_layer)  # cropping

        if self.up_mode != 'transposed':
            # We need to reduce the channel dimension with a conv layer
            up_layer = self.conv0(up_layer)  # convolution 0
        up_layer = self.act0(up_layer)  # activation 0

        if self.normalization:
            up_layer = self.norm0(up_layer)  # normalization 0

        merged_layer = torch.cat((up_layer, cropped_encoder_layer), 1)  # concatenation
        y = self.conv1(merged_layer)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1

        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # acivation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2

        return y


class UNet(nn.Module):
    """U-Net for semantic segmentation.
    """
    def __init__(self, in_channels: int, out_channels: int, n_blocks: int=4,
                 start_filters: int=32, activation: str='relu', normalization: str='batch',
                 padding: int=1, up_mode: str='transposed', final_activation: str=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.padding = padding
        self.up_mode = up_mode

        assert final_activation in ['sigmoid', 'tanh', None]
        self.final_act = final_activation
        self.activations_dict = {'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # create encoder path
        for i in range(self.n_blocks):
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2 ** i)
            pooling = True if i < self.n_blocks - 1 else False

            down_block = DownBlock(in_channels=num_filters_in,
                                   out_channels=num_filters_out,
                                   pooling=pooling,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   padding=self.padding)

            self.down_blocks.append(down_block)

        # save the number of filters in the bridge
        self.down_num_filters_out = num_filters_out

        # create decoder path (requires only n_blocks-1 blocks)
        for i in range(n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            up_block = UpBlock(in_channels=num_filters_in,
                               out_channels=num_filters_out,
                               activation=self.activation,
                               normalization=self.normalization,
                               padding=self.padding,
                               up_mode=self.up_mode)

            self.up_blocks.append(up_block)

        # final convolution
        self.conv_final = nn.Conv2d(num_filters_out, self.out_channels, kernel_size=1, stride=1,
                                    padding=0, bias=True)

        # add the list of modules to current module
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

    def forward(self, x: torch.tensor):
        encoder_output = []

        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        if self.final_act is not None:
            x = self.activations_dict.get(self.final_act)(x)

        return x

    def __repr__(self):
        attributes = ({attr_key: self.__dict__[attr_key] for attr_key
            in self.__dict__.keys() if '_' not in attr_key[0] and 'training' not in attr_key})
        d = {self.__class__.__name__: attributes}
        return f'{d}'
