import torch
from torch import nn

from models.unet import UpBlock, UNet


class InceptionedUpBlock(UpBlock):
    """Inceptioned version of the up block that takes into account
    additional filters added by the inception.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 activation: str = 'relu', normalization: str = None, padding: int = 1,
                 up_mode: str = 'transposed'):
        super().__init__(in_channels, out_channels, activation, normalization, padding, up_mode)

        self.skip_channels = skip_channels

        self.conv1 = nn.Conv2d(self.skip_channels + self.out_channels, self.out_channels,
                               kernel_size=3, stride=1, padding=self.padding, bias=True)


class InceptionedUNet(UNet):
    """Inceptioned U-Net for semantic segmentation.
    """
    def __init__(self, in_channels: int, out_channels: int, image_size: int, batch_size: int,
                 n_blocks: int=4, start_filters: int=32, activation: str='relu', normalization: str='batch',
                 padding: int = 1, incept_type: str='dilated_c', fc_size: str='full',
                 dilation_sizes: list=[1, 2, 4, 8, 16], up_mode: str='transposed', final_activation: str=None):
        super().__init__(in_channels, out_channels, n_blocks, start_filters,
                         activation, normalization, padding, up_mode, final_activation)

        assert incept_type in ['dilated_c', 'dilated_s', 'fc']
        assert fc_size in ['full', 'half']
        self.image_size = image_size
        self.batch_size = batch_size
        self.incept_type = incept_type
        self.fc_size = fc_size
        self.dilation_sizes = dilation_sizes

        self.incept_dict = {
            'dilated_s': self.dilated_stacked,
            'dilated_c': self.dilated_cascaded,
            'fc': self.fully_connected,
        }

        self.diluted_convs = nn.ModuleList([
            nn.Conv2d(self.down_num_filters_out, self.down_num_filters_out, kernel_size=3,
                      stride=1, padding=d, dilation=d, bias=True)
            for d in self.dilation_sizes
        ])

        self.fc_channels = None  # calculated in _find_padding() insede _create_linear_layers()
        self.fc = self._create_linear_layers()

        num_filters_out = self._create_up_blocks(n_blocks, incept_type)

        # update num of ouput filters in final convolution
        self.conv_final = nn.Conv2d(num_filters_out, self.out_channels, kernel_size=1, stride=1,
                                    padding=0, bias=True)


    def forward(self, x: torch.tensor):
        encoder_output = []

        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Inception with diluted convs
        incept_fn = self.incept_dict.get(self.incept_type)
        x = incept_fn(x)

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        if self.final_act is not None:
            x = self.activations_dict.get(self.final_act)(x)

        return x

    def dilated_stacked(self, x: torch.tensor) -> torch.tensor:
        incept_x = []
        for diluted_conv in self.diluted_convs:
            out = diluted_conv(x)
            incept_x.append(out)

        return torch.cat(incept_x, dim=1)

    def dilated_cascaded(self, x: torch.tensor) -> torch.tensor:
        incept_x = []
        out = x
        for diluted_conv in self.diluted_convs:
            out = diluted_conv(out)
            incept_x.append(out)

        return torch.cat(incept_x, dim=1)

    def fully_connected(self, x: torch.tensor) -> torch.tensor:
        input_shape = x.shape
        out = self.fc(x)
        out = torch.reshape(out, (input_shape[0], -1, input_shape[2], input_shape[3]))
        result = torch.cat((out, x), dim=1)

        return result

    def _calculate_bridge_input_shape(self):
        out_channels = self.down_blocks[-1].out_channels
        out_img_sizes = [self.image_size // (2**i) for i in range(1, self.n_blocks)]
        out_img_size = out_img_sizes[-1]

        return (self.batch_size, out_channels, out_img_size, out_img_size)

    def _find_padding(self, input_shape: tuple, fc_size: int) -> tuple:
        channel_size = input_shape[2] * input_shape[3]
        num_full_channels = fc_size // channel_size + 1
        self.fc_channels = num_full_channels
        padding = num_full_channels * channel_size - fc_size

        if padding % 2 == 0:
            return (padding // 2, padding // 2)
        return ((padding // 2, padding // 2 + 1))

    def _create_linear_layers(self):
        input_shape = self._calculate_bridge_input_shape()
        fc_size = input_shape[1] * (input_shape[2] // 9) * (input_shape[3] // 9)
        padding = self._find_padding(input_shape, fc_size)
        if self.fc_size == 'full':
            middle_fc_size = fc_size
        if self.fc_size == 'half':
            middle_fc_size = fc_size // 2

        return nn.Sequential(
            nn.AvgPool2d(3, 3), nn.AvgPool2d(3, 3), nn.Flatten(),
            nn.Linear(fc_size, middle_fc_size, device='cuda:0'), nn.ReLU(),
            nn.Linear(middle_fc_size, fc_size, device='cuda:0'), nn.ReLU(),
            nn.ConstantPad1d(padding, 0)
        )

    def _create_up_blocks(self, n_blocks: int, mode: str) -> int:
        self.up_blocks = nn.ModuleList([])

        # update decoder path (requires only n_blocks-1 blocks), to account for the increased
        # number of channels in the bridge
        if 'dilated' in mode:
            num_filters_out = self.down_num_filters_out * (len(self.dilation_sizes))
            divider = 4
        if 'fc' in mode:
            num_filters_out = self.down_num_filters_out + self.fc_channels
            divider = 2

        for i in range(n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // divider
            skip_channels = self.down_blocks[-(2 + i)].out_channels

            up_block = InceptionedUpBlock(in_channels=num_filters_in,
                                        skip_channels=skip_channels,
                                        out_channels=num_filters_out,
                                        activation=self.activation,
                                        normalization=self.normalization,
                                        padding=self.padding,
                                        up_mode=self.up_mode)

            self.up_blocks.append(up_block)

        return num_filters_out


class InceptionedUNetMultifocal(InceptionedUNet):

    def __init__(self, in_channels: int, out_channels: int, image_size: int, batch_size: int,
                 n_blocks: int = 4, start_filters: int = 32, activation: str = 'relu',
                 normalization: str = 'batch', padding: int = 1, incept_type: str = 'dilated_c',
                 fc_size: str = 'full', dilation_sizes: list = [1, 2, 4, 8, 16],
                 up_mode: str = 'transposed', final_activation: str = None):
        super().__init__(in_channels, out_channels, image_size, batch_size, n_blocks,
                         start_filters,activation, normalization, padding, incept_type,
                         fc_size, dilation_sizes, up_mode, final_activation)

        conv_final_in_channels = self.conv_final.in_channels
        self.conv_final = nn.Conv2d(conv_final_in_channels + self.out_channels, self.out_channels,
                                    kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.tensor, neighbour_pred: torch.tensor):
        # accepts two inputs, image (x) and prediction from
        # the neighbour focal plane (neighbour_pred)

        encoder_output = []
        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Inception with diluted convs
        incept_fn = self.incept_dict.get(self.incept_type)
        x = incept_fn(x)

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)

        x = torch.cat((x, neighbour_pred), dim=1)
        x = self.conv_final(x)
        if self.final_act is not None:
            x = self.activations_dict.get(self.final_act)(x)

        return x