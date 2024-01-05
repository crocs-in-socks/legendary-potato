import torch
import torch.nn as nn

from ModelArchitecture.SlimUNETR2D.Decoder import Decoder
from ModelArchitecture.SlimUNETR2D.Encoder import Encoder


class SlimUNETR2D(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        embed_dim=96,
        embedding_dim=16,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3,
    ):
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            embed_dim: deepest semantic channels
            embedding_dim: position code length
            channels: selection list of downsampling feature channel
            blocks: depth list of slim blocks
            heads: multiple set list of attention computations in parallel
            r: list of stride rate
            dropout: dropout rate
        Examples::
            # for 2D single channel input with size (128, 128), 3-channel output.
            >>> net = SlimUNETR(in_channels=4, out_channels=3, embedding_dim=64)

            # for 2D single channel input with size (96, 96), 2-channel output.
            >>> net = SlimUNETR(in_channels=1, out_channels=2, embedding_dim=27)

        """
        super(SlimUNETR2D, self).__init__()
        self.Encoder = Encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            embedding_dim=embedding_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            dropout=dropout,
        )
        self.Decoder = Decoder(
            out_channels=out_channels,
            embed_dim=embed_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            dropout=dropout,
        )

    def forward(self, x):
        embeding, hidden_states_out, (B, C, W, H) = self.Encoder(x)
        x = self.Decoder(embeding, hidden_states_out, (B, C, W, H))
        return torch.sigmoid(x)


if __name__ == "__main__":
    x = torch.randn(size=(1, 4, 128, 128))
    model = SlimUNETR2D(
        in_channels=4,
        out_channels=3,
        embed_dim=96,
        embedding_dim=64,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        distillation=False,
        dropout=0.3,
    )
    print(model(x).shape)
