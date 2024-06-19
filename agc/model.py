import torch
from torch import nn, Tensor, FloatTensor, LongTensor
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F
import math
from einops import rearrange
import numpy as np

from transformers import PretrainedConfig, PreTrainedModel


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class VectorQuantize(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z: Tensor):
        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj.forward(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj.forward(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id: LongTensor):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: LongTensor):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: Tensor):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int | list = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z: Tensor, num_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors"""
        latent = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []

        if num_quantizers is None:
            num_quantizers = self.n_codebooks
        if self.training:
            num_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            num_quantizers[:n_dropout] = dropout[:n_dropout]
            num_quantizers = num_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= num_quantizers:
                break

            latent_i, commitment_loss_i, codebook_loss_i, indices_i, _ = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device)
                < num_quantizers
            )
            latent = latent + latent_i * mask[:, None, None]
            residual = residual - latent_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)

        z = torch.stack(codebook_indices, dim=1)
        # latent = latent.permute(0, 2, 1)

        return z, latent, commitment_loss, codebook_loss

    def from_indices(self, z: Tensor):
        z_q = 0.0
        z_p = []
        n_codebooks = z.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(z[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q


def vae_sample(mean: Tensor, scale: Tensor):
    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    kl = (mean * mean + var - logvar - 1).sum(1).mean()

    return latents, kl


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x: Tensor, alpha: Tensor):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


# Blocks


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: Tensor):
        return self.block.forward(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(2, d_model, kernel_size=7, padding=3)]  # made stereo

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x: Tensor):
        return self.block.forward(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=0 if stride % 2 == 0 else 1
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x: Tensor):
        return self.block.forward(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: list[int],
        d_out: int = 2,  # made stereo
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.model.forward(x)


class AGCConfig(PretrainedConfig):
    model_type = "agc"

    def __init__(
        self,
        continuous: bool = False,
        encoder_dim: int = 64,
        encoder_rates: list[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: list[int] = [8, 8, 4, 2],
        num_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: bool = False,
        vae: bool = False,
        sample_rate: int = 48000,
        **kwargs,
    ):
        self.continuous = continuous
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.vae = vae
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim
        if latent_dim is None:
            self.latent_dim = encoder_dim * (2 ** len(encoder_rates))

        super().__init__(**kwargs)


class AGC(PreTrainedModel):
    config_class = AGCConfig

    def __init__(
        self,
        config: AGCConfig,
    ):
        super().__init__(config)
        encoder_dim = config.encoder_dim
        encoder_rates = config.encoder_rates
        latent_dim = config.latent_dim
        decoder_dim = config.decoder_dim
        decoder_rates = config.decoder_rates
        num_codebooks = config.num_codebooks
        codebook_size = config.codebook_size
        codebook_dim = config.codebook_dim
        quantizer_dropout = config.quantizer_dropout
        sample_rate = config.sample_rate

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.continuous = config.continuous

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim, encoder_rates, latent_dim * 2 if config.vae else latent_dim
        )

        self.num_codebooks = num_codebooks

        if not self.continuous:
            self.quantizer = ResidualVectorQuantize(
                latent_dim,
                num_codebooks,
                codebook_size,
                codebook_dim,
                quantizer_dropout,
            )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.latent_std = 4.7

    def preprocess(self, audio: Tensor, sample_rate: int):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio = nn.functional.pad(audio, (0, right_pad))
        audio += 1e-5

        return audio

    def encode(self, audio: Tensor):
        """Encode given audio data and return quantized latent codes"""

        audio = self.preprocess(audio, self.sample_rate)
        raw_latent = self.encoder.forward(audio)

        if self.config.vae:
            mean, scale = raw_latent.chunk(2, dim=1)
            raw_latent = vae_sample(mean, scale)[0]

        if not self.continuous:
            z = self.quantizer.forward(raw_latent)[0]
        else:
            z = raw_latent
            z = z / self.latent_std

        return z

    def decode(self, z: FloatTensor | LongTensor) -> Tensor:
        """Decode quantized codes and return audio data"""

        if not self.continuous:
            latent = self.quantizer.from_indices(z)
        else:
            latent = z
            latent = latent * self.latent_std

        return self.decoder.forward(latent)