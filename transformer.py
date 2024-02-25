import jax.numpy as jnp
from flax import linen as nn
import jax


def make_causal_mask(x):
    length = x.shape[1]
    mask = jnp.triu(
        jnp.ones((length, length), dtype=jnp.bool_), k=1
    )  # Create upper triangular matrix
    return mask


class TransformerNetwork(nn.Module):
    vocab_size: int
    block_size: int
    n_embed: int
    num_heads: int
    dff: int
    depth: int
    dropout_rate: float
    mask_encoder: bool

    @nn.compact
    def __call__(self, lq, hq, train):
        # we squeeze because there is a channel in the audio and we want to turn
        # the whole thing into tokens
        lq, hq = jnp.squeeze(lq, axis=-1), jnp.squeeze(hq, axis=-1)
        # Embedding layers for lq (low quality - encoder input)
        token_embedding_encoder = nn.Embed(
            num_embeddings=self.vocab_size, features=self.n_embed
        )
        position_embedding_encoder = nn.Embed(
            num_embeddings=self.block_size, features=self.n_embed
        )

        # Token and position embeddings for lq
        seq_len_encoder = lq.shape[1]
        positions_encoder = jnp.arange(seq_len_encoder)[None, :]
        encoder_input = token_embedding_encoder(lq) + position_embedding_encoder(
            positions_encoder
        )
        encoder_mask = make_causal_mask(encoder_input) if self.mask_encoder else None

        # Dropout for encoder embeddings
        encoder_input = nn.Dropout(rate=self.dropout_rate)(
            encoder_input, deterministic=not train
        )

        # Encoder
        for _ in range(self.depth):
            encoder_input = EncoderLayer(
                d_model=self.n_embed,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate,
            )(encoder_input, encoder_mask, train)
        encoder_output = encoder_input
        # Embedding layers for hq (high quality - decoder input)
        token_embedding_decoder = nn.Embed(
            num_embeddings=self.vocab_size, features=self.n_embed
        )
        position_embedding_decoder = nn.Embed(
            num_embeddings=self.block_size, features=self.n_embed
        )

        # Token and position embeddings for hq
        seq_len_decoder = hq.shape[1]
        positions_decoder = jnp.arange(seq_len_decoder)[None, :]
        decoder_input = token_embedding_decoder(hq) + position_embedding_decoder(
            positions_decoder
        )
        decoder_mask = make_causal_mask(decoder_input)

        # Dropout for decoder embeddings
        decoder_input = nn.Dropout(rate=self.dropout_rate)(
            decoder_input, deterministic=not train
        )

        # Decoder
        for _ in range(self.depth):
            decoder_input = DecoderLayer(
                d_model=self.n_embed,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate,
            )(decoder_input, encoder_output, decoder_mask, train)

        decoder_input = nn.LayerNorm()(decoder_input)
        # could theoretically only do this on the last timestep during inference
        output = nn.Dense(features=self.vocab_size, use_bias=False)(decoder_input)

        return output


class EncoderLayer(nn.Module):
    d_model: int
    num_heads: int
    dff: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, encoder_mask, train):
        ln1 = nn.LayerNorm()(x)
        attn_output = nn.SelfAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(ln1, mask=encoder_mask, deterministic=not train)
        attn_output = nn.Dropout(rate=self.dropout_rate)(
            attn_output, deterministic=not train
        )
        x = x + attn_output

        ln2 = nn.LayerNorm()(x)
        ffn_output = PositionwiseFeedForward(
            d_model=self.d_model, dff=self.dff, dropout_rate=self.dropout_rate
        )(ln2, train)
        x = x + ffn_output

        return x


class DecoderLayer(nn.Module):
    d_model: int
    num_heads: int
    dff: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, encoder_output, decoder_mask, train):
        ln1 = nn.LayerNorm()(x)
        attn1_output = nn.SelfAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(ln1, mask=decoder_mask, deterministic=not train)
        attn1_output = nn.Dropout(rate=self.dropout_rate)(
            attn1_output, deterministic=not train
        )
        x = x + attn1_output

        ln2 = nn.LayerNorm()(x)
        attn2_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(ln2, encoder_output, deterministic=not train)
        attn2_output = nn.Dropout(rate=self.dropout_rate)(
            attn2_output, deterministic=not train
        )
        x = x + attn2_output

        ln3 = nn.LayerNorm()(x)
        ffn_output = PositionwiseFeedForward(
            d_model=self.d_model, dff=self.dff, dropout_rate=self.dropout_rate
        )(ln3, train)
        x = x + ffn_output

        return x


class PositionwiseFeedForward(nn.Module):
    d_model: int
    dff: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, train):
        dense1 = nn.Dense(self.dff)(x)
        gelu = nn.gelu(dense1)
        dropout = nn.Dropout(rate=self.dropout_rate)(gelu, deterministic=not train)
        dense2 = nn.Dense(self.d_model)(dropout)
        return dense2


if __name__ == "__main__":
    model = TransformerNetwork(
        vocab_size=2**16,
        block_size=2**11,
        n_embed=2**10,
        num_heads=2**5,
        dff=2**11,
        depth=2**2,
        dropout_rate=0.2,
        mask_encoder=True
    )
    print(
        model.tabulate(
            jax.random.key(0),
            jnp.ones((2**2, 2**11, 1), dtype=jnp.int32),
            jnp.ones((2**2, 2**11, 1), dtype=jnp.int32),
            train=False,
        )
    )
