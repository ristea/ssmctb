import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, GlobalAveragePooling2D


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        # attention takes three inputs: queries, keys, and values,
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)

        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )

        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        # Transformer block multi-head Self Attention
        self.multiheadselfattention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        out1 = self.layernorm1(inputs)
        attention_output = self.multiheadselfattention(out1)
        out2 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out2)
        return self.layernorm2(out2 + ffn_output)


class ChannelWiseTransformerBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            num_patches,
            num_layers,  # depth
            num_heads,
            mlp_dim,
            d_model  # dim head
    ):
        super(ChannelWiseTransformerBlock, self).__init__()
        # create patches based on patch_size
        # image_size/patch_size==0
        self.avg_pool = GlobalAveragePooling2D()
        self.d_model = d_model
        self.patch_proj = self.create_postional_embedding(num_patches, d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim)
            for _ in range(num_layers)
        ]

    def create_postional_embedding(self, num_patches, d_model):
        self.pos_emb = self.add_weight("pos_emb", shape=(1, num_patches, d_model))
        return Dense(d_model)

    def call(self, input):
        x = self.avg_pool(input)
        x = tf.expand_dims(x, -1)

        x = self.patch_proj(x)

        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x)

        x = tf.reduce_mean(x, -1)
        x = tf.keras.activations.sigmoid(x)
        x = tf.expand_dims(x, 1)
        x = tf.expand_dims(x, 1)

        x = x * input
        return x


def masked_conv(input, kernel_dim, dilation, filters):
    '''
        input: The input data
        name: The name of the layer in the graph
        kernel_dim: The dimension of the sub-kernels ' k' ' from the paper
        dilation: The dilation dimension 'd' from the paper
        filters: The number of filter at the output (usually the same with the number of filter from the input)
        reduction_ratio: The reduction ratio for the SE block ('r' from the paper)
    '''
    pad = kernel_dim + dilation
    border_input = kernel_dim + 2 * dilation + 1

    sspcab_input = tf.pad(input, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), "REFLECT")

    sspcab_1 = tf.layers.conv2d(inputs=sspcab_input[:, :-border_input, :-border_input, :],
                                filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu)
    sspcab_3 = tf.layers.conv2d(inputs=sspcab_input[:, border_input:, :-border_input, :],
                                filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu)
    sspcab_7 = tf.layers.conv2d(inputs=sspcab_input[:, :-border_input, border_input:, :],
                                filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu)
    sspcab_9 = tf.layers.conv2d(inputs=sspcab_input[:, border_input:, border_input:, :],
                                filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu)
    sspcab_out = sspcab_1 + sspcab_3 + sspcab_7 + sspcab_9
    return sspcab_out


def SSMCTB(input, name, kernel_dim, dilation, filters, cw_transformer):
    with tf.variable_scope('decoder/SSMCTB' + name) as scope:
        sspcab = masked_conv(input, kernel_dim=kernel_dim, dilation=dilation, filters=filters)
        tr_out = cw_transformer(sspcab)
        return tr_out

# model = VisionTransformer(
#     num_patches=64,
#     patch_size=1,
#     num_layers=4,
#     d_model=128,
#     num_heads=4,
#     mlp_dim=128,
#     dropout=0.1,
# )

# img = tf.random.normal(shape=[1, 8, 8, 64])
# model(img)
#
# model.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     metrics=["accuracy"],
# )
