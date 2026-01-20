# deepbgc/models/transformer.py
"""Stage-aware Transformer model for DeepBGC/PlantBGC.

This file is designed to be a drop-in replacement for the original transformer.py.

Key goals:
- Keep the same public API as KerasRNN (fit / predict) so existing post-processing works.
- Allow switching between three training stages via the 'stage' build_param:
  * stage1: token-level BGC detection (binary cross-entropy)
  * stage2: masked-embedding reconstruction (mean squared error) for domain adaptation
  * stage3: weak-supervision fine-tuning (binary cross-entropy)
- Preserve RNN-like callback/checkpoint behavior by reusing KerasRNN.fit for stage1/stage3.
"""

from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as K
from keras import optimizers
from keras.layers import Layer, Dense, Dropout, Input, TimeDistributed, Activation
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects

from .rnn import KerasRNN, precision, recall, auc_roc


class LayerNorm(Layer):
    """Minimal LayerNorm implementation (compatible with older Keras versions)."""

    def __init__(self, eps=1e-6, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma", shape=(dim,), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=(dim,), initializer="zeros", trainable=True)
        super(LayerNorm, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        var = K.mean(K.square(x - mean), axis=-1, keepdims=True)
        x_hat = (x - mean) / K.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class AddSinusoidalPositionalEncoding(Layer):
    """Adds sinusoidal positional encoding to (B, T, D) sequences."""

    def __init__(self, d_model, **kwargs):
        super(AddSinusoidalPositionalEncoding, self).__init__(**kwargs)
        if int(d_model) % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encoding.")
        self.d_model = int(d_model)

    def call(self, x):
        # x: (B, T, D)
        seq_len = tf.shape(x)[1]
        d_model = self.d_model

        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)   # (T, 1)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)          # (1, D)

        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(i / 2.0)) / float(d_model))
        angles = position * angle_rates                                    # (T, D)

        sines = tf.sin(angles[:, 0::2])
        cosines = tf.cos(angles[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)                # (T, D)
        pos_encoding = pos_encoding[tf.newaxis, :, :]                      # (1, T, D)
        return x + pos_encoding


class MultiHeadSelfAttention(Layer):
    """Multi-head self-attention (TF1-compatible, uses matmul + softmax)."""

    def __init__(self, d_model, num_heads, dropout=0.0, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.head_dim = self.d_model // self.num_heads
        self.dropout_rate = float(dropout)

        self.wq = Dense(self.d_model, use_bias=False, name="wq")
        self.wk = Dense(self.d_model, use_bias=False, name="wk")
        self.wv = Dense(self.d_model, use_bias=False, name="wv")
        self.wo = Dense(self.d_model, use_bias=False, name="wo")
        self.drop = Dropout(self.dropout_rate)

    def _split_heads(self, x):
        # (B, T, D) -> (B, H, T, Hd)
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        x = tf.reshape(x, [b, t, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, x, training=None):
        q = self._split_heads(self.wq(x))
        k = self._split_heads(self.wk(x))
        v = self._split_heads(self.wv(x))

        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.drop(weights, training=training)

        out = tf.matmul(weights, v)
        out = tf.transpose(out, [0, 2, 1, 3])  # (B, T, H, Hd)
        b = tf.shape(out)[0]
        t = tf.shape(out)[1]
        out = tf.reshape(out, [b, t, self.d_model])
        return self.wo(out)


def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout, name_prefix="enc"):
    """Standard Transformer encoder block: MHA + FFN with residual + LayerNorm."""
    attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout,
                                  name=name_prefix + "_mha")(x)
    attn = Dropout(dropout, name=name_prefix + "_attn_drop")(attn)
    x = LayerNorm(name=name_prefix + "_ln1")(x + attn)

    ff = Dense(ff_dim, activation="relu", name=name_prefix + "_ff1")(x)
    ff = Dropout(dropout, name=name_prefix + "_ff_drop1")(ff)
    ff = Dense(d_model, name=name_prefix + "_ff2")(ff)
    ff = Dropout(dropout, name=name_prefix + "_ff_drop2")(ff)
    x = LayerNorm(name=name_prefix + "_ln2")(x + ff)
    return x


def _pad_3d_sequences(seqs_2d, maxlen, dtype=np.float32):
    """Pad a list of (T, D) arrays to (N, maxlen, D)."""
    n = len(seqs_2d)
    d = seqs_2d[0].shape[1]
    out = np.zeros((n, maxlen, d), dtype=dtype)
    for i, a in enumerate(seqs_2d):
        L = min(a.shape[0], maxlen)
        if L > 0:
            out[i, :L, :] = a[:L, :]
    return out


class KerasTransformer(KerasRNN):
    """Stage-aware Transformer sequence model (API-compatible with KerasRNN).

    Parameters (build_params):
      - stage: "stage1" | "stage2" | "stage3"
      - freeze_backbone: if True, freeze all layers except the task head
      - mlm_eps / mlm_weight_from_diff: stage2 weighting knobs

    Notes:
      - For stage1/stage3, training uses the original KerasRNN.fit() generator/callback flow.
      - For stage2, we train a reconstruction head with temporal sample weights focusing on masked positions.
    """

    def __init__(self,
                 batch_size=1,
                 d_model=128,
                 num_heads=4,
                 ff_dim=256,
                 num_layers=2,
                 dropout=0.1,
                 stage="stage1",
                 freeze_backbone=False,
                 mlm_eps=1e-6,
                 mlm_weight_from_diff=True,
                 loss="binary_crossentropy",
                 activation="sigmoid"):
        super(KerasTransformer, self).__init__(batch_size=batch_size,
                                              hidden_size=int(d_model),
                                              loss=loss,
                                              stateful=False,
                                              activation=activation,
                                              return_sequences=True)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.stage = str(stage).lower()
        self.freeze_backbone = bool(freeze_backbone)

        self.mlm_eps = float(mlm_eps)
        self.mlm_weight_from_diff = bool(mlm_weight_from_diff)

        # Keep the same attribute name as KerasRNN for consistency.
        self.train_model = None

    def _apply_freeze(self, model):
        """Freeze backbone layers when switching stages, keep head layers trainable."""
        if not self.freeze_backbone:
            return
        for layer in model.layers:
            if layer.name.startswith("head_") or layer.name in ("logits", "prob", "head_recon"):
                layer.trainable = True
            else:
                layer.trainable = False

    def _make_optimizer(self, optimizer_name, learning_rate, decay):
        if optimizer_name is None:
            optimizer_name = "adam"

        optimizer_args = {}
        if learning_rate is not None:
            optimizer_args["lr"] = learning_rate
        if decay is not None:
            optimizer_args["decay"] = decay

        if optimizer_name == "adam":
            return optimizers.Adam(**optimizer_args)

        if optimizer_args:
            raise ValueError("Optimizer %s is not implemented for custom params yet" % optimizer_name)
        return optimizer_name

    def _build_backbone(self, inp):
        x = Dense(self.d_model, name="proj")(inp)
        x = AddSinusoidalPositionalEncoding(self.d_model, name="posenc")(x)
        x = Dropout(self.dropout, name="emb_drop")(x)

        for i in range(self.num_layers):
            x = transformer_encoder_block(
                x, self.d_model, self.num_heads, self.ff_dim, self.dropout,
                name_prefix="enc_%d" % i
            )
        return x

    def _build_model(self, input_size, stacked_sizes=None, fully_connected_sizes=None,
                     optimizer_name=None, learning_rate=None, decay=None, custom_batch_size=None):
        # Transformer ignores stacked_sizes; fully_connected_sizes is optional for a small task head MLP.
        if fully_connected_sizes is None:
            fully_connected_sizes = []

        inp = Input(shape=(None, int(input_size)),
                    batch_size=custom_batch_size or self.batch_size,
                    name="x")

        x = self._build_backbone(inp)

        # Optional MLP head layers (kept for compatibility with existing JSON configs).
        for j, sz in enumerate(fully_connected_sizes or []):
            x = TimeDistributed(Dense(int(sz), activation="relu"), name="head_fc_%d" % j)(x)

        optimizer = self._make_optimizer(optimizer_name, learning_rate, decay)
        logging.debug("Using optimizer %s", str(optimizer_name))

        if self.stage in ("stage1", "stage3"):
            logits = TimeDistributed(Dense(1), name="logits")(x)
            out = Activation(self.activation, name="prob")(logits)
            model = Model(inputs=inp, outputs=out)

            model.compile(
                loss=self.loss,
                optimizer=optimizer,
                sample_weight_mode="temporal",
                metrics=["accuracy", precision, recall, auc_roc],
            )
            self._apply_freeze(model)
            return model

        if self.stage == "stage2":
            # Reconstruct input embeddings (masked-embedding reconstruction).
            recon = TimeDistributed(Dense(int(input_size)), name="head_recon")(x)
            model = Model(inputs=inp, outputs=recon)

            model.compile(
                loss="mse",
                optimizer=optimizer,
                sample_weight_mode="temporal",
                metrics=["mse"],
            )
            self._apply_freeze(model)
            return model

        raise ValueError("Unknown stage: %s" % self.stage)

    def _build_stage2_generator(self, X_list, y_list, timesteps, shuffle):
        """Build a Stage2 generator that yields (x_batch, y_batch, temporal_weights)."""
        if not X_list:
            return None, None

        input_size = X_list[0].shape[1]
        seq_length = int(np.sum([len(X) for X in X_list]))
        num_batches = int(np.ceil(np.ceil(seq_length / self.batch_size) / timesteps))
        maxlen = num_batches * timesteps
        logging.info("Initializing Stage2 generator: %s batches from sequence length %s", num_batches, seq_length)

        X_arr = [X.values.astype(np.float32) for X in X_list]
        Y_arr = [Y.values.astype(np.float32) for Y in y_list]

        def generator():
            while True:
                idx = np.random.permutation(len(X_arr)) if shuffle else np.arange(len(X_arr))
                X_batches = np.array_split(np.array(X_arr, dtype=object)[idx], self.batch_size)
                Y_batches = np.array_split(np.array(Y_arr, dtype=object)[idx], self.batch_size)

                X_batches = [np.concatenate(b) if len(b) and b[0].size else np.empty((0, input_size), dtype=np.float32)
                             for b in X_batches]
                Y_batches = [np.concatenate(b) if len(b) and b[0].size else np.empty((0, input_size), dtype=np.float32)
                             for b in Y_batches]

                X_pad = _pad_3d_sequences(X_batches, maxlen=maxlen, dtype=np.float32)
                Y_pad = _pad_3d_sequences(Y_batches, maxlen=maxlen, dtype=np.float32)

                # Temporal weights: focus loss on masked positions only (where X differs from Y).
                if self.mlm_weight_from_diff:
                    diff = np.abs(X_pad - Y_pad)
                    w = (np.max(diff, axis=-1) > self.mlm_eps).astype(np.float32)  # (B, maxlen)
                else:
                    w = np.ones((self.batch_size, maxlen), dtype=np.float32)

                # (B, maxlen, D) -> (num_batches, B, timesteps, D)
                X_pad = np.swapaxes(X_pad.reshape(self.batch_size, num_batches, timesteps, input_size), 0, 1)
                Y_pad = np.swapaxes(Y_pad.reshape(self.batch_size, num_batches, timesteps, input_size), 0, 1)
                w = np.swapaxes(w.reshape(self.batch_size, num_batches, timesteps), 0, 1)

                for xb, yb, wb in zip(X_pad, Y_pad, w):
                    yield xb, yb, wb

        return generator, num_batches

    def fit(self, X_list, y_list, timesteps=256, validation_size=0.0, num_epochs=10, verbose=1,
            debug_progress_path=None, fully_connected_sizes=None, shuffle=True, stacked_sizes=None,
            early_stopping=None, positive_weight=None, weighted=False, optimizer=None,
            learning_rate=None, decay=None, validation_X_list=None, validation_y_list=None):
        """Stage-aware fit().

        - stage1/stage3: identical training flow to KerasRNN.fit()
        - stage2: requires X_list (masked embeddings) and y_list (original embeddings)
        """
        if self.stage in ("stage1", "stage3"):
            return super(KerasTransformer, self).fit(
                X_list=X_list,
                y_list=y_list,
                timesteps=timesteps,
                validation_size=validation_size,
                num_epochs=num_epochs,
                verbose=verbose,
                debug_progress_path=debug_progress_path,
                fully_connected_sizes=fully_connected_sizes,
                shuffle=shuffle,
                stacked_sizes=stacked_sizes,
                early_stopping=early_stopping,
                positive_weight=positive_weight,
                weighted=weighted,
                optimizer=optimizer,
                learning_rate=learning_rate,
                decay=decay,
                validation_X_list=validation_X_list,
                validation_y_list=validation_y_list,
            )

        if self.stage != "stage2":
            raise ValueError("fit() called with unsupported stage: %s" % self.stage)

        import keras

        if not isinstance(X_list, list) or not isinstance(y_list, list):
            raise AttributeError("Stage2 expects X_list and y_list to be lists of DataFrames.")
        if len(X_list) != len(y_list):
            raise ValueError("Stage2 expects X_list and y_list to have the same length.")

        input_size = X_list[0].shape[1]

        # Build train-time model (batch_size=self.batch_size) and inference model (batch_size=1).
        self.train_model = self._build_model(
            input_size=input_size,
            stacked_sizes=stacked_sizes,
            fully_connected_sizes=fully_connected_sizes,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            decay=decay,
            custom_batch_size=None,
        )
        self.model = self._build_model(
            input_size=input_size,
            stacked_sizes=stacked_sizes,
            fully_connected_sizes=fully_connected_sizes,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            decay=decay,
            custom_batch_size=1,
        )

        get_train_gen, train_num_batches = self._build_stage2_generator(
            X_list=X_list, y_list=y_list, timesteps=timesteps, shuffle=shuffle
        )
        train_gen = get_train_gen()

        validation_data = None
        validation_steps = None
        if validation_X_list is not None and validation_y_list is not None:
            get_val_gen, val_num_batches = self._build_stage2_generator(
                X_list=validation_X_list, y_list=validation_y_list, timesteps=timesteps, shuffle=shuffle
            )
            validation_data = get_val_gen()
            validation_steps = val_num_batches
        elif validation_size:
            logging.warning("Stage2 ignores validation_size; provide validation_X_list/validation_y_list if needed.")

        callbacks = []
        if debug_progress_path:
            callbacks.append(keras.callbacks.TensorBoard(
                log_dir=debug_progress_path,
                histogram_freq=0,
                batch_size=self.batch_size,
                write_graph=True,
                write_grads=False,
                write_images=False,
            ))

        if early_stopping:
            callbacks.append(keras.callbacks.EarlyStopping(
                min_delta=early_stopping.get("min_delta"),
                monitor=early_stopping.get("monitor", "val_loss"),
                patience=early_stopping.get("patience", 10),
                mode=early_stopping.get("mode", "min"),
                verbose=1,
            ))

        history = self.train_model.fit_generator(
            generator=train_gen,
            steps_per_epoch=train_num_batches,
            shuffle=False,
            epochs=num_epochs,
            validation_data=validation_data,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose,
        )

        # Copy weights to inference model so predict() works the same way as KerasRNN.
        self.model.set_weights(self.train_model.get_weights())
        return history

    def predict(self, X):
        """Predict token-level probabilities (stage1/stage3) or reconstruction error (stage2)."""
        if len(X.shape) != 2:
            raise AttributeError("Can only be called on a single 2-dimensional feature matrix")
        if self.model is None:
            raise AttributeError("Cannot predict using untrained model")

        batch_matrix = X.values.reshape(1, X.shape[0], X.shape[1])
        preds = self.model.predict(batch_matrix, batch_size=1)

        if self.stage in ("stage1", "stage3"):
            return pd.Series(preds[0, :, 0], X.index)

        # stage2: return per-timestep MSE (useful for sanity checks, not used for BGC calling).
        recon = preds[0, :, :]
        mse = np.mean((recon - X.values) ** 2, axis=-1)
        return pd.Series(mse, X.index)


# Register custom layers so model_from_json() (used in KerasRNN pickling) can deserialize Transformer models.
get_custom_objects().update({
    "LayerNorm": LayerNorm,
    "AddSinusoidalPositionalEncoding": AddSinusoidalPositionalEncoding,
    "MultiHeadSelfAttention": MultiHeadSelfAttention,
})
