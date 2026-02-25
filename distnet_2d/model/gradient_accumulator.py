import tensorflow as tf

# adapted from : # from https://github.com/andreped/GradientAccumulator/blob/main/gradient_accumulator/accumulators.py
class GradientAccumulator:
    def __init__(self, accum_steps, model):
        self.model = model
        self.accum_steps = tf.constant(accum_steps, dtype=tf.int32, name="accum_steps")
        self.accum_step_counter = tf.Variable(0, dtype=tf.int32, trainable=False, name="accum_counter",
                                              synchronization=tf.VariableSynchronization.ON_READ,
                                              aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                              )
        self.first_call = True
        self.gradient_accumulation = None
        self.scale = tf.cast(1./accum_steps, tf.float32)

        # --- BN handling ---
        self._bn_layers = get_all_bn_layers(model)
        print(f"Accumulator: BN layers: {self._bn_layers.keys() }")
        self._bn_mean_accum = {}
        self._bn_ex2_accum = {}
        self._bn_moving_mean_snapshot = {}
        self._bn_moving_var_snapshot = {}
        self._init_bn_accumulators()

        self._reinit_grad_accum()

    # ------------------------------------------------------------------
    # Public API — only these three methods are called from training loop
    # ------------------------------------------------------------------

    def init_train_step(self):
        """Call at the very beginning of each training step."""
        if self.first_call:
            self._reinit_grad_accum()
            self.first_call = False
        self.accum_step_counter.assign_add(1)

        # Snapshot BN moving stats BEFORE the forward pass so we can
        # restore them after the forward pass runs (BN will corrupt them
        # with single-mini-batch statistics during training=True).
        self._snapshot_bn_stats()

    def post_forward_step(self):
        """
        Call AFTER forward pass, BEFORE tape.gradient().
        Recovers sub-batch stats, then restores moving averages to their
        pre-forward-pass values so they are not corrupted by sub-batch updates.
        """
        self._accumulate_bn_stats_from_ema()
        self._restore_bn_stats()

    def accumulate_gradients(self, gradients, scale=True):
        """Call after tape.gradient()."""
        for i in range(len(self.gradient_accumulation)):
            if gradients[i] is not None:
                if scale:
                    gradients[i] = self.scale * gradients[i]
                self.gradient_accumulation[i].assign_add(gradients[i], read_value=False)

    def apply_gradients(self):
        """Call at the end of each training step."""
        # If accum_step_counter reach the accum_steps then we apply accumulated gradients to update the variables
        # otherwise do nothing
        tf.cond(tf.equal(self.accum_step_counter, self.accum_steps), true_fn=self._apply_accu_gradients,
                false_fn=lambda: None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reinit_grad_accum(self):
        # reinitialize gradient accumulator
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False,
            name="accum_" + str(i),
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            ) for i, v in enumerate(self.model.trainable_variables)
        ]

    def _apply_accu_gradients(self):
        # apply accumulated gradients
        self.model.optimizer.apply_gradients(zip(self.gradient_accumulation, self.model.trainable_variables))

        # Apply the correctly accumulated BN moving statistics
        self._apply_bn_stats()

        # reset
        self.accum_step_counter.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.model.trainable_variables[i], dtype=tf.float32), read_value=False)

        self._reset_bn_accumulators()

    # --- BN helpers ---
    def _init_bn_accumulators(self):
        for layer_name, layer in self._bn_layers.items():
            self._bn_mean_accum[layer_name] = tf.Variable(
                tf.zeros_like(layer.moving_mean), trainable=False,
                name=f"bn_mean_accum_{layer_name}"
            )
            self._bn_ex2_accum[layer_name] = tf.Variable(
                tf.zeros_like(layer.moving_variance), trainable=False,
                name=f"bn_ex2_accum_{layer_name}"
            )
            self._bn_moving_mean_snapshot[layer_name] = tf.Variable(
                tf.zeros_like(layer.moving_mean), trainable=False,
                name=f"bn_mean_snap_{layer_name}"
            )
            self._bn_moving_var_snapshot[layer_name] = tf.Variable(
                tf.zeros_like(layer.moving_variance), trainable=False,
                name=f"bn_var_snap_{layer_name}"
            )

    def _snapshot_bn_stats(self):
        """Save current moving stats before the forward pass."""
        for layer_name, layer in self._bn_layers.items():
            self._bn_moving_mean_snapshot[layer_name].assign(layer.moving_mean)
            self._bn_moving_var_snapshot[layer_name].assign(layer.moving_variance)

    def _accumulate_bn_stats_from_ema(self):
        """
        Recover batch_mean and batch_var from the EMA update BN just applied:
            moving_new = moving_old * m + batch_stat * (1 - m)
            => batch_stat = (moving_new - moving_old * m) / (1 - m)

        Accumulate:
            - scaled mean:          mu_i / N
            - scaled E[X^2] term:  (sigma_i^2 + mu_i^2) / N
        """
        for layer_name, layer in self._bn_layers.items():
            m = layer.momentum
            snap_mean = self._bn_moving_mean_snapshot[layer_name]
            snap_var = self._bn_moving_var_snapshot[layer_name]

            batch_mean = (layer.moving_mean - snap_mean * m) / (1.0 - m)
            batch_var = (layer.moving_variance - snap_var * m) / (1.0 - m)

            self._bn_mean_accum[layer_name].assign_add(batch_mean * self.scale)
            # accumulate E[X^2] = var + mean^2
            self._bn_ex2_accum[layer_name].assign_add( (batch_var + tf.square(batch_mean)) * self.scale)

    def _restore_bn_stats(self):
        """Roll back the EMA update BN made during the forward pass."""
        for layer_name, layer in self._bn_layers.items():
            layer.moving_mean.assign(self._bn_moving_mean_snapshot[layer_name])
            layer.moving_variance.assign(self._bn_moving_var_snapshot[layer_name])

    def _apply_bn_stats(self):
        """
        Exact full-batch variance via law of total variance:
            E[X]      = mean(mu_i)
            E[X^2]    = mean(sigma_i^2 + mu_i^2)
            Var(full) = E[X^2] - E[X]^2
        """
        for layer_name, layer in self._bn_layers.items():
            m = layer.momentum

            full_mean = self._bn_mean_accum[layer_name]  # E[X]
            full_var = (self._bn_ex2_accum[layer_name] - tf.square(full_mean))  # E[X^2] - E[X]^2

            layer.moving_mean.assign( layer.moving_mean * m + full_mean * (1.0 - m))
            layer.moving_variance.assign( layer.moving_variance * m + full_var * (1.0 - m))

    def _reset_bn_accumulators(self):
        for layer_name, layer in self._bn_layers.items():
            self._bn_mean_accum[layer_name].assign(tf.zeros_like(layer.moving_mean))
            self._bn_ex2_accum[layer_name].assign(tf.zeros_like(layer.moving_variance))


def _get_sub_layer_dict(layer):
    if hasattr(layer, 'layers'):
        return {l.name: l for l in layer.layers}
    else:
        res = {}
        for attribute, value in vars(layer).items():
            if not attribute.startswith("_"):
                if isinstance(value, tf.keras.layers.Layer):
                    res[value.name] = value
                elif isinstance(value, (list, tuple)):
                    for l in value:
                        if isinstance(l, tf.keras.layers.Layer):
                            res[l.name] = l
        return res


def _flatten_model_layers(layer, layers = {}, prefix=""):
    sub_layers = _get_sub_layer_dict(layer)
    if len(sub_layers) == 0:
        layers[prefix+layer.name] = layer
    else:
        for l in sub_layers.values():
            _flatten_model_layers(l, layers, prefix =prefix + layer.name + "/" if not isinstance(layer, tf.keras.Model) else "")


def get_all_bn_layers(model):
    """Recursively collect all BatchNormalization layers in the model."""
    res = {}
    _flatten_model_layers(model, res)
    return {n:l for n, l in res.items() if isinstance(l, tf.keras.layers.BatchNormalization)}