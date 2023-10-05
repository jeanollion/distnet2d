import tensorflow as tf
# gradient accumulation code from : # from https://github.com/andreped/GradientAccumulator/blob/main/gradient_accumulator/accumulators.py
class GradientAccumulator():
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
        self.reinit_grad_accum()

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.model.optimizer.apply_gradients(zip(self.gradient_accumulation, self.model.trainable_variables))

        # reset
        self.accum_step_counter.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.model.trainable_variables[i], dtype=tf.float32), read_value=False)

    def reinit_grad_accum(self):
        # reinitialize gradient accumulator
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False,
            name="accum_" + str(i),
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            ) for i, v in enumerate(self.model.trainable_variables)
        ]

    def accumulate_gradients(self, gradients, scale=True):
        for i in range(len(self.gradient_accumulation)):
            if gradients[i] is not None:
                if scale:
                    gradients[i] = self.scale * gradients[i]
                self.gradient_accumulation[i].assign_add(gradients[i], read_value=False)

    def apply_gradients(self):
        # If accum_step_counter reach the accum_steps then we apply accumulated gradients to update the variables
        # otherwise do nothing
        tf.cond(tf.equal(self.accum_step_counter, self.accum_steps), true_fn=self.apply_accu_gradients,
                false_fn=lambda: None)

    def init_train_step(self):
        if self.first_call:
            self.reinit_grad_accum()
            self.first_call = False
        self.accum_step_counter.assign_add(1)
