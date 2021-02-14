try:
    import tensorflow as tf
except ImportError:
    raise ImportError("Tensorflow is not installed, impossible to import `alig.tf.AliG`")


def minimize(optimizer, loss, global_step=None, var_list=None,
             gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP, aggregation_method=None,
             colocate_gradients_with_ops=False, name=None,
             grad_loss=None):
    """
    Re-write of tf.train.Optimizer.minimize
    """
    # first part of method is identical to tf
    grads_and_vars = optimizer.compute_gradients(
        loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
        raise ValueError(
            "No gradients provided for any variable, check your graph for ops"
            " that do not support gradients, between variables %s and loss %s." %
            ([str(v) for _, v in grads_and_vars], loss))

    # compute step-size here
    grad_sqrd_norm = sum(tf.norm(grad) ** 2 for grad, _ in grads_and_vars)
    optimizer._learning_rate = loss / (grad_sqrd_norm + optimizer.eps)
    if optimizer._max_lr is not None:
        optimizer._learning_rate = tf.clip_by_value(optimizer._learning_rate, clip_value_min=0,
                                                clip_value_max=optimizer._max_lr)

    return optimizer.apply_gradients(grads_and_vars, global_step=global_step,
                                name=name)


class AliGwithMomentum(tf.compat.v1.train.MomentumOptimizer):
    """Optimizer that implements the AliG algorithm.
    """

    def __init__(self, max_lr=None, momentum=0, use_locking=False, name="AliG", eps=1e-5):
        super(AliGwithMomentum, self).__init__(
            learning_rate=None, momentum=momentum, use_locking=use_locking,
            name=name, use_nesterov=True)
        self._max_lr = max_lr
        self.eps = eps

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):
        return minimize(self, loss, global_step=global_step, var_list=var_list,
                        gate_gradients=gate_gradients, aggregation_method=aggregation_method,
                        colocate_gradients_with_ops=colocate_gradients_with_ops, name=name,
                        grad_loss=grad_loss)


class AliGwithoutMomentum(tf.compat.v1.train.GradientDescentOptimizer):
    """Optimizer that implements the AliG algorithm.
    """

    def __init__(self, max_lr=None, use_locking=False, name="AliG", eps=1e-5):
        super(AliGwithoutMomentum, self).__init__(
            learning_rate=None, use_locking=use_locking, name=name)
        self._max_lr = max_lr
        self.eps = eps

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):
        return minimize(self, loss, global_step=global_step, var_list=var_list,
                        gate_gradients=gate_gradients, aggregation_method=aggregation_method,
                        colocate_gradients_with_ops=colocate_gradients_with_ops, name=name,
                        grad_loss=grad_loss)


def AliG(max_lr=None, momentum=0, use_locking=False, name="AliG", eps=1e-5):
    if momentum < 0:
        raise ValueError("Momentum cannot be negative ({})".format(momentum))
    elif momentum > 0:
        return AliGwithMomentum(max_lr=max_lr, momentum=momentum,
                                use_locking=use_locking, name=name, eps=eps)
    else:
        return AliGwithoutMomentum(max_lr=max_lr, use_locking=use_locking, name=name, eps=eps)
