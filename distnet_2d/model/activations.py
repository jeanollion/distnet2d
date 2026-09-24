import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="distnet_2d")
class ScaledSoftsign:
    """c*softsign(x/c) = x / (1 + |x|/c): smooth, zero-centered, bounded to (-c, c)
    with unit slope at the origin. Polynomial (quadratic) tails -> far less gradient
    vanishing than tanh, and its zero-centering substitutes for a normalization layer
    while bounding the residual stream. NOTE: softsign(+/-inf)=nan (unlike tanh/relu6
    which map +/-inf -> finite), so this bounds the OUTPUT (preventing downstream
    overflow) but is NOT a hard inf-backstop for its own conv; safe in a fully-bounded
    stack where no op produces inf. Callable, so tf.keras.activations.get returns it
    unchanged."""
    def __init__(self, max_value=1.):
        self.max_value = float(max_value)
    def __call__(self, x):
        c = tf.cast(self.max_value, x.dtype)
        return c * tf.nn.softsign(x / c)
    def get_config(self):
        return {"max_value": self.max_value}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="distnet_2d")
class CappedReLU:
    """ReLU clamped to [0, max_value]:  min(relu(x), max_value). Bounds activation
    magnitude (low-precision stability / prevents fp16 overflow in deep residual
    streams) while keeping plain-ReLU behavior below max_value. Callable, so
    tf.keras.activations.get(instance) returns it unchanged."""
    def __init__(self, max_value=32):
        self.max_value = float(max_value)
    def __call__(self, x):
        return tf.keras.activations.relu(x, max_value=float(self.max_value))
    def get_config(self):
        return {"max_value": self.max_value}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="distnet_2d")
class SmoothCappedReLU:
    """Hard ReLU on the left, smooth saturation to c on the right = relu(x) - SmeLU(x-c).
    Left tail is EXACT ReLU (x<0 -> 0, kink at 0, keeps gating); the right tail is the only thing
    smoothed: identity & slope-1 over (0, c-b), a SmeLU shoulder, then saturates to c. Bounds the
    residual stream for fp16 while keeping plain-ReLU semantics; gradient is exactly 1 in-band
    (no attenuation until the cap), 0 above it. SmeLU per Shamir et al. 2020. Smoothing half-width
    b = beta*c (beta dimensionless) -> scale-homogeneous S(x)=c*S0(x/c). Clip-based, no
    exp/tanh/sqrt; custom gradient recomputes the gate from x -> stores only ONE tensor (x).
    grad = step(x) - clip((x-c+b)/2b, 0, 1). NOTE: finite inputs (incl. very large) map to [0,c];
    literal +/-inf -> nan, so use in a bounded stack. Callable -> tf.keras.activations.get returns
    it unchanged."""
    def __init__(self, max_value=32, beta=0.03):
        self.max_value = float(max_value)
        self.beta = float(beta)
    def __call__(self, x):
        c = tf.cast(self.max_value, x.dtype)
        b = tf.cast(self.beta, x.dtype) * c
        @tf.custom_gradient
        def f(x):
            z = x - c
            m = tf.clip_by_value((z + b) / (2. * b), 0., 1.)  # hard-sigmoid gate for the right shoulder
            out = tf.nn.relu(x) - (z * m + b * m * (1. - m))  # = relu(x) - SmeLU(x-c)
            def grad(dy):  # recompute the gate from x only -> backward stores a single tensor (x)
                g = tf.cast(x > 0., x.dtype) - tf.clip_by_value((x - c + b) / (2. * b), 0., 1.)
                return dy * g
            return out, grad
        return f(x)
    def get_config(self):
        return {"max_value": self.max_value, "beta": self.beta}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="distnet_2d")
class DoublySmoothCappedReLU:
    """Capped ReLU with the SAME SmeLU shoulder smoothing at BOTH ends (at 0 and at c):
    f(x) = SmeLU_b(x) - SmeLU_b(x - c). SmeLU (Shamir et al. 2020) is a quadratically smoothed
    ReLU of half-width b. Unlike SmoothCappedReLU -- which keeps an EXACT hard ReLU kink at 0
    and only smooths the upper cap -- this also rounds the lower kink: x<<0 -> 0 (smooth),
    slope-1 plateau over (b, c-b), smooth saturation to c for x>>c. Both shoulders are quadratic
    and symmetric in width; the gradient ramps 0->1 across [-b, b], is exactly 1 in-band, then
    ramps 1->0 across [c-b, c+b] -> no hard kink and no dead-unit gate anywhere, so even units
    sitting near 0 keep a non-zero gradient (vs hard ReLU's exactly-0 sub-threshold gradient).
    Smoothing half-width b = beta*c (dimensionless beta -> scale-homogeneous S(x)=c*S0(x/c)).
    Clip-based, no exp/tanh/sqrt; custom gradient recomputes both ramps from x -> stores only
    ONE tensor (x). NOTE finite inputs (incl. very large) map to [0, c]; literal +/-inf -> nan,
    so use in a bounded stack. Callable -> tf.keras.activations.get returns it unchanged."""
    def __init__(self, max_value=32, beta=0.03):
        self.max_value = float(max_value)
        self.beta = float(beta)
    def __call__(self, x):
        c = tf.cast(self.max_value, x.dtype)
        b = tf.cast(self.beta, x.dtype) * c
        @tf.custom_gradient
        def f(x):
            def smelu(z):  # SmeLU_b: 0 for z<=-b, (z+b)^2/(4b) for |z|<=b, z for z>=b
                m = tf.clip_by_value((z + b) / (2. * b), 0., 1.)
                return z * m + b * m * (1. - m)
            out = smelu(x) - smelu(x - c)  # smooth shoulder at 0 AND at c
            def grad(dy):  # SmeLU'(z) = clip((z+b)/2b, 0, 1); recompute both ramps from x only
                gl = tf.clip_by_value((x + b) / (2. * b), 0., 1.)
                gr = tf.clip_by_value((x - c + b) / (2. * b), 0., 1.)
                return dy * (gl - gr)
            return out, grad
        return f(x)
    def get_config(self):
        return {"max_value": self.max_value, "beta": self.beta}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def _parse_cap_beta(tail):
    """Parse a ``'<cap>'`` or ``'<cap>b<beta>'`` spec tail for the smooth-capped family.

    The optional ``b`` suffix encodes ``beta`` (the smoothing half-width factor, always
    < 1) written WITHOUT its decimal point: since beta starts with ``0``, the decimal
    point is re-inserted after the first digit. E.g. ``'32b003'`` -> cap=32, beta=0.03;
    ``'30b0015'`` -> cap=30, beta=0.015; ``'32'`` -> cap=32, beta=None (class default).
    Returns ``(cap, beta_or_None)``; raises ``ValueError`` on malformed input.
    """
    if "b" in tail:
        cap_s, beta_s = tail.split("b", 1)
        cap = float(cap_s)
        if beta_s == "":
            return cap, None
        beta = float(beta_s[0] + "." + beta_s[1:]) if len(beta_s) > 1 else float(beta_s)
        return cap, beta
    return float(tail), None

def get_activation(spec):
    """Resolve an activation spec to something usable as a layer `activation`.
      'reluN'     (e.g. 'relu6', 'relu30')     -> CappedReLU(N)       (hard ReLU capped at N)
      'screluN'   (e.g. 'screlu30')            -> SmoothCappedReLU(N) (smooth cap at N, hard ReLU at 0)
      'dscreluN'  (e.g. 'dscrelu30')           -> DoublySmoothCappedReLU(N) (smooth shoulder at BOTH 0 and N)
      'screluN'/'dscreluN' accept an optional 'b<beta>' suffix encoding beta without its
        decimal point (beta < 1, leading 0): e.g. 'dscrelu32b003' -> cap=32, beta=0.03;
        'screlu30b0015' -> cap=30, beta=0.015. Without the suffix the class default beta is used.
      'softsignN' (e.g. 'softsign30')          -> ScaledSoftsign(N)   (N*softsign(x/N), two-sided)
      'silu'/'swish', 'gelu', 'elu' -> the corresponding Keras built-in activation function.
      'relu', 'tanh', a callable, None, ... -> returned unchanged
        (tf.keras.activations.get handles them downstream).
    """
    if isinstance(spec, str):
        s = spec.lower()
        if s in ("silu", "swish", "gelu", "elu"):
            return tf.keras.activations.get(s)
        if s.startswith("dscrelu") and len(s) > 7:
            try:
                cap, beta = _parse_cap_beta(s[7:])
                return DoublySmoothCappedReLU(cap) if beta is None else DoublySmoothCappedReLU(cap, beta=beta)
            except ValueError:
                pass
        elif s.startswith("screlu") and len(s) > 6:
            try:
                cap, beta = _parse_cap_beta(s[6:])
                return SmoothCappedReLU(cap) if beta is None else SmoothCappedReLU(cap, beta=beta)
            except ValueError:
                pass
        elif s.startswith("relu") and len(s) > 4:
            try:
                return CappedReLU(float(s[4:]))
            except ValueError:
                pass
        elif s.startswith("softsign") and len(s) > 8:
            try:
                return ScaledSoftsign(float(s[8:]))
            except ValueError:
                pass
    return spec
