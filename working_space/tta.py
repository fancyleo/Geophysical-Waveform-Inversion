"""Test-time augmentation (flip variants) for seismic -> velocity inference.

Why these are physical symmetries (unlike the plain receiver-only flip that this
repo tested and rejected):

  * The five sources sit at x grid cells [1, 18, 35, 53, 70] (10/180/350/530/700 m),
    which is a set *closed under reflection* about the array centre (cell 35.5).
  * Mirroring the whole acquisition therefore maps receivers j -> 71-j **and**
    relabels source i as source 6-i, i.e. the source *channel order* must be
    reversed as well. Reversing only the receiver axis (the old ``xflip``)
    leaves each source channel pointing at the mirrored position of a different
    source -> input/label mispairing -> false-negative experiment.
  * Time reversal is additionally motivated by the 15 Hz Ricker wavelet being
    time-symmetric.

  variant          dims flipped on (B, 5, 1000, 70)      inverse on (B, 70, 70)
  ---------------  -----------------------------------  ----------------------
  none             -                                     -
  src_recv         (-3, -1)  source + receiver          flip(-1)
  time_recv        (-2, -1)  time + receiver            flip(-1)
  time_src_recv    (-3,-2,-1) all three                 flip(-1)

Every variant is an involution whose effect on the velocity map is a mirror of
the receiver/width axis, so one prediction is averaged with its flipped-back
counterpart.
"""

import torch

# Dimension indices for a (B, n_src=5, n_steps=1000, n_recv=70) batch:
#   -3 = source channel, -2 = time, -1 = receiver.
FLIP_VARIANTS = {
    "none": None,
    "src_recv": (-3, -1),
    "time_recv": (-2, -1),
    "time_src_recv": (-3, -2, -1),
}


def model_predict(model, batch, tta="none"):
    """Single-model prediction, averaged over the identity and the flip view."""
    dims = FLIP_VARIANTS[tta]
    if dims is None:
        return model(batch)
    out = model(batch)
    flipped = model(torch.flip(batch, dims=dims))
    # Undo the augmentation on the prediction: mirror the width (receiver) axis.
    return 0.5 * (out + torch.flip(flipped, dims=(-1,)))


def ensemble_predict(batch, models, tta="none", scales=None):
    """Equal-weight average of every model's (optionally TTA'd) prediction.

    ``scales`` (optional) is a per-model list of ``(mean, std)`` denormalisation
    constants. Providing it averages in RAW velocity space, which is required
    when the members were trained under different normalisation conventions
    (see eval_holdout.py / infer.py). Without it the average happens in whatever
    normalized space the models output, which is only safe for homogeneous runs.
    """
    total = None
    for index, model in enumerate(models):
        out = model_predict(model, batch, tta).float()
        if scales is not None:
            mean, std = scales[index]
            out = out * std + mean
        total = out if total is None else total + out
    return total / len(models)
