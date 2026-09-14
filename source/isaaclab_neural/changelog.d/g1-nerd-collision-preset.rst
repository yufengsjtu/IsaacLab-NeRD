Fixed
^^^^^

* Fixed flat and rough G1 NeRD environments to resolve the stock physics preset
  before copying collision and shape settings, avoiding the unintended
  ``margin=0.01`` fallback. Retained NeRD's single neural solver substep and
  disabled CUDA graphs. Existing dynamics checkpoints must be re-evaluated
  against the corrected contact configuration; policies trained with the old
  defaults must be revalidated and retrained if they relied on that behavior.
