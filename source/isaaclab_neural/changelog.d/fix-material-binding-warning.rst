Fixed
^^^^^

* Fixed noisy USD ``BindingsAtPrim`` warnings during dataset generation,
  training, and evaluation by applying
  :class:`~pxr.UsdShade.MaterialBindingAPI` to robot visual mesh prims
  before Newton ingests the stage.
