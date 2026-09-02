Added
^^^^^

* Added ``contact_filter="solver_active"`` to project only solver-included
  Newton native contact pairs from recorded separation.

Fixed
^^^^^

* Fixed Newton native-contact separation to avoid counting contact thickness
  twice when surface-point offsets are present. Regenerate native-contact
  datasets with schema version 3 before training.
