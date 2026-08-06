Fixed
^^^^^

* Fixed flat-terrain rollout evaluation crashing in
  :meth:`~isaaclab_neural.eval.training_evaluator.TrainingRolloutEvaluator._restore_terrain_context`
  when datasets store sentinel ``terrain_level`` / ``terrain_type`` values but
  the active :class:`~isaaclab.terrains.terrain_importer.TerrainImporter` has no
  curriculum ``terrain_levels`` / ``terrain_types`` attributes.
