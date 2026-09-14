Changed
^^^^^^^

* Simplified ordinary Franka Reach to the historical Legacy asset and drive
  recipe, removing the temporary response-experiment configuration layer and
  its dedicated tests. The standard controller and physics presets remained
  available. Users should omit old asset/drive selectors when training Legacy
  policies and should not load Menagerie checkpoints into the Legacy recipe.
