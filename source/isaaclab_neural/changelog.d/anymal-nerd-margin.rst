Fixed
^^^^^

* Set the flat and rough ANYmal-C NeRD fallback collision margin to 0.0 m,
  matching stock Newton/MJWarp and dataset generation. Left other shape and
  solver settings unchanged. Revalidate existing dynamics and policy checkpoints
  under the aligned margin before comparing rollout results.
