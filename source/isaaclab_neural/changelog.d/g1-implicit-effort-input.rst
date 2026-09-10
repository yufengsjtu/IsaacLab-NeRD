Fixed
^^^^^

* Fixed G1 implicit-actuator dataset collection to record applied-effort telemetry
  in the ``joint_f`` input after analytic Newton steps, matching the Franka
  recording path. Preserved floating-base force slots and backend joint order.
* Fixed closed-loop NeRD to supply the same effort features after actuator
  telemetry, while keeping recorded forces unchanged during dataset evaluation
  and leaving purely explicit-actuator controls untouched.
* Required regeneration of G1 datasets recorded with zero ``joint_f`` and
  recomputation of dataset statistics. Retrain affected dynamics checkpoints;
  keep the existing mixed-data recipe and use a fresh dataset cache directory.
