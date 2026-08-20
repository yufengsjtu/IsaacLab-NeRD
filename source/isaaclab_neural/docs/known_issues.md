# Known Issues

| ID | Status | Issue | Evidence / note |
|---|---|---|---|
| `ROUGH-DATA-001` | Unresolved | In 400-frame trajectories, the robot often only reaches the rough region near the end; most samples remain on the center flat platform. | Observed in local dataset replay. |
| `PD-GAIN-001` | Unresolved | The default-task zero-action dataset uses the CLI-default PD ranges (`Kp=[20, 80]`, `Kd=[0.5, 4]`) instead of the explicit ranges used by other default-task datasets (`Kp=[30, 200]`, `Kd=[0, 4]`). | The OSMO launcher omits explicit PD arguments for zero-action specs, while the dataset task still enables PD randomization. |
| `TERRAIN-LEVEL-001` | Resolved | Repeated rollout resets and terrain curriculum updates concentrated training trajectories at level 0. | Dataset-generation resets now sample levels uniformly while preserving each environment's terrain type. |
