# Known Issues

| ID | Status | Issue | Evidence / note |
|---|---|---|---|
| `ROUGH-DATA-001` | Resolved | In 400-frame trajectories, the robot often only reached the rough region near the end; most samples remained on the center flat platform. | Dataset resets now sample `x/y` within `±1 m` and place the root above the highest scanned terrain point; a 400-step smoke increased non-random rough-footprint frames from 9.8% to 60.8% with no initial penetration. |
| `PD-GAIN-001` | Unresolved | The default-task zero-action dataset uses the CLI-default PD ranges (`Kp=[20, 80]`, `Kd=[0.5, 4]`) instead of the explicit ranges used by other default-task datasets (`Kp=[30, 200]`, `Kd=[0, 4]`). | The OSMO launcher omits explicit PD arguments for zero-action specs, while the dataset task still enables PD randomization. |
| `TERRAIN-LEVEL-001` | Resolved | Repeated rollout resets and terrain curriculum updates concentrated training trajectories at level 0. | Dataset-generation resets now sample levels uniformly while preserving each environment's terrain type. |
