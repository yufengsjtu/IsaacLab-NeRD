Fixed
^^^^^

* Fixed policy evaluation to restrict inference mode to policy execution so
  environment resets could update Transformer history safely.
* Fixed evaluation video recording to use Isaac Lab video recorders and Newton
  visualization. Recording stayed headless unless a visualizer was explicitly
  requested. A zero video interval recorded a single clip.
* Fixed importing evaluation helpers to avoid executing the evaluation CLI.
