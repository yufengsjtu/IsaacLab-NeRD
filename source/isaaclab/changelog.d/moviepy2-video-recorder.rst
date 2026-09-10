Fixed
^^^^^

* Fixed :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder` to import
  ``ImageSequenceClip`` from moviepy 2 (``from moviepy import ImageSequenceClip``)
  when ``moviepy.editor`` is missing. Moviepy 2 removed that module, so
  ``--video`` failed with ``pip install 'moviepy<2'`` even when moviepy was
  installed.
