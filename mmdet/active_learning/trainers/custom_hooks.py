from copy import deepcopy

from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class ModelParametersRecorderHook(Hook):
    """Model Parameters Recorder Hook that records model parameters prior to
    training.

    This is to record the model parameters after the model possibly gets feed
    into a hook that registers additional parameters / buffers to the model,
    for example `BaseEMAHook`.
    """
    def before_run(self, runner):
        self.model_parameters = deepcopy(runner.model.state_dict())

    def get_model_parameters(self):
        if not hasattr(self, "model_parameters"):
            raise RuntimeError(
                "Model parameters have not been registered into the hook!"
            )
        return self.model_parameters
