from walk.envs.leg_imitation import MyoAssistLegImitation
from walk.envs.leg26_estimator import MetabolicEnergyEstimator26


class MyoAssistLeg26Imitation(MyoAssistLegImitation):
    def _setup(self, *, env_params, reference_data=None, **kwargs):
        super()._setup(
            env_params=env_params, reference_data=reference_data, **kwargs
        )

        num_muscles = int(self.sim.model.nu)
        if num_muscles != 26:
            print(
                f"[Leg26] warning: sim.model.nu={num_muscles} != 26, "
                f"MEE will be built with actual size"
            )

        mee_alpha = float(getattr(env_params, "mee_alpha", 1.5))
        mee_beta = float(getattr(env_params, "mee_beta", 1.0))
        self._mee_estimator = MetabolicEnergyEstimator26(
            num_muscles=num_muscles, alpha=mee_alpha, beta=mee_beta
        )

        assert self._mee_estimator.muscle_masses.shape[0] == num_muscles
