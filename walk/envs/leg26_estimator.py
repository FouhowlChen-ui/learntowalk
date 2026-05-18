import numpy as np

from walk.envs.leg_base import MetabolicEnergyEstimator


class MetabolicEnergyEstimator26(MetabolicEnergyEstimator):
    MUSCLE_MASSES_26 = np.array(
        [
            0.300, 0.500, 0.263, 0.080, 0.040, 0.030, 0.546, 0.170, 0.193, 0.510, 0.280, 0.400, 0.110,
            0.300, 0.500, 0.263, 0.080, 0.040, 0.030, 0.546, 0.170, 0.193, 0.510, 0.280, 0.400, 0.110,
        ],
        dtype=np.float64,
    )

    def __init__(self, num_muscles: int = 26, alpha: float = 1.5, beta: float = 1.0):
        if num_muscles > self.MUSCLE_MASSES_26.shape[0]:
            raise ValueError(
                f"num_muscles={num_muscles} exceeds preset 26 length"
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.muscle_masses = self.MUSCLE_MASSES_26[:num_muscles].copy()
        self._mass_alpha = self.muscle_masses ** self.alpha
        self.reset()
