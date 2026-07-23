"""Direction-anchored variant of the core-guided two-stage GCNM.

This changes only the network output parameterization.  Dataset, supervision,
optimizer, losses, split policy, physics construction, and evaluation remain
the same as for :class:`CoreGuidedGCNMStage`.
"""

from __future__ import annotations

from gcnm_pvi.core_guided_model import CoreGuidedGCNMStage


class DirectionAnchoredCoreStage(CoreGuidedGCNMStage):
    """Predict a bounded correction around the current LM direction.

    Stage 1 returns ``d_LM + f_theta``.  Stage 2 returns
    ``sigma_current + d_LM + f_theta``.  Because the inherited output head is
    zero initialized, each stage starts exactly at its physics proposal.
    """

    def __init__(
        self,
        *,
        accumulate_current: bool,
        correction_limit: float,
    ) -> None:
        super().__init__(
            node_features=6,
            hidden=64,
            residual=False,
            correction_limit=correction_limit,
        )
        self.accumulate_current = bool(accumulate_current)

    def forward(self, data, *, return_aux: bool = False):
        output = super().forward(data, return_aux=return_aux)
        if return_aux:
            learned, core_logits, attention = output
            prediction = data.x[:, 1:2] + learned
            if self.accumulate_current:
                prediction = data.x[:, 0:1] + prediction
            return prediction, core_logits, attention
        prediction = data.x[:, 1:2] + output
        if self.accumulate_current:
            prediction = data.x[:, 0:1] + prediction
        return prediction
