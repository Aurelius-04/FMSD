import torch.nn as nn

from src.models.hmse import HeterogeneousMultimodalSensorEncoder
from src.models.sdpm import SensorToDecisionModule


class FMSDNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hmse = HeterogeneousMultimodalSensorEncoder(cfg)
        self.sdpm = SensorToDecisionModule(cfg)

    def forward(self, batch):
        representation = self.hmse(
            batch["environmental"],
            batch["soil"],
            batch["canopy"],
            batch["structured"],
        )
        prediction = self.sdpm(representation, batch["price"])
        prediction["representation"] = representation
        return prediction
