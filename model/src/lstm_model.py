import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size=70,
        hidden_size=64,
        num_layers=1,
        num_classes=2,
        dropout=0.0
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)

        # Use the final timestep
        last_output = output[:, -1, :]

        return self.classifier(last_output)