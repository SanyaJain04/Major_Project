import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerCNNBiLSTM(nn.Module):
    def __init__(self, transformer_model, num_classes,
                 hidden_dim=256, cnn_filters=100,
                 kernel_sizes=[3,5,7], lstm_layers=2, dropout=0.3):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(transformer_model)
        hidden_size = self.transformer.config.hidden_size

        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, cnn_filters, k, padding=(k-1)//2)
            for k in kernel_sizes
        ])

        self.lstm = nn.LSTM(
            cnn_filters * len(kernel_sizes),
            hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids, attention_mask).last_hidden_state
        out = out.permute(0, 2, 1)

        conv_out = [torch.relu(conv(out)) for conv in self.convs]
        out = torch.cat(conv_out, dim=1)
        out = out.permute(0, 2, 1)

        _, (hidden, _) = self.lstm(out)
        h = torch.cat((hidden[-2], hidden[-1]), dim=1)

        h = self.bn(h)
        h = self.dropout(h)
        return self.fc(h)
