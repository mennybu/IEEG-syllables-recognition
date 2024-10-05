class TransformerClassifier(nn.Module):
    def init(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout):
        super(TransformerClassifier, self).init()
        self.encoder = nn.Linear(input_dim, model_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(model_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x
