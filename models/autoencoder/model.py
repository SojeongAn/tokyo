
class AutoEncoder(nn.Module):
  """ Basic Autoencoder, parameterized weights. """
  def __init__(self, input_dim, hidden_dim1, hidden_dim2):
    super(AutoEncoder, self).__init__()
    
    self.encoder = nn.Sequential(
      nn.Linear(input_dim, hidden_dim1),
      nn.ReLU(),
      nn.Linear(hidden_dim1, hidden_dim2),
      nn.ReLU()
    )
    
    self.decoder = nn.Sequential(
      nn.Linear(hidden_dim2, hidden_dim1),
      nn.ReLU(),
      nn.Linear(hidden_dim1, input_dim),
      nn.ReLU()
    )
  
  def forward(self, x):
    out = x.view(x.size(0), -1)
    out = self.encoder(out)
    out = self.decoder(out)
    out = out.view(x.size())
    return out


class VAE(nn.Module):
    """Variational autoencoder, parameterized by a generative network."""

    def __init__(self, latent_size, data_size):
        super().__init__()
        self.register_buffer("p_z_loc", torch.zeros(latent_size))
        self.register_buffer("p_z_scale", torch.ones(latent_size))
        self.log_p_z = NormalLogProb()
        self.log_p_x = BernoulliLogProb()
        self.generative_network = NeuralNetwork(
            input_size=latent_size, output_size=data_size, hidden_size=latent_size * 2
        )

    def forward(self, z, x):
        """Return log probability of model."""
        log_p_z = self.log_p_z(self.p_z_loc, self.p_z_scale, z).sum(-1, keepdim=True)
        logits = self.generative_network(z)
        # unsqueeze sample dimension
        logits, x = torch.broadcast_tensors(logits, x.unsqueeze(1))
        log_p_x = self.log_p_x(logits, x).sum(-1, keepdim=True)
        return log_p_z + log_p_x
