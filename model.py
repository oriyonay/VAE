import timm
import torch
import torch.nn as nn

class VAE(nn.Module):
    '''
    Produces 3x256x256 images (output dims = 2 ** (len(decoder_dims) + 1))
    beta: how much more important is KL divergence than reconstruction?
    emb_dim: latent space dimension
    enc_dim: shape of encoder output
    decoder_dims: channel dimensions of the decoder
    freeze_encoder: should the pre-trained encoder be frozen?
    '''
    def __init__(self, beta=1., emb_dim=128, enc_dim=512, 
                 decoder_dims=[512, 256, 256, 128, 128, 64, 64],
                 freeze_encoder=True):
        super().__init__()

        self.beta = beta
        self.emb_dim = emb_dim
        self.enc_dim = enc_dim
        self.decoder_dims = decoder_dims
        self.freeze_encoder = freeze_encoder

        self.encoder = timm.create_model(
            'resnetv2_50x1_bitm', 
            pretrained=True, 
            num_classes=enc_dim
        )

        if freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.mu = nn.Linear(enc_dim, emb_dim)
        self.logvar = nn.Linear(enc_dim, emb_dim)

        modules = []
        self.decoder_input = nn.Linear(emb_dim, decoder_dims[0]*4)

        for i in range(len(decoder_dims)-1):
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    decoder_dims[i],
                    decoder_dims[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0
                ),
                nn.BatchNorm2d(decoder_dims[i+1]),
                nn.ReLU()
            )
            modules.append(block)

        modules.append(nn.Sequential(
            nn.ConvTranspose2d(
                decoder_dims[-1],
                decoder_dims[-1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(decoder_dims[-1]),
            nn.ReLU(),
            nn.Conv2d(
                decoder_dims[-1], 
                out_channels=3,
                kernel_size=4, 
                padding=1
            ),
            nn.Sigmoid()
        ))

        self.decoder = nn.Sequential(*modules)
        
    def forward(self, x):
        assert x.shape[-1] == x.shape[-2] == self.output_dim

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)

        reconstruction_loss = nn.BCELoss(reduction='sum')(reconstructed, x) / x.shape[0]
        kld_loss = ((mu ** 2 + logvar.exp() - 1 - logvar) / 2).mean()
        loss = reconstruction_loss + (self.beta * kld_loss)

        return loss

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.decoder_dims[0], 2, 2)
        result = self.decoder(result)
        return result
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)
    
    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.emb_dim).to(device)
        return self.decode(z)
    
    @property
    def output_dim(self):
        return 2 ** (len(self.decoder_dims) + 1)
    
    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @staticmethod
    def load(filename):
        '''
        Loads model weights and configuration
        '''
        state_dict = torch.load(filename)
        model = VAE(**state_dict['args'])
        model.load_state_dict(state_dict['state_dict'])
        return model

    def save(self, filename):
        '''
        Saves model weights and configuration
        '''
        state_dict = {
            'args': {
                'beta': self.beta,
                'emb_dim': self.emb_dim,
                'enc_dim': self.enc_dim,
                'decoder_dims': self.decoder_dims
            },
            'state_dict': self.state_dict()
        }
        torch.save(state_dict, filename)