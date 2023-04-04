import config
from model import VAE
from trainer import Trainer
import utils

dataloader = utils.get_data(
    config.datapath,
    config.transform,
    config.batch_size
)

model = VAE()
optimizer = config.optimizer(model.parameters(), lr=config.lr)

assert model.output_dim == config.img_size

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device=config.device
)

trainer.train(
    dataloader=dataloader,
    n_iters=config.n_iters,
    checkpoint_every=config.checkpoint_every,
    checkpoint_path=config.checkpoint_path
)