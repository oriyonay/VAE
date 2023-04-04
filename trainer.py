'''
The main training loop, in one neat class
'''

import matplotlib.pyplot as plt
import os
from tqdm.auto import trange

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.losses = []

    def training_step(self, x):
        loss = self.model(x)
        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, dataloader, n_iters, checkpoint_every=None, 
              checkpoint_path=None, contains_labels=False):
        if checkpoint_path:
            os.makedirs(checkpoint_path, exist_ok=True)
            
        progress_bar = trange(n_iters)
        for i in progress_bar:
            if contains_labels:
                x = next(dataloader)[0].to(self.device)
            else:
                x = next(dataloader).to(self.device)
            self.training_step(x)
            progress_bar.set_description(f'loss: {self.losses[-1]:.3f}')

            if checkpoint_every and ((i+1) % checkpoint_every) == 0:
                filename = os.path.join(checkpoint_path, 'model.ckpt')
                self.model.save(filename)

    def plot_losses(self):
        plt.plot(range(len(self.losses)), self.losses)
        plt.show()