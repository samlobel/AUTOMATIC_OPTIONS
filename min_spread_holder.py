import numpy as np


class MinSpreadHolder(object):
    def __init__(self, dim, min_spread=0.05):
        # Should start out high... OR, should start out not existing...
        self.min_obs = None
        self.max_obs = None
        self.spread_obs = np.ones([dim], dtype=np.float32) * min_spread
        self.min_spread = min_spread

    def add_min(self, batch):
        min_batch = batch.min(axis=0)
        if self.min_obs is None:
            self.min_obs = min_batch
        else:
            self.min_obs = np.minimum(self.min_obs, min_batch)

    def add_spread(self, batch):
        max_batch = batch.max(axis=0)
        if self.max_obs is None:
          self.max_obs = max_batch
        else:
          self.spread_obs = np.maximum((self.max_obs - self.min_obs), self.min_spread)

    def add_batch(self, batch):
        self.add_min(batch)
        self.add_spread(batch)

    def get_min_spread(self):
        return self.min_obs, self.spread_obs

    def transform(self, obs_or_batch):
        transformed = (obs_or_batch - self.min_obs) / self.spread_obs
        print(transformed.min(), transformed.max())
        return (obs_or_batch - self.min_obs) / self.spread_obs
