from jetstream_interpolate_convcnp.learning.tasks.dataset import SampleSettings
import numpy as np

class Sampler:
    def __init__(self, settings):
        self.settings = settings
        self.global_sampler = SampleSettings(settings)

    def sample_readings(self, n, mode='train'):
        # sample n readings from the amdar dataset.
        # for efficiency, sample from the dates then sample from the readings for those dates.
        # weight the sampling by the number of samples for each date to ensure we get a representative sample of the data.

        # first sample n dates from the train_dates distribution
        date_dist = self.global_sampler.train_dates if mode == 'train' else self.global_sampler.test_dates

        if not date_dist:
            raise ValueError(f"No dates available for mode='{mode}'")

        dates = list(date_dist.keys())
        weights = np.array(list(date_dist.values()), dtype=float)
        probabilities = weights / weights.sum()

        sampled_date_idx = np.random.choice(len(dates), size=n, replace=True, p=probabilities)
        sampled_dates = [dates[idx] for idx in sampled_date_idx]
        
        date_sample = {date: date_dist[date] for date in sampled_dates}
        
        # now sample an index for each date based on the number of samples for that date
        samples_idx = [(date, np.random.choice(date_sample[date], size=1, replace=False)[0]) for date in sampled_dates]

        return samples_idx