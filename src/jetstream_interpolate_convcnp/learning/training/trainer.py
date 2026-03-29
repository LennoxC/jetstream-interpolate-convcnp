from jetstream_interpolate_convcnp.learning.model.construct_model import ConstructModel
from jetstream_interpolate_convcnp.learning.tasks.sampler import Sampler
from jetstream_interpolate_convcnp.learning.tasks.tasks import TaskBuilder

class Trainer:
    def __init__(self, settings):
        self.settings = settings
        self.sampler = Sampler(settings)
        self.task_builder = TaskBuilder(settings)

        model_constructor = ConstructModel(settings['model'])
        self.model = model_constructor.model

    def train(self):
        #for step in range(self.sampler.train_size // self.settings['training']['batch_size']):
        for step in range(1): # just one step for now to test the training loop
            batch_idx = self.sampler.sample_readings(self.settings['training']['batch_size'], mode='train')
            tasks = self.task_builder.build_tasks(batch_idx)
            
        pass

    def validate(self):
        # validate model on validation set

        pass

    def diagnostics(self, epoch):
        # print diagnostics such as training loss, validation loss, etc.

        pass

    def run(self):
        for epoch in range(self.settings['training']['num_epochs']):
            self.train()
            self.validate()
            self.diagnostics(epoch)