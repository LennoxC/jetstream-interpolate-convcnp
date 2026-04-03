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
            amdar_tasks, ecmwf_tasks = self.task_builder.build_tasks(batch_idx)
            
            print(f"Step {step}: AMDAR batch shape: {amdar_tasks.shape}, ECMWF batch shape: {ecmwf_tasks.shape}")
            # verify that non-zero samples are present in the batch
            print(f"Step {step}: AMDAR batch non-zero samples: {(amdar_tasks.abs().sum(dim=[0, 2, 3, 4]) > 0).sum()}, ECMWF batch non-zero samples: {(ecmwf_tasks.abs().sum(dim=[0, 2, 3, 4]) > 0).sum()}")
            # print the first few velocity values for the first sample in the batch
            print(f"Step {step}: AMDAR batch first sample velocity values: {amdar_tasks[0, :, 0, 0, 0]}, ECMWF batch first sample velocity values: {ecmwf_tasks[0, :, 0, 0, 0]}")

            # show in paraview
            if step == 0:
                from jetstream_interpolate_convcnp.plotting.vti import save_batch_to_vtk
                save_batch_to_vtk(amdar_tasks, filename_prefix="amdar_train")
                save_batch_to_vtk(ecmwf_tasks, filename_prefix="ecmwf_train")
        

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