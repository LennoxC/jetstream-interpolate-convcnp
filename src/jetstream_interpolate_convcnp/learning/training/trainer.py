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
        for step in range(1): # just one step for now to test the training loop
            print("Sampling batch...")
            batch_idx = self.sampler.sample_readings(self.settings['training']['batch_size'], mode='train')
            print("Building tasks...")
            amdar_tasks, ecmwf_tasks, metadata = self.task_builder.build_tasks(batch_idx)
            
            # show in paraview
            if step == 0 and self.settings['execute']['vtk_output']:
                from jetstream_interpolate_convcnp.plotting.vti import save_batch_to_vtk
                print(metadata)
                save_batch_to_vtk(amdar_tasks, filename_prefix=f"{self.settings['environment']['xtk_dir']}/amdar_train")
                save_batch_to_vtk(ecmwf_tasks, filename_prefix=f"{self.settings['environment']['xtk_dir']}/ecmwf_train")
        

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