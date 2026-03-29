import neuralprocesses.torch as nps
import torch
from jetstream_interpolate_convcnp.utils.device import device

class ConstructModel:
    def __init__ (self, args):
        self.args = args
        self.device = device
        self.model = self.construct_model()
        
        # diagnostics
        print(f"Torch Version: {torch.__version__}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

    def construct_model(self):
        cnp = nps.construct_convgnp(
            dim_x=3,
            dim_yc=(2,),
            dim_yt=2,
            likelihood=self.args['likelihood'],
            conv_arch=self.args['conv_arch'],
            unet_channels=tuple(self.args['unet_channels']),
            unet_kernels=self.args['unet_kernels'],
            points_per_unit=self.args['points_per_unit'])
        
        return cnp
