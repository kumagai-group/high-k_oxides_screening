import warnings
import torch
import torch.nn as nn
from torch_geometric.nn import DataParallel
from model.model import E3NNModel
from model.params import E3NNHyperParams
from common.utils import to_gpu
from dotenv import load_dotenv
load_dotenv()

warnings.simplefilter("ignore")


def create_e3nn_model(
    e3nn_params: E3NNHyperParams, 
) -> nn.Module:
    model = E3NNModel(
        irreps_in=e3nn_params.irreps_in,
        irreps_sh=e3nn_params.irreps_sh,
        irreps_hid=e3nn_params.irreps_hid,
        irreps_out=e3nn_params.irreps_out,
        dim_edge=e3nn_params.dim_edge,
        n_conv=e3nn_params.n_conv,
        activation=e3nn_params.get_act_func(),
        is_site_pred=e3nn_params.is_site_pred,
        output_allsite=e3nn_params.output_allsite,
    )

    if torch.cuda.is_available():
        model = DataParallel(model, device_ids=e3nn_params.device_ids)
        model = to_gpu(model, c=e3nn_params.c)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num tranable params:", trainable_params)

    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()}")

    return model
