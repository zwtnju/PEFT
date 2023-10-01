from pynvml import *
import logging

from transformers import HoulsbyInvConfig, PfeifferConfig, ParallelConfig, PrefixTuningConfig, LoRAConfig, IA3Config, \
    AdapterConfig

logger = logging.getLogger(__name__)


def show_gpu():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    total_memory = 0
    total_used = 0

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)

        total_memory += (info.total // 1048576)
        total_used += (info.used // 1048576)

    logger.info("name: [{}], num: [{}], total: [{} M], used: [{} M]."
                .format(nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(0)), device_count, total_memory, total_used))

    nvmlShutdown()


def get_model_size(model, required=True):
    if required:
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model_size = sum(p.numel() for p in model.parameters())
    return "{}M".format(round(model_size / 1e+6))


def getAdapter(adapter_type):
    if "houlsby" in adapter_type:
        adapter_config = HoulsbyInvConfig()
    elif "pfeiffer" in adapter_type:
        adapter_config = PfeifferConfig()
    elif "parallel" in adapter_type:
        adapter_config = ParallelConfig()
    elif "prefix_tuning" in adapter_type:
        adapter_config = PrefixTuningConfig()
    elif "lora" in adapter_type:
        adapter_config = LoRAConfig()
    elif "ia3" in adapter_type:
        adapter_config = IA3Config()
    else:
        adapter_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor={'default': 16},
                                       non_linearity='gelu')
    return adapter_config


ADAPTER_TYPE = ["houlsby", "pfeiffer", "parallel", "prefix_tuning", "lora", "ia3"]
