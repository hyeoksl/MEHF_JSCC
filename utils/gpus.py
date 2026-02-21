from pynvml_utils import nvidia_smi

__all__ = ['getUsableGPUs']

def getUsableGPUs():
    nvsmi = nvidia_smi.getInstance()
    res_util_list = nvsmi.DeviceQuery("utilization.gpu, memory.used")["gpu"]
    gpu_id = None
    for i in [1, 2, 3, 0]:
        mem_used = res_util_list[i]['fb_memory_usage']['used']
        gpu_util = res_util_list[i]['utilization']['gpu_util']
        if mem_used < 1000 and gpu_util == 0:
            gpu_id = i
            break
    return gpu_id

