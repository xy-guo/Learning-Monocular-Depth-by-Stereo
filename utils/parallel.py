import torch
from torch.nn.parallel import DataParallel
from torch.nn import Module
from torch.autograd import Variable
from torch.nn.parallel import gather


class DataParallelOnlyGatherFirst(DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelOnlyGatherFirst, self).__init__(module, device_ids=device_ids, output_device=output_device,
                                                          dim=dim)

    def gather(self, outputs, output_device):
        outputs = zip(*outputs)
        for idx, data in enumerate(outputs):
            if idx == 0:
                # gather the first return item
                outputs[idx] = gather(outputs[idx], output_device, dim=self.dim)
            else:
                # for the other items, return the items on GPU:0 (do not gather)
                outputs[idx] = outputs[idx][0]
        return outputs
