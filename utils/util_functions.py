import os
import torch
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable

from datasets.utils import evaluate_images_abs as evaluate_images


def to_cuda_vars(vars_dict):
    new_dict = {}
    for k, v in vars_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.cuda()
    return new_dict


def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_iterative_func
def unsqueeze_dim0_tensor(data):
    if isinstance(data, torch.Tensor):
        return data.unsqueeze(0)
    else:
        return data


def save_scalars(logger, mode_tag, scalar_dict, global_step):
    for tag, values in scalar_dict.items():
        if isinstance(values, list):
            for i, value in enumerate(values):
                logger.add_scalar('{}/{}_{}'.format(mode_tag, tag, i), value.cpu().data.numpy(), global_step)
        else:
            logger.add_scalar('{}/{}'.format(mode_tag, tag), values.cpu().data.numpy(), global_step)


def save_images(logger, mode_tag, images_dict, global_step):
    for tag, values in images_dict.items():
        if not isinstance(values, list):
            values = [values]
        for i, value in enumerate(values):
            if isinstance(value, Variable):
                np_value = value.data.cpu().numpy()[0:1, ::-1, :, :].copy()  # pick only one images
            else:
                np_value = value.transpose([0, 3, 1, 2])[0:1]
            logger.add_image('{}/{}_{}'.format(mode_tag, tag, i),
                             vutils.make_grid(torch.from_numpy(np_value), padding=0, nrow=1, normalize=True,
                                              scale_each=True),
                             global_step)

def fliplr(tensor):
    inv_idx = Variable(torch.arange(tensor.size(3)-1, -1, -1).long()).cuda()
    # or equivalently torch.range(tensor.size(0)-1, 0, -1).long()
    inv_tensor = tensor.index_select(3, inv_idx)
    return inv_tensor.contiguous()


def save_kitti_metrics(logger, mode_tag, disps, filenames, global_step):
    eval_results = evaluate_images(disps, filenames)
    # outputs["disp_left_est"][0].cpu().data.numpy()[:, 0, :, :].copy(), test_files = sample["left_fn"][:len(sample["left_fn"]) // 2]
    for tag, value in eval_results.items():
        if isinstance(value, float) or (isinstance(value, np.float32)):
            logger.add_scalar('{}/metrics_{}'.format(mode_tag, tag), np.array(value, dtype=np.float32), global_step)
        else:
            print("{}/{}".format(tag, value))
            raise


def adjust_learning_rate(optimizer, epoch, base_lr, lrepochs):
    assert isinstance(lrepochs, str)
    splits = lrepochs.split(':')
    assert len(splits) == 1 or len(splits) == 2

    # parse downscale rate (after :), default downscale rate is 10
    downscale_rate = 2. if len(splits) == 1 else float(splits[1])
    # parse downscale epochs (before :) (when to down-scale the learning rate)
    downscale_epochs = [int(eid_str) for eid_str in splits[0].split(',')]
    print("downscale_epochs: {}".format(downscale_epochs))
    print("downscale_rate: {}".format(downscale_rate))

    lr = base_lr
    for eid in downscale_epochs:
        if epoch >= eid:
            lr /= downscale_rate
        else:
            break
    print("setting learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def find_latest_checkpoint(work_dir):
    # all_ckpt_filenames = [fn for fn in os.listdir(work_dir) if fn.startswith("model_params")]
    # all_steps = [int(fn.split(".")[-2].split('_')[-1]) for fn in all_ckpt_filenames]
    # latest_step = np.max(all_steps)
    # latest_model_param_fn = os.path.join(work_dir, 'model_params_%d.pkl' % latest_step)
    # latest_optimizer_param_fn = os.path.join(work_dir, 'optimizer_params_%d.pkl' % latest_step)
    # return latest_step, latest_model_param_fn, latest_optimizer_param_fn
    all_ckpt_filenames = [fn for fn in os.listdir(work_dir) if fn.startswith("checkpoint")]
    all_ckpt_filenames = sorted(all_ckpt_filenames, key=lambda x: int(x.split(".")[-2].split('_')[-1]))
    return os.path.join(work_dir, all_ckpt_filenames[-1])


def count_model_parameters(model):
    total_num_parameters = 0
    for param in model.parameters():
        total_num_parameters += np.array(param.data.shape).prod()
    return total_num_parameters


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


