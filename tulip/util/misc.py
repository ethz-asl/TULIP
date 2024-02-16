# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch._six import inf

import itertools


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def initialize_decoder_weights(pretrain_model):

    for k in list(pretrain_model.keys()):
        if k.__contains__('layers.0'):
            new_key = k.replace('layers.0', 'layers_up.2')
            new_key = new_key.replace('downsample', 'upsample') if new_key.__contains__('downsample') else new_key
            pretrain_model[k] = pretrain_model[new_key]
            del pretrain_model[new_key]
        if k.__contains__('layers.1'):
            new_key = k.replace('layers.1', 'layers_up.1')
            new_key = new_key.replace('downsample', 'upsample') if new_key.__contains__('downsample') else new_key
            pretrain_model[k] = pretrain_model[new_key]
            del pretrain_model[new_key]

        if k.__contains__('layers.2'):
            new_key = k.replace('layers.2', 'layers_up.0')
            new_key = new_key.replace('downsample', 'upsample') if new_key.__contains__('downsample') else new_key
            pretrain_model[k] = pretrain_model[new_key]
            del pretrain_model[new_key]

    for k in list(pretrain_model.keys()):
        if k.__contains__('head') or \
            k.__contains__('decoder_pred') or \
            k.__contains__('skip_connection') or \
            k.__contains__('first_patch_expanding') or \
            k.__contains__('output_weights') or \
            k.__contains__('up'):
            print(f"Removing key {k} from pretrained checkpoint")
            del pretrain_model[k]

    print(pretrain_model.keys())
    return pretrain_model



def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):       
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def check_match(a, b):
    if type(a) == int and type(b) == int:
        return a == b
    elif type(a) == int or type(b) == int:
        return False
    else:
        return a.shape == b.shape


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # have to change some name in the pretrain weights, can be removed in the further experiments
        model_checkpoint = checkpoint['model']
        for k in list(model_checkpoint.keys()):
            if k == 'head.weight':
                model_checkpoint['decoder_pred.weight'] = model_checkpoint['head.weight']
                del model_checkpoint['head.weight']
            elif k == 'pixel_shuffle_layer.conv_expand.0.weight':
                model_checkpoint['ps_head.conv_expand.0.weight'] = model_checkpoint['pixel_shuffle_layer.conv_expand.0.weight']
                del model_checkpoint['pixel_shuffle_layer.conv_expand.0.weight']
            elif k == 'pixel_shuffle_layer.conv_expand.0.bias':
                model_checkpoint['ps_head.conv_expand.0.bias'] = model_checkpoint['pixel_shuffle_layer.conv_expand.0.bias']
                del model_checkpoint['pixel_shuffle_layer.conv_expand.0.bias']

        
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval) and not (hasattr(args, 'analyze') and args.analyze) :
            
            saved_optimizer_state_dict = checkpoint['optimizer']

            # print(optimizer.param_groups[0]['params'])

            # print(optimizer.state.keys())

            current_group_all_params = list(optimizer.param_groups[0]['params']) + list(optimizer.param_groups[0]['params'])

            for saved_state, current_state in zip(saved_optimizer_state_dict['state'], current_group_all_params):
                
                # print(saved_optimizer_state_dict['state'][saved_state]['exp_avg'].shape, 
                #       current_state.shape)
                pass
            # print(saved_optimizer_state_dict['state'].keys())

            # # saved_optimizer_state_dict['param_groups'].reverse()
            # params_sub1 = saved_optimizer_state_dict['param_groups'][0]['params']
            # params_sub2 = saved_optimizer_state_dict['param_groups'][1]['params']

            # print(saved_optimizer_state_dict['param_groups'][0]['params'])
            # print(saved_optimizer_state_dict['param_groups'][1].keys())

            # for key in saved_optimizer_state_dict['param_groups'][0].keys():
            #     print(key)
            #     print(saved_optimizer_state_dict['param_groups'][0][key])


            # print(saved_optimizer_state_dict['param_groups']

            # all_params_group = []
            # for param_group in saved_optimizer_state_dict['param_groups']:
            #     all_params_group.extend(param_group)


            # sub_group_1 = []
            # sub_group_2 = []

            # sub_group_1.append(param_group for i, param_group in enumerate(all_params_group) if i < 130)
            # sub_group_2.append(param_group for i, param_group in enumerate(all_params_group) if i >= 130)


            # saved_optimizer_state_dict['param_groups'][0] = sub_group_1
            # saved_optimizer_state_dict['param_groups'][1] = sub_group_2
            # print(optimizer.param_groups[0]['params'][81])
            # saved_scaler = checkpoint['scaler']

            # print(saved_scaler.keys())

            # print(saved_scaler['scale'], loss_scaler.state_dict()['scale'])
            # print(saved_scaler['growth_factor'], loss_scaler.state_dict()['growth_factor'])
            # print(saved_scaler['backoff_factor'], loss_scaler.state_dict()['backoff_factor'])
            # print(saved_scaler['growth_interval'], loss_scaler.state_dict()['growth_interval'])
            # print(saved_scaler['_growth_tracker'], loss_scaler.state_dict()['_growth_tracker'])



            # # Check and filter state (like momentum, RMS, etc.)
            # for param_tensor in saved_optimizer_state_dict['state']:
            #     if param_tensor in list(model_without_ddp.parameters()):
            #         filtered_optimizer_state_dict['state'][param_tensor] = saved_optimizer_state_dict['state'][param_tensor]

            # Check and filter parameter groups

            num_params = []
            total_params = 0

            for current_group in optimizer.param_groups:
                num_params.append((total_params, total_params + len(current_group['params'])))
                total_params += len(current_group['params'])

            for i in range(len(num_params)):
                num_params[i] = (-(num_params[i][0] - total_params) - 1 , -(num_params[i][1] - total_params) - 1)


            # for saved_group, num_params_range in zip(saved_optimizer_state_dict['param_groups'], num_params):
            #     saved_group['params'] = [x for x in range(num_params_range[0], num_params_range[1], -1)]

            #     print(saved_group['params'] )


            optimizer.load_state_dict(saved_optimizer_state_dict)
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x