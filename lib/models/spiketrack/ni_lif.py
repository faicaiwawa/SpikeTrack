import torch
import torch.nn as nn
import torch.nn.functional as F

class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value, max_value):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None

class MultiSpike(nn.Module):
    def __init__(
            self,
            min_value=0,
            max_value=4,
            Norm=None,
    ):
        super().__init__()
        if Norm == None:
            self.Norm = max_value
        else:
            self.Norm = Norm
        self.min_value = min_value
        self.max_value = max_value

    @staticmethod
    def spike_function(x, min_value, max_value):
        return Quant.apply(x, min_value, max_value)

    def __repr__(self):
        return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"

    def forward(self, x):  # B C H W
        return self.spike_function(x, min_value=self.min_value, max_value=self.max_value) / (self.Norm)


class mem_update(nn.Module):
    def __init__(self, time_step, Norm=None, skip_ts=False):
        super(mem_update, self).__init__()
        self.qtrick = MultiSpike(Norm=Norm)  # change the max value
        self.skip_ts = skip_ts
        self.time_step = time_step

        if not self.skip_ts:
            self.decay = nn.Parameter(torch.full((time_step - 1,), init_sigmoid_param(0.25)))

    def forward(self, x):
        # x.shape [T, *]
        transpose_flag = False

        if not self.skip_ts:
            if x.shape[0] > 3: # temporary solution, for temporal-fusion-mlp neuron input shape [B*H*W, C, T]
                x = x.permute(2,0,1)
                transpose_flag = True
            output = torch.zeros_like(x)
            mem_old = 0
            time_window = x.shape[0]

            if time_window > 3:
                raise ValueError("warning! network has bugs, time window dimension > 3 in somewhere.")
                    
            spike = torch.zeros_like(x[0]).to(x.device)
            for i in range(time_window):
                if i >= 1:
                    alpha = self.decay[i - 1].sigmoid()
                    mem = (mem_old - spike.detach()) * alpha + x[i]
                else:
                    mem = x[i]
                spike = self.qtrick(mem)
                mem_old = mem.clone()
                output[i] = spike
            if transpose_flag:
                output = output.permute(1,2,0)
        else:
            output = self.qtrick(x)
        return output

def init_sigmoid_param(p):
    return torch.log(torch.tensor(p) / (1 - torch.tensor(p)))



