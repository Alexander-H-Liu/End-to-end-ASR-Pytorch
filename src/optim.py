import torch


class Optimizer():
    def __init__(self, parameters, tf_start, tf_end, tf_step, optimizer, lr, lr_scheduler, *kwargs):
        
        # Setup teacher forcing scheduler
        self.tf_rate = lambda step: max(tf_end, tf_start-(tf_start-tf_end)*step/tf_step)

        # Setup torch optimizer
        opt = getattr(torch.optim,optimizer)
        if lr_scheduler == 'warmup':
            warmup_step = 4000.0
            lr_func = lambda step: init_lr * warmup_step **0.5 * np.minimum((step+1)*warmup_step**-1.5,(step+1)**-0.5 )
            self.opt = opt(parameters,lr=1.0)
            self.lr_scheduler = LambdaLR(self.opt,lr_func)
        else:
            self.lr_scheduler = None
            self.opt = opt(parameters,lr=lr)

    def pre_step(self, step):
        self.opt.zero_grad()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return self.tf_rate(step)

    def step(self):
        self.opt.step()



