import numpy as np
import matplotlib.pyplot as plt

from fastai.vision.all import SchedExp, SchedPoly, SchedCos, ParamScheduler


def _get_scheduler(sched_cls, lr_from, lr_to, mom_from, mom_to, *args, **kwargs):
    scheds = {'lr': sched_cls(lr_from, lr_to, *args, **kwargs)}
    if mom_from is not None and mom_to is not None:
        scheds['mom'] = sched_cls(mom_from, mom_to, *args, **kwargs)
    sh = ParamScheduler(scheds)
    return sh


def get_exp_scheduler(lr_from, lr_to, mom_from=None, mom_to=None):
    return _get_scheduler(SchedExp, lr_from, lr_to, mom_from, mom_to)


def get_poly_scheduler(lr_from, lr_to, mom_from=None, mom_to=None, *, power=0.5):
    return _get_scheduler(SchedPoly, lr_from, lr_to, mom_from, mom_to, power=power)


def get_cos_scheduler(lr_from, lr_to, mom_from=None, mom_to=None):
    return _get_scheduler(SchedCos, lr_from, lr_to, mom_from, mom_to)


def plot_scheduler(sh, epochs=30):
    assert 'lr' in sh.scheds

    # compute progress of lrs and moms
    x = np.linspace(0, 1, epochs)
    lrs = [sh.scheds['lr'](i) for i in x]
    moms = [sh.scheds['mom'](i) for i in x] if 'mom' in sh.scheds else None

    # plot results
    if moms is not None:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        axs = [ax, None]
    for arr, name, ax in zip([lrs, moms], ['lr', 'mom'], axs):
        if arr is not None:
            ax.plot(arr)
            ax.set(title=name, ylabel=name)
    plt.show()


# from fastai.vision.all import Learner, CancelStepException
# class EffectiveBatchLearner(Learner):
#     def set_accumulation_bs(self, total_batch_size=512):
#         assert np.log2(total_batch_size).is_integer(), 'Batch Size should be power of 2'
#         bs = self.dls.bs
#         assert total_batch_size >= bs
#         self.accumulation_steps = total_batch_size / bs
#         print(f'Accumulation steps: {self.accumulation_steps}')
#         assert self.accumulation_steps > 0 and self.accumulation_steps.is_integer()

#     def _do_one_batch(self):
#         self.pred = self.model(*self.xb)
#         self('after_pred')
#         if len(self.yb):
#             self.loss_grad = self.loss_func(self.pred, *self.yb)
#             self.loss_grad = self.loss_grad / self.accumulation_steps
#             self.loss = self.loss_grad.clone()
#         self('after_loss')
#         if not self.training or not len(self.yb):
#             return

#         self('before_backward')
#         self.loss_grad.backward()

#         # gradient accumulation
#         i = self.iter
#         if (i+1) % self.accumulation_steps == 0:
#             self._with_events(self.opt.step, 'step', CancelStepException)
#             self.opt.zero_grad()
