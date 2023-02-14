import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple

import torch
from torch.nn.utils.prune import _validate_pruning_amount, _validate_pruning_amount_init, _compute_nparams_toprune
from torch.nn.utils.prune import BasePruningMethod, global_unstructured, L1Unstructured

class LossUnstructured(BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor by zeroing out the ones
    with the lowest L1-norm.
    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        # if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
        #     # largest=True --> top k; largest=False --> bottom k
        #     # Prune the smallest k
        #     t = torch.abs(t).view(-1)
        #     # ind = (t == 0).nonzero().view(-1)
        #     # t[ind] = torch.tensor(520.).to(t.device)
        #     topk = torch.topk(t, k=nparams_toprune, largest=True)
        #     # topk will have .indices and .values
        #     mask.view(-1)[topk.indices] = 0

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            # t = t.view(-1)
            t = t.view(-1)
            # ind = (t < 0).nonzero().view(-1)
            # t[ind] = torch.tensor(0.).to(t.device)
            topk = torch.topk(t, k=nparams_toprune, largest=False)
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        # if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
        #     # largest=True --> top k; largest=False --> bottom k
        #     # Prune the smallest k
        #     t = torch.abs(t).view(-1)
        #     # ind = (t == 0).nonzero().view(-1)
        #     # t[ind] = torch.tensor(520.).to(t.device)
        #     topk = torch.topk(t, k=nparams_toprune, largest=True)
        #     # topk will have .indices and .values30.43  
        #     rand_ind = torch.randperm(len(topk.indices))[:int(len(topk.indices)*0.5)]
        #     mask.view(-1)[topk.indices[rand_ind]] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super(LossUnstructured, cls).apply(
            module, name, amount=amount, importance_scores=importance_scores
        )

def loss_unstructured(module, name, amount, importance_scores=None):
    r"""Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified `amount` of (currently unpruned) units with the
    lowest L1-norm.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.
    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        importance_scores (torch.Tensor): tensor of importance scores (of same
            shape as module parameter) used to compute mask for pruning.
            The values in this tensor indicate the importance of the corresponding
            elements in the parameter being pruned.
            If unspecified or None, the module parameter will be used in its place.
    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module
    Examples:
        >>> m = prune.l1_unstructured(nn.Linear(2, 3), 'weight', amount=0.2)
        >>> m.state_dict().keys()
        odict_keys(['bias', 'weight_orig', 'weight_mask'])
    """
    LossUnstructured.apply(
        module, name, amount=amount, importance_scores=importance_scores
    )
    return module


