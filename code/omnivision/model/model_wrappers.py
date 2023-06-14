# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from code.omnivision.data.api import VisionSample

import logging
import gc
class MIMOHeadWrapper(nn.Module):
    """Attaches multiple input multiple output heads to the trunk using forward hooks.

    Args:
        trunk: Any model to which you want to attach the heads to.
        heads: A list of dicts with the following keys:
            fork_module: The module which the head will be applied to. It can be an
                empty string, in which case the head is attached to the trunk's output.
            head: The head which is to be attached.
            input_key: The head will only run on inputs with this key. If set to
                `None` the head will be applied to all inputs.
            output_key: The head will produce this output key. If set to `None`, the
                output key will be the same as the input key.

            An example heads value can look like -
            ```
            [
                {
                    "fork_module": "layer_1.layer_a.layer_alpha",
                    "head": nn.Linear(in_feat, out_feat),
                    "input_key": "dataset_1",
                    "output_key": "out_1",
                },
                {
                    "fork_module": "",
                    "head": nn.Linear(in_feat, out_feat),
                    "input_key": "dataset_1",
                    "output_key": "out_2",
                },
                {
                    "fork_module": "",
                    "head": nn.Linear(in_feat, out_feat),
                    "input_key": "dataset_2",
                    "output_key": "out_3",
                },
                {
                    "fork_module": "",
                    "head": nn.Conv2d(in_feat, out_feat),
                    "input_key": None,
                    "output_key": None,
                },
            ]
            ```
        trunk_fields: A list of dicts with the following keys:
            input_key: The input key this rule applies to. If `None`, applies to all
                inputs.
            args: These specific keys will be fetched from the sample and passed as
                *args to the trunk for the specified `input_key`.
            kwargs: These specific keys will be fetched from the sample and passed as
                **kwargs to the trunk for the specified `input_key`.

            Example -
            ```
            [
                {
                    "input_key": "dataset_1",
                    "args": ["vision"]
                },
                {
                    "input_key": "dataset_2",
                    "args": ["vision"],
                    "kwargs": {"mask": "mask"}
                },
            ]
            ```

        Note that two heads cannot produce the same output key in the same forward pass.

    Returns:
        A dict with keys corresponding to the output keys which match with the input key.
    """

    @dataclass
    class HeadArgs:
        fork_module: str
        head: nn.Module
        input_key: Optional[str]
        output_key: Optional[str]

    @dataclass
    class TrunkFieldArgs:
        input_key: Optional[str]
        args: List[str] = field(default_factory=list)
        kwargs: Dict[str, str] = field(default_factory=dict)

    def __init__(
        self,
        trunk: nn.Module,
        heads: List[Dict],
        trunk_fields: List[Dict],
        handle_list_inputs=False,
        disable_hooks=False,
    ) -> None:
        """WARNING: handle_list_inputs is a hack which needs to be refactored away.
        disable_hooks only works with 1 head. Gets output without use of hooks"""
        super().__init__()

        self.trunk = trunk
        self.handle_list_inputs = handle_list_inputs

        # cast to HeadArgs for input validation
        heads = [self.HeadArgs(**head_dict) for head_dict in heads]
        # cast to TrunkFieldArgs for input validation
        trunk_fields = [
            self.TrunkFieldArgs(**trunk_fields_dict)
            for trunk_fields_dict in trunk_fields
        ]

        self.head_name_to_fork_module = {}
        self.heads = nn.ModuleList()
        self.head_input_keys = []
        self.head_output_keys = []
        self.head_fork_modules = []

        for head_args in heads:
            self.heads.append(head_args.head)
            self.head_input_keys.append(head_args.input_key)
            self.head_output_keys.append(head_args.output_key)
            self.head_fork_modules.append(head_args.fork_module)

        self.trunk_field_args = {}
        self.trunk_field_kwargs = {}
        for trunk_fields_elem in trunk_fields:
            input_key = trunk_fields_elem.input_key
            if input_key in self.trunk_field_args:
                raise KeyError(
                    f"Multiple trunk_fields specified for the same input_key: {input_key}"
                )
            self.trunk_field_args[input_key] = trunk_fields_elem.args
            self.trunk_field_kwargs[input_key] = trunk_fields_elem.kwargs

        # outputs is used as a temporary storage of the head outputs
        self.outputs = {}

        # input_key is used to specify which key is currently being processed
        self.input_key = None

        # handles to the hooks which can be used for removing the hooks if needed
        self.disable_hooks = disable_hooks
        if not self.disable_hooks:
            self.hook_handles = []
            self._register_hooks()

    def _register_hooks(self):
        for i, head in enumerate(self.heads):
            fork_module_name = self.head_fork_modules[i]

            def hook_fn(
                module,
                module_in,
                module_out,
                # the following variables are passed as kwargs in the closure to avoid
                # late binding in python
                head_method=head,
                in_key=self.head_input_keys[i],
                out_key=self.head_output_keys[i],
            ):
                if in_key is not None and self.input_key != in_key:
                    return
                if out_key is None:
                    out_key = self.input_key
                if out_key in self.outputs:
                    # reset state before raising
                    self.outputs = {}
                    self.input_key = None
                    raise ValueError(
                        f"Two heads produced the same output key `{out_key}` during forward"
                    )
                self.outputs[out_key] = head_method(module_out)

            fork_module = self.trunk.get_submodule(fork_module_name)
            self.hook_handles.append(fork_module.register_forward_hook(hook_fn))

    def _get_trunk_fields(self):
        fields_args = self.trunk_field_args.get(self.input_key)
        fields_kwargs = self.trunk_field_kwargs.get(self.input_key)
        if fields_args is None:
            assert fields_kwargs is None
            fields_args = self.trunk_field_args.get(None)
            fields_kwargs = self.trunk_field_kwargs.get(None)
            if fields_args is None:
                assert fields_kwargs is None
                raise ValueError(
                    f"No trunk fields specified for input key: {self.input_key}"
                )
        return fields_args, fields_kwargs

    def freeze_trunk(self, excluded_keywords=None):
        if excluded_keywords is None:
            excluded_keywords = []
        logging.info(f"Freezing trunk with excluded keywords {excluded_keywords}")
        for name, param in self.trunk.named_parameters():
            if param.requires_grad and all([x not in name for x in excluded_keywords]):
                param.requires_grad = False

    def forward_sub_batch(self, sub_batch, *args, **kwargs):
        assert isinstance(sub_batch, VisionSample), f"Received {type(sub_batch)}"
        fields_args, fields_kwargs = self._get_trunk_fields()
        sample_args = [getattr(sub_batch, arg) for arg in fields_args]
        sample_kwargs = {
            key: getattr(sub_batch, field) for key, field in fields_kwargs.items()
        }
        self.trunk(*sample_args, *args, **sample_kwargs, **kwargs)

    def forward_sub_batch_without_hook(self, sub_batch, *args, **kwargs):
        assert isinstance(sub_batch, VisionSample), f"Received {type(sub_batch)}"
        fields_args, fields_kwargs = self._get_trunk_fields()
        sample_args = [getattr(sub_batch, arg) for arg in fields_args]
        sample_kwargs = {
            key: getattr(sub_batch, field) for key, field in fields_kwargs.items()
        }
        features = self.trunk(*sample_args, *args, **sample_kwargs, **kwargs)
        assert len(self.heads) == 1, "forward_sub_batch_without_hook can only be used with 1 head"
        head_key = self.head_output_keys[0] if self.head_output_keys[0] is not None else self.input_key
        return {head_key: self.heads[0](features)}


    def forward(self, batch, *args, **kwargs) -> Dict:
        assert isinstance(batch, Mapping)
        assert len(self.outputs) == 0
        for key, sub_batch in batch.items():
            self.input_key = key
            if self.handle_list_inputs and isinstance(sub_batch.vision, Sequence):
                # FIXME: this only handles list inputs for the field "vision"
                assert len(batch) == 1
                out_vals = []
                for e in sub_batch.vision:
                    e_batch = copy.copy(sub_batch)
                    e_batch.vision = e
                    if self.disable_hooks:
                        self.outputs = self.forward_sub_batch_without_hook(sub_batch, *args, **kwargs)
                    else:
                        self.forward_sub_batch(e_batch, *args, **kwargs)
                    assert len(self.outputs) == 1
                    out_key, out_val = self.outputs.popitem()
                    out_vals.append(out_val)
                return {out_key: torch.cat(out_vals)}
            else:
                if self.disable_hooks:
                    return self.forward_sub_batch_without_hook(sub_batch, *args, **kwargs)
                else:
                    self.forward_sub_batch(sub_batch, *args, **kwargs)
        outputs = self.outputs
        self.input_key = None
        self.outputs = {}
        return outputs


class AverageWrapper(nn.Module):
    '''
    Multiple MiMO Head Wrappers
    '''
    def __init__(
            self,
            trunk: nn.Module,
            heads: List[Dict],
            trunk_fields: List[Dict],
            handle_list_inputs=False,
            num_models=1,
            combine_mode=None,
    ) -> None:
        """WARNING: handle_list_inputs is a hack which needs to be refactored away."""
        super().__init__()
        self.model = MIMOHeadWrapper(copy.deepcopy(trunk), copy.deepcopy(heads), copy.deepcopy(trunk_fields),
                                     handle_list_inputs,
                                     disable_hooks=False
                                     )
        _, self.names = make_functional(self.model, exclude_prefix=['head'])
        torch.cuda.empty_cache()
        self.num_models = num_models
        self.combine_mode = combine_mode
        self.model_weights = []
        if self.combine_mode == 'learned_average_softmax':
            self.mixing_ratio = torch.nn.Parameter(data=torch.ones(self.num_models))
        elif self.combine_mode == 'learned_average_layer_softmax':
            self.mixing_ratio = torch.nn.Parameter(data=torch.ones(len(self.names), self.num_models))

    def load_averaged_weights_to_model(self):
        if self.combine_mode == 'learned_average_softmax':
            mixing_ratio = torch.nn.functional.softmax(self.mixing_ratio, dim=-1)
            averaged_weights = self.average_weights(self.model_weights,
                                                    mixing_ratio,
                                                    exclude_prefix=['heads.'])
            return averaged_weights
        elif self.combine_mode == 'learned_average_layer_softmax':
            mixing_ratio = torch.nn.functional.softmax(self.mixing_ratio, dim=-1)
            averaged_weights = self.average_weights(self.model_weights,
                                                    mixing_ratio,
                                                    exclude_prefix=['heads.'],
                                                    by_layer=True)
            return averaged_weights
        else:
            raise Exception(f'Invalid combine mode {self.combine_mode}')


    def model_weights_to_cuda(self, weights):
        for weight_dic in weights:
            for layer in weight_dic:
                weight_dic[layer] = weight_dic[layer].detach().cuda()


    def freeze_trunk(self, excluded_keywords=None):
        if excluded_keywords is None:
            excluded_keywords = []
        assert hasattr(self.model, 'trunk'), "Require model to be an instance of MIMOHeadWrapper with attribute trunk"
        for name, param in self.model.trunk.named_parameters():
            if param.requires_grad and all([x not in name for x in excluded_keywords]):
                param.requires_grad = False
    def load_state_dict(self, weights: Dict, strict=True):
        if 'models' in weights:
            self.model_weights = weights['models']
            # key_info = self.model.load_state_dict(self.model_weights[0], strict=False)
            self.model_weights_to_cuda(self.model_weights)
            return '',''
        else:
            return super(AverageWrapper, self).load_state_dict(weights, strict=strict)

    def average_weights(self, weights, scale, exclude_prefix=[], by_layer=False):
        new_weights_dic = {}
        for i, layer in enumerate(self.names):
            if any([layer[:len(prefix)] == prefix for prefix in exclude_prefix]):
                continue

            mixing_ratio = scale[i] if by_layer else scale
            for weights_dic, s in zip(weights, mixing_ratio):
                if layer not in new_weights_dic:
                    new_weights_dic[layer] = weights_dic[layer] * s
                else:
                    new_weights_dic[layer] += weights_dic[layer] * s
        return new_weights_dic

    def forward(self, batch, *args, **kwargs) -> Dict:

        assert isinstance(batch, Mapping)
        gc.collect() # Important: Prevents memory leak
        averaged_weights = self.load_averaged_weights_to_model()
        out = torch.nn.utils.stateless.functional_call(self.model,
                                             averaged_weights,
                                             batch, kwargs=kwargs)

        return out

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])
def make_functional(model, exclude_prefix=[]):
    orig_params = tuple(model.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(model.named_parameters()):
        if any([name[:len(prefix)] == prefix for prefix in exclude_prefix]):
            continue
        del_attr(model, name.split("."))
        names.append(name)
    return orig_params, names
