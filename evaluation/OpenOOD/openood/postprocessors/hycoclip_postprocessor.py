from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor

from openood.networks.hycoclip.lorentz import pairwise_inner, pairwise_angle_at_origin



class HyCoCLIPPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(HyCoCLIPPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False
        self.scoring_function = pairwise_inner

    def setup(self, net, id_loader_dict, ood_loader_dict, scoring_function=None):
        self._net = net
        self._id_classifier = id_loader_dict
        self.scoring_function = scoring_function if scoring_function is not None else self.scoring_function

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        if net is None:
            assert hasattr(self, "_net"), "Net is not set up. Please call setup() first."
            net = self._net
        
        with torch.inference_mode():
            image_feats = net.encode_image(data.to(net.device), project=True)

        scores = self.scoring_function(self._id_classifier, image_feats, net.curv.exp())

        # check if the scoring function is the pairwise inner product
        # or the entailment score
        if "pairwise_inner" in self.scoring_function.__name__ or "entailment_score" in self.scoring_function.__name__:
            pred_fn = lambda x: torch.max(x, dim=0)
        elif "pairwise_dist" in self.scoring_function.__name__:
            pred_fn = lambda x: torch.min(x, dim=0)

        conf, pred = pred_fn(scores)

        return pred, conf
