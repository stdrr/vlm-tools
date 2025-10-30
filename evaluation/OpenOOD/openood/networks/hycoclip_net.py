import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .hycoclip.models import HyCoCLIP, MERU, CLIPBaseline
from .hycoclip.utils.checkpointing import CheckpointManager
from .hycoclip.encoders.text_encoders import TransformerTextEncoder
from .hycoclip.encoders.image_encoders import build_timm_vit
from .hycoclip.tokenizer import Tokenizer

from .hycoclip import lorentz as L


def prepare_model(train_config):
    textual = TransformerTextEncoder(arch="L12_W512", vocab_size=49408, context_length=77)
    if train_config == 'hyclip_vit_l':
        visual = build_timm_vit(arch="vit_large_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = CLIPBaseline(visual= visual, textual = textual, embed_dim = 512)
    elif train_config == 'hyclip_vit_b':
        visual = build_timm_vit(arch="vit_base_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = CLIPBaseline(visual= visual, textual = textual, embed_dim = 512)
    elif train_config == 'hyclip_vit_s':
        visual = build_timm_vit(arch="vit_small_mocov3_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = CLIPBaseline(visual= visual, textual = textual, embed_dim = 512)
    elif train_config == 'meru_vit_l':
        visual = build_timm_vit(arch="vit_large_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = MERU(visual=visual, textual=textual, embed_dim=512, curv_init=1.0, learn_curv=True, entail_weight=0.2,)
    elif train_config == 'meru_vit_b':
        visual = build_timm_vit(arch="vit_base_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = MERU(visual=visual, textual=textual, embed_dim=512, curv_init=1.0, learn_curv=True, entail_weight=0.2,)
    elif train_config == 'meru_vit_s':
        visual = build_timm_vit(arch="vit_small_mocov3_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = MERU(visual=visual, textual=textual, embed_dim=512, curv_init=1.0, learn_curv=True, entail_weight=0.2,)
    elif 'hycoclip_vit_s' in train_config:
        visual = build_timm_vit(arch="vit_small_mocov3_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = HyCoCLIP(visual=visual, textual=textual, embed_dim=512, curv_init=1.0, learn_curv=True, entail_weight=0.2,)
    elif 'hycoclip_vit_b' in train_config:
        visual = build_timm_vit(arch="vit_base_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = HyCoCLIP(visual=visual, textual=textual, embed_dim=512, curv_init=1.0, learn_curv=True, entail_weight=0.2,)
    else:
        raise NotImplementedError
    return model




# https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
@torch.inference_mode()
def zeroshot_classifier(model, tokenizer, classnames, templates):
    zeroshot_weights = []
    for classname in tqdm(classnames):
        texts = [template.format(classname)
                    for template in templates]  # format with class
        texts = tokenizer(texts)  # tokenize
        class_embeddings = model.encode_text(
            texts, project=False)  # embed with text encoder
        if isinstance(model, (MERU, HyCoCLIP)):
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding * model.textual_alpha.exp()
            class_embedding = L.exp_map0(class_embedding, model.curv.exp())
        else:
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats.mean(dim=0)
            text_feats /= text_feats.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
    return zeroshot_weights


class HyperbolicCLIPZeroshot(nn.Module):
    def __init__(self, classnames, templates, model_name, checkpoint):
        super().__init__()
        self.preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),])
        self.tokenizer = Tokenizer()
        self.model = prepare_model(model_name)
        self.model.cuda()
        CheckpointManager(model=self.model).load(checkpoint)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.zeroshot_weights = zeroshot_classifier(self.model, self.tokenizer, classnames,
                                                    templates)
        
    def load_state_dict(self, state_dict, strict = True, assign = False):
        return self

    @torch.inference_mode()
    def forward(self, x):
        image_features = self.model.encode_image(x, project=True)
        if isinstance(self.model, (MERU, HyCoCLIP)):
            logits = L.pairwise_inner(image_features, self.zeroshot_weights, self.model.curv.exp())
        else:
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ self.zeroshot_weights
        return logits
    
    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=256, normalize=False):
        num_text = len(texts)
        text_embeds = []
        tqdm_loader = tqdm(range(0, num_text, text_batch_size))
        tqdm_loader.set_description("Computing text embeddings")
        for i in tqdm_loader:
            text = texts[i: min(num_text, i + text_batch_size)]
            text_input = self.tokenizer(text)
            text_feats = self.model.encode_text(text_input, project=False)
            if isinstance(self.model, (MERU, HyCoCLIP)):
                text_feats  = text_feats.mean(dim=0)
                text_feats  = text_feats  * self.model.textual_alpha.exp()
                text_feats  = L.exp_map0(text_feats , self.model.curv.exp())
            else:
                text_feats /= text_feats.norm(dim=-1, keepdim=True)
                text_feats = text_feats.mean(dim=0)
                text_feats /= text_feats.norm()
            text_embeds.append(text_feats)

        text_embeds = torch.cat(text_embeds, dim=0)
        return text_embeds
    
    @torch.no_grad()
    def get_image_embeddings(self, image_loader, normalize=False):
        image_embeds = []
        tqdm_loader = tqdm(image_loader)
        tqdm_loader.set_description("Computing image embeddings")
        for batch in tqdm_loader:
            images = batch["image"]
            image_feats = self.model.encode_image(images, project=True)
            image_embeds.append(image_feats)

        image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds
    
    @torch.no_grad()
    def get_retrieval_scores_dataset(self, loader):
        captions = loader.dataset.text
        text_embeds = self.get_text_embeddings(captions, normalize=True)
        image_embeds = self.get_image_embeddings(loader, normalize=True)
        if isinstance(self.model, (MERU, HyCoCLIP)):
            scores = L.pairwise_inner(image_embeds, text_embeds, self.model.curv.exp())
        else:
            scores = image_embeds @ text_embeds.T
        scores = scores.cpu().numpy()
        return scores

    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        scores = []
        tqdm_loader = tqdm(joint_loader)
        tqdm_loader.set_description("Computing retrieval scores")
        for batch in tqdm_loader:
            image_options = []
            for i_option in batch["image_options"]:
                image_embeddings = self.model.encode_image(i_option.to(self.device), project=True).cpu().numpy()  # B x D
                image_options.append(np.expand_dims(image_embeddings, axis=1))

            caption_options = []
            for c_option in batch["caption_options"]:
                caption_tokenized = self.tokenizer(list(c_option))
                caption_embeddings = self.model.encode_text(caption_tokenized, project=False)  # B x D
                if isinstance(self.model, (MERU, HyCoCLIP)):
                    caption_embeddings = caption_embeddings * self.model.textual_alpha.exp()
                    caption_embeddings = L.exp_map0(caption_embeddings, self.model.curv.exp())
                else:
                    caption_embeddings /= caption_embeddings.norm(dim=-1, keepdim=True)
                caption_embeddings = caption_embeddings.cpu().numpy()
                caption_options.append(np.expand_dims(caption_embeddings, axis=1))

            image_options = np.concatenate(image_options, axis=1)  # B x K x D
            caption_options = np.concatenate(caption_options, axis=1)  # B x L x D
            if isinstance(self.model, (MERU, HyCoCLIP)):
                batch_scores = L.pairwise_inner_batched(torch.tensor(image_options).cuda(), torch.tensor(caption_options).cuda(), self.model.curv.exp())
            else:
                batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options)  # B x K x L
            scores.append(batch_scores)

        all_scores = np.concatenate(scores, axis=0)  # N x K x L
        return all_scores