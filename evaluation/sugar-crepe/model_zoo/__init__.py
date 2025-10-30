import clip
from .config import prepare_model
from extlib.hycoclip.utils.checkpointing import CheckpointManager
import torchvision.transforms as T

CACHE_DIR = "."

def get_model(model_name, checkpoint, device, root_dir=CACHE_DIR):
    """
    Helper function that returns a model and a potential image preprocessing function.
    """
    if "openai-clip" in model_name:
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        model, image_preprocess = clip.load(variant, device=device, download_root=root_dir)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    elif model_name == "clip_vit_l":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(checkpoint)
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess

    elif model_name == "meru_vit_l":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(checkpoint)
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess

    elif model_name == "clip_vit_b":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(checkpoint)
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess

    elif model_name == "meru_vit_b":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(checkpoint)
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess

    elif model_name == "clip_vit_s":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(checkpoint)
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess

    elif model_name == "meru_vit_s":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(checkpoint)
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess
    
    elif model_name == "hycoclip_vit_s":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(checkpoint)
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess
    
    elif model_name == "hycoclip_vit_b":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(checkpoint)
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess