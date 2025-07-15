import logging
import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, scene_num, cfg, prompt_learner=None, text_encoder=None):
        super(build_transformer, self).__init__()
        self.use_adapter = cfg.MODEL.MLP_ADAPTER
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.scene_num = scene_num
        self.sie_coe = cfg.MODEL.SIE_COE
        
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(cfg, self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        self.cam_vis = False
        self.attnmap_vis = False

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * scene_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
            print('scene number is : {}'.format(scene_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(scene_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('scene number is : {}'.format(scene_num))
        
        dataset_name = cfg.DATASETS.NAMES
        if prompt_learner is None:
            self.prompt_learner = PromptLearner(cfg, num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        else:
            self.prompt_learner = prompt_learner

        if text_encoder is None:
            self.text_encoder = TextEncoder(clip_model)
        else:
            self.text_encoder = text_encoder

    def forward(self, x, proto=None, label=None, cam_label= None, view_label=None, get_text=False, get_image=False, peft_stage=True):
        if proto is not None:
            proto_proj = proto @ self.image_encoder.proj
            return proto_proj

        if get_text == True:
            prompts = self.prompt_learner(label) 
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features
        
        if get_image == True:
            if self.use_adapter:
                image_features_last, image_features, image_features_proj, attn_map = self.image_encoder(x, peft_stage=peft_stage) 
            else:
                image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features[0]
            elif self.model_name == 'ViT-B-16':
                return image_features[:,0]
            
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) #B,512  B,128,512
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None

            if self.use_adapter:
                image_features_last, image_features, image_features_proj, attn_map = self.image_encoder(x, cv_embed, attnmap_vis=self.attnmap_vis) #B,512  B,128,512
            else:
                image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) #B,512  B,128,512

            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 

        if self.attnmap_vis:
            return attn_map
        
        if self.cam_vis == True:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return cls_score

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj
        else:
            return torch.cat([img_feature, img_feature_proj], dim=1), torch.cat([feat, feat_proj], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class build_multi_branch(nn.Module):

    def __init__(self, num_class, camera_num, scene_num, cfg):
        super(build_multi_branch, self).__init__()
        self.num_classes = num_class
        self.cfg = cfg
        self.model1 = build_transformer(num_class, camera_num, scene_num, cfg)
        if cfg.MODEL.VISUAL_BRANCH == 'single':
            self.model2 = self.model1
            if self.cfg.DATASETS.NAMES != 'RGBN300':
                self.model3 = self.model1
        elif cfg.MODEL.VISUAL_BRANCH == 'multi':
            if cfg.MODEL.TEXT_BRANCH == 'single':
                self.model2 = build_transformer(num_class, camera_num, scene_num, cfg, prompt_learner=self.model1.prompt_learner, text_encoder = self.model1.text_encoder)
                if self.cfg.DATASETS.NAMES != 'RGBN300':
                    self.model3 = build_transformer(num_class, camera_num, scene_num, cfg, prompt_learner=self.model1.prompt_learner, text_encoder = self.model1.text_encoder)
            elif cfg.MODEL.TEXT_BRANCH == 'multi':
                self.model2 = build_transformer(num_class, camera_num, scene_num, cfg, text_encoder = self.model1.text_encoder)
                if self.cfg.DATASETS.NAMES != 'RGBN300':
                    self.model3 = build_transformer(num_class, camera_num, scene_num, cfg, text_encoder = self.model1.text_encoder)

    def forward(self, x1=None, x2=None, x3=None, x1_proto=None, x2_proto=None, x3_proto=None, \
                label=None, cam_label= None, view_label=None, get_text=False, get_image=False, \
                    force_traning=False, peft_stage=True):
        model1 = self.model1(x1, proto=x1_proto, label=label, cam_label=cam_label, view_label=view_label, get_text=get_text, get_image=get_image, peft_stage=peft_stage)
        model2 = self.model2(x2, proto=x2_proto, label=label, cam_label=cam_label, view_label=view_label, get_text=get_text, get_image=get_image, peft_stage=peft_stage)
        if self.cfg.DATASETS.NAMES != 'RGBN300':
            model3 = self.model3(x3, proto=x3_proto, label=label, cam_label=cam_label, view_label=view_label, get_text=get_text, get_image=get_image, peft_stage=peft_stage)

        if self.training or force_traning:
            if self.cfg.DATASETS.NAMES != 'RGBN300':
                return model1, model2, model3
            else:
                return model1, model2
        else:
            if self.cfg.DATASETS.NAMES != 'RGBN300':
                if self.model1.attnmap_vis:
                    return model1, model2, model3
                return torch.cat([model1[0], model2[0], model3[0]],dim=1), torch.cat([model1[1], model2[1], model3[1]],dim=1)
            else:
                return torch.cat([model1[0], model2[0]],dim=1), torch.cat([model1[1], model2[1]],dim=1)
                
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'cv_embed' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

def make_model(cfg, num_class, camera_num, scene_num):
    model = build_multi_branch(num_class, camera_num, scene_num, cfg)
    return model


from .clip_adapter import clip as clip_adapter
from .clip import clip
def load_clip_to_cpu(config, backbone_name, h_resolution, w_resolution, vision_stride_size):
    if config.MODEL.MLP_ADAPTER:
        url = clip_adapter._MODELS[backbone_name]
        model_path = clip_adapter._download(url)
    else:
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if config.MODEL.MLP_ADAPTER:
        model = clip_adapter.build_model_parallel_adapter(config, state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    else:
        model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        self.prompt_num = cfg.MODEL.PROMPT_NUM


        # ------------build text prompt------------
        vehicle_datasets = {"MSVR310", "WMVEID863", "RGBN300", "RGBNT100"}
        object_type = "vehicle" if dataset_name in vehicle_datasets else "person"
        x_tokens = " ".join(["X"] * self.prompt_num)
        ctx_init = f"A photo of a {x_tokens} {object_type}."
        # -----------------------------------------

        logger = logging.getLogger("ICPL.image_train")
        logger.info(ctx_init)

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = self.prompt_num
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = self.prompt_num
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 

