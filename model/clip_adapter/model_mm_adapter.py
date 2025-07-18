import logging
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .msa import MultiheadAttention

logger = logging.getLogger("ICPL.image_train")

GLOBAL_CONFIG = None

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim) 
        self.num_heads = num_heads

    def forward(self, x): 
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC  #32,2048,7,7 ->49, 32, 2048
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC  50,32,2048
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        ) 

        return x 

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=1) 
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): 
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype) 
        x = stem(x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x3 = self.layer3(x) 
        x4 = self.layer4(x3) 
        xproj = self.attnpool(x4) 

        return x3, x4, xproj 


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=0.1, drop_out=0., peft=False):
        super().__init__()
        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = MultiheadAttention(d_model, n_head, config=GLOBAL_CONFIG)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.n_head = n_head
        # if peft:
        if GLOBAL_CONFIG.MODEL.MLP_ADAPTER:
            self.MLP_Adapter_RGB = Adapter(d_model, GLOBAL_CONFIG.MODEL.MLP_ADP_MID_DIM, drop_out, scale, learnable_scalar=GLOBAL_CONFIG.MODEL.LEARNABLE_SCALAR)
            self.MLP_Adapter_NIR = Adapter(d_model, GLOBAL_CONFIG.MODEL.MLP_ADP_MID_DIM, drop_out, scale, learnable_scalar=GLOBAL_CONFIG.MODEL.LEARNABLE_SCALAR)
            if GLOBAL_CONFIG.DATASETS.NAMES != 'RGBN300':
                self.MLP_Adapter_TIR = Adapter(d_model, GLOBAL_CONFIG.MODEL.MLP_ADP_MID_DIM, drop_out, scale, learnable_scalar=GLOBAL_CONFIG.MODEL.LEARNABLE_SCALAR)
        if GLOBAL_CONFIG.MODEL.ATT_ADAPTER == True and GLOBAL_CONFIG.MODEL.ATT_PREFIX == 'lora_qk':
            self.Q_Adapter = QKVAdapter(d_model, GLOBAL_CONFIG.MODEL.ATT_MID_DIM, drop_out)
            self.K_Adapter = QKVAdapter(d_model, GLOBAL_CONFIG.MODEL.ATT_MID_DIM, drop_out)
            self.V_Adapter = QKVAdapter(d_model, GLOBAL_CONFIG.MODEL.ATT_MID_DIM, drop_out)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_lora=None, k_lora=None, v_lora=None):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, need_weights=False, attn_mask=self.attn_mask, 
                         q_lora=q_lora, k_lora=k_lora, v_lora=v_lora)[0]

    def forward(self, x: torch.Tensor, layer_idx=None, peft_stage=True):
        ## x shape [HW+1, BT, D]
        n, bs, d = x.shape

        if layer_idx is None or layer_idx < 0 or not GLOBAL_CONFIG.MODEL.MLP_ADAPTER or not peft_stage:
            xn = self.ln_1(x)
            x = x + self.attention(xn, xn, xn)
            x = x + self.mlp(self.ln_2(x))
        else:
            if GLOBAL_CONFIG.MODEL.ATT_ADAPTER == True and GLOBAL_CONFIG.MODEL.ATT_PREFIX == 'lora_qk':
                x_q = self.Q_Adapter(x)
                x_k = self.K_Adapter(x)
                # x_v = self.V_Adapter(x)
                x_v = None
            else:
                x_q = None
                x_k = None
                x_v = None

            xn = self.ln_1(x)
            x = x + self.attention(xn, xn, xn, x_q, x_k, x_v)

            x_mlp = self.mlp(self.ln_2(x))

            if GLOBAL_CONFIG.DATASETS.NAMES != 'RGBN300':
                x_mlp_rgb, x_mlp_nir, x_mlp_tir = x_mlp.chunk(3, dim=1)
                x_rgb, x_nir, x_tir = x.chunk(3, dim=1)

                x_rgb = self.MLP_Adapter_RGB(x_rgb, x_mlp_rgb) + x_rgb
                x_nir = self.MLP_Adapter_NIR(x_nir, x_mlp_nir) + x_nir
                x_tir = self.MLP_Adapter_TIR(x_tir, x_mlp_tir) + x_tir
                
                x = torch.cat([x_rgb, x_nir, x_tir], dim=1)
            else:
                x_mlp_rgb, x_mlp_nir = x_mlp.chunk(2, dim=1)
                x_rgb, x_nir = x.chunk(2, dim=1)

                x_rgb = self.MLP_Adapter_RGB(x_rgb, x_mlp_rgb) + x_rgb
                x_nir = self.MLP_Adapter_NIR(x_nir, x_mlp_nir) + x_nir

                x = torch.cat([x_rgb, x_nir], dim=1)

            # x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, scale=1., drop_out=0.1, peft=False):
        super().__init__()
        self.width = width
        self.layers = layers
        # dpr = [x.item() for x in torch.linspace(0, drop_out, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, drop_out, peft) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class Adapter(nn.Module):
    def __init__(self, in_dim, mid_dim, dropout=0.0, s=0.1, learnable_scalar=False):
        super().__init__()
        # down --> non linear --> up 
        self.down_proj = nn.Linear(in_dim, mid_dim)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(mid_dim, in_dim) 
        self.dropout = nn.Dropout(dropout)

        if learnable_scalar:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(s)

        # initialization 
        nn.init.kaiming_uniform_(self.down_proj.weight) 
        nn.init.zeros_(self.up_proj.weight) 
        nn.init.zeros_(self.down_proj.bias) 
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x, o_x):
        # x is (BT, HW+1, D)
        down = self.down_proj(x) 
        down = self.act(down) 
        down = self.dropout(down) 
        up = self.up_proj(down)
        output = o_x + up * self.scale
        return output

class QKVAdapter(nn.Module):
    def __init__(self, in_dim, mid_dim, dropout=0.0):
        super().__init__()
        # down --> non linear --> up 
        self.down_proj = nn.Linear(in_dim, mid_dim)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(mid_dim, in_dim) 
        self.dropout = nn.Dropout(dropout) 

        # initialization 
        nn.init.kaiming_uniform_(self.down_proj.weight) 
        nn.init.zeros_(self.up_proj.weight) 
        nn.init.zeros_(self.down_proj.bias) 
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        down = self.down_proj(x)
        down = self.act(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        return up

class VisionTransformer(nn.Module):
    def __init__(self, h_resolution: int, w_resolution: int, patch_size: int, stride_size: int, width: int, layers: int, heads: int, output_dim: int, adapter_scale=0.1):
        super().__init__()
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution*w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, scale=GLOBAL_CONFIG.MODEL.MLP_ADP_SCALE, peft=True)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim)) # 线性投影层

    def forward(self, x: torch.Tensor, cv_emb = None, peft_stage=True):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if cv_emb != None: 
            x[:,0] = x[:,0] + cv_emb
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(11):
            x = self.transformer.resblocks[i](x, i+1, peft_stage) 
        x11 = x
        x12 = self.transformer.resblocks[11](x11, 12, peft_stage)

        x11 = x11.permute(1, 0, 2)  # LND -> NLD  
        x12 = x12.permute(1, 0, 2)  # LND -> NLD  

        x12 = self.ln_post(x12)  

        if self.proj is not None:
            xproj = x12 @ self.proj

        return x11, x12, xproj # [bs, 129, 768] [bs, 129, 768] [bs, 129, 512]


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 vision_stride_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 h_resolution: int, 
                 w_resolution: int, adapter_scale=0.1
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=h_resolution*w_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                h_resolution = h_resolution,
                w_resolution = w_resolution,
                patch_size = vision_patch_size,
                stride_size = vision_stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim, adapter_scale=adapter_scale
            )
            
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text): 
        x = self.token_embedding(text).type(self.dtype)  

        x = x + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype) 

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection 

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp16)


def build_model(config, state_dict: dict, h_resolution: int, w_resolution: int, vision_stride_size: int, adapter_scale=0.1):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = config
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else: #RN50
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0] #77 (77,512)
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size, vision_stride_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        h_resolution, w_resolution, adapter_scale=adapter_scale
    )
    if vit:
        state_dict["visual.positional_embedding"] = resize_pos_embed(state_dict["visual.positional_embedding"], model.visual.positional_embedding, h_resolution, w_resolution)
    else: #RN50
        state_dict["visual.attnpool.positional_embedding"] = resize_pos_embed(state_dict["visual.attnpool.positional_embedding"], model.visual.attnpool.positional_embedding, h_resolution, w_resolution)
    
    
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
            
    convert_weights(model)

    model.load_state_dict(state_dict, strict=False)
    return model.eval()

import math
def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
      
    ntok_new = posemb_new.shape[0] #129,2048

    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid))) #14
    logger.info('Position embedding resize to height:{} width: {}'.format(hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2) 
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear') 
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze()], dim=0)
    return posemb