import torch
from sam import Sam
from image_encoder import ImageEncoderViT
from mask_decoder import MaskDecoder
from prompt_encoder import PromptEncoder
import cv2
from functools import partial
from transformer import TwoWayTransformer

prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size
encoder_embed_dim = 768
encoder_depth = 2
encoder_num_heads = 12
encoder_global_attn_indexes = ()

sam = Sam(
    image_encoder=ImageEncoderViT(
        depth=2,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    ),
    prompt_encoder=PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    ),
    mask_decoder=MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    ),
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
)

image = cv2.imread('../../demo/src/assets/data/dogs.jpg')
image = torch.tensor(image)
image = image.permute(2,0,1)
c,h,w = image.shape
image = image.to('cuda')
data = list()
dic = dict()
dic['image'] = image
dic['original_size'] = (h,w)
data.append(dic)
data.append(dic)
model = sam.to('cuda')
print(model(data, False))