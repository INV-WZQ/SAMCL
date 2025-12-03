import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.modeling import Sam
from utils import generate_heatmap


class _LoRA_qkv(nn.Module):
    def __init__(self, r, attn: nn.Module, A, dim_in=384, dim_out=384):
        super().__init__()
        self.r = r
        self.attn = attn
        self.A = A
        self.B = nn.Linear(r, dim_out)
        self.norm1 = nn.LayerNorm(r)
        self.norm2 = nn.LayerNorm(r)
        self.cnt = self.B.weight.numel() + self.B.bias.numel() 
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def forward(self, x, prompt_aug=None):
        attn = self.attn(x)  
        adapter_ = self.A(x)
        if prompt_aug is not None and adapter_.size(0)!=prompt_aug.size(0):
            prompt_aug_ = F.interpolate(prompt_aug, size=(14, 14), mode='nearest')
            x_expanded = torch.cat([prompt_aug_[i].unsqueeze(0).repeat(int(adapter_.size(0)/prompt_aug.size(0)), 1, 1, 1) for i in range(prompt_aug.size(0))], dim=0).permute(0,3,2,1)
        elif prompt_aug is not None:
            x_expanded = prompt_aug.permute(0,3,2,1)
        if prompt_aug is not None:
            adapter_ = self.norm1(adapter_) + self.norm2(x_expanded)
        else:
            adapter_ = self.norm1(adapter_)
        adapter_ = self.B(adapter_)
        return attn + adapter_

class Extra(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.cnt = 0
        self.layer = nn.Linear(r, r)
        self.norm = nn.LayerNorm(r)
        self.cnt = self.layer.weight.numel() + self.layer.bias.numel()

class Adapter_Sam(nn.Module):
    def __init__(self, sam_model: Sam, r=10, lora_layer=None, SAM2=False):
        super(Adapter_Sam, self).__init__()
        if SAM2:
            image_encoder = sam_model.image_encoder.trunk
            self.SAM2=True
        else:
            image_encoder = sam_model.image_encoder
            self.SAM2=False
        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(6, len(image_encoder.blocks)-2))  
        self.r = r
        self.cnt = 0
        for name, param in sam_model.named_parameters():
            param.requires_grad=False
        
        self.extra = Extra(r)
        self.A = nn.Linear(384, 10)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.A.bias)
        self.cnt += self.extra.cnt + self.A.weight.numel() + self.A.bias.numel()
        
        for t_layer_i, blk in enumerate(image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            if blk.attn.dim_out != 384:break
            blk.attn = _LoRA_qkv(r, blk.attn, self.A, dim_out=blk.attn.dim_out)
            self.cnt+= blk.attn.cnt
        
        self.sam = sam_model
        self.count_param()


    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        
        extra = {f"extra": self.extra.cpu()}
        lora_layer = {f"lora_layer": self.lora_layer}
        num_layer = len(self.lora_layer)
        A = {f"A": self.A.cpu()}
        B= {f"B_{i:03d}": self.sam.image_encoder.trunk.blocks[self.lora_layer[i]].attn.B.cpu() for i in range(num_layer)}
        norm1 = {f"norm1_{i:03d}": self.sam.image_encoder.trunk.blocks[self.lora_layer[i]].attn.norm1.cpu() for i in range(num_layer)}
        norm2 = {f"norm2_{i:03d}": self.sam.image_encoder.trunk.blocks[self.lora_layer[i]].attn.norm2.cpu() for i in range(num_layer)}
        
        merged_dict = {**extra, **lora_layer, **A, **B, **norm1, **norm2}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str, args) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename, weights_only=False)#, map_location='cuda:0')

        self.extra = state_dict[f'extra'].cpu().cuda()
        self.A = state_dict['A'].cpu().cuda()#to(f'cuda:{args.rank}')
        self.lora_layer = state_dict[f'lora_layer']
        i = 0
        for t_layer_i, blk in enumerate(self.sam.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            B = state_dict[f'B_{i:03d}']
            norm1 = state_dict[f'norm1_{i:03d}']
            norm2 = state_dict[f'norm2_{i:03d}']
            i+=1
            self.sam.image_encoder.trunk.blocks[t_layer_i].attn.A = self.A
            self.sam.image_encoder.trunk.blocks[t_layer_i].attn.B = B.cpu().cuda()#.to(f'cuda:{args.rank}')
            self.sam.image_encoder.trunk.blocks[t_layer_i].attn.norm1 = norm1.cpu().cuda()#.to(f'cuda:{args.rank}')
            self.sam.image_encoder.trunk.blocks[t_layer_i].attn.norm2 = norm2.cpu().cuda()#.to(f'cuda:{args.rank}')

    def count_param(self):
        print("USE Param: ", self.cnt, " Memory: ", self.cnt*32/8/1024/1024, "MB", "(Extra:", self.extra.cnt*32/8/1024/1024, "MB)")
    
    def forward_SAM(self, batched_input, multimask_output=False, image_size=1024, boxes=None, points=None, begin=-1, end=-1):
        if points is not None:
            prompt_heat = generate_heatmap((9, 9), points, 64, 64).cuda().float()
            prompt_heat = prompt_heat.repeat(1, self.r, 1, 1)
            prompt_heat = F.relu(self.extra.norm(self.extra.layer(prompt_heat.permute(0,2,3,1)))).permute(0,3,1,2)
        x = batched_input
        if begin==-1:
            input_images = self.sam.preprocess(batched_input)
            input_images = self.sam.image_encoder.patch_embed(input_images)
            if self.sam.image_encoder.pos_embed is not None:
                input_images = input_images + self.sam.image_encoder.pos_embed
            x = input_images

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points, boxes=boxes, masks=None
        )
        
        for t_layer_i, blk in enumerate(self.sam.image_encoder.blocks):
            if t_layer_i<begin:continue
            if t_layer_i in self.lora_layer:
                x = blk(x, prompt_heat)
            else:
                x = blk(x)
            if t_layer_i==end:
                embed = x.detach()
                embed = embed.permute(0, 3, 1, 2)
                embed = embed.detach().mean(-1).mean(-1).tolist()
                return x, embed
        prompt_heat_ = self.extra.upward(prompt_heat.permute(0,2,3,1)).permute(0,3,1,2)
        prompt_heat = self.extra.upward_(prompt_heat.permute(0,2,3,1)).permute(0,3,1,2)
        image_embeddings = self.sam.image_encoder.neck(x.permute(0, 3, 1, 2) + prompt_heat_) + prompt_heat

        image_pe = self.sam.prompt_encoder.get_dense_pe()
        low_res_masks, iou_predictions,_ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings, 
            multimask_output=multimask_output
        )
        
        masks = self.sam.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }
        return outputs

    def forward_SAM2(self, batched_input, multimask_output=False, image_size=1024, boxes=None, points=None, begin=-1, end=-1, mid_embed=None):
        if points is not None:
            prompt_heat = generate_heatmap((9, 9), points, 64, 64).cuda().float()
            prompt_heat = prompt_heat.repeat(1, self.r, 1, 1)
            prompt_heat = F.relu(self.extra.norm(self.extra.layer(prompt_heat.permute(0,2,3,1)))).permute(0,3,1,2)
        x = batched_input
        return self.sam(x, points=points, prompt_heat=prompt_heat, begin=begin, end=end, mid_embed=mid_embed)
    
    def forward(self, batched_input, multimask_output=False, image_size=1024, boxes=None, points=None, begin=-1, end=-1, mid_embed=None):
        if self.SAM2:
            return self.forward_SAM2(batched_input, multimask_output, image_size, boxes, points, begin, end, mid_embed=mid_embed)
        return self.forward_SAM(batched_input, multimask_output, image_size, boxes, points, begin, end)