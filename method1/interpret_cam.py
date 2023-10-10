
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization
import cv2


def show_image_relevance(image_relevance, image, orig_image):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image)
    axs[0].axis('off')

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs[1].imshow(vis)
    axs[1].axis('off')
    
    plt.savefig('grad_cam.png')


def interpret(image, texts, model, target_index, device, start_layer):  # 一张图片和一个text的
    batch_size = texts.shape[0]
    # images = image.repeat(batch_size, 1, 1, 1)
    
    output = model(image)  # 197，512  如果乘了prompt的话，就变成了197，1006  这个应该是有梯度才对的！
    output = output / output.norm(dim=-1, keepdim=True)
    output_g = output[:, 0] @ texts[0].t()  # [32, 200]
    output_l = torch.topk(output[:, 1:] @ texts[0].t(),k=18, dim=1)[0].mean(dim=1)  # [32, 200]
    logits = (output_g + output_l) * 20
    b, c = logits.shape
    logits = logits[:, :c//2]  # 选择pos prompt进行查看
    one_hot = np.zeros((logits.shape[0], logits.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits.shape[0]), target_index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits)
    model.zero_grad()

    image_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer == -1: 
      # calculate index of last layer 
      start_layer = len(image_attn_blocks) - 1
    
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)  # 这里用的是mean
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    return image_relevance  # [1, 196]