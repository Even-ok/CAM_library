import argparse
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
from PIL import Image
from method1.interpret_cam import interpret, show_image_relevance
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

parser = argparse.ArgumentParser()
parser.add_argument("--image-path",             type=str,   default=None    )
parser.add_argument("--use-cuda",         type=bool,   default=False,    )
# parser.add_argument("--context-length",         type=int,   default=77,     )
# parser.add_argument("--vocab-size",             type=int,   default=49408,  )
# parser.add_argument("--transformer-width",      type=int,   default=512,    )
# parser.add_argument("--transformer-heads",      type=int,   default=8,      )
# parser.add_argument("--transformer-layers",     type=int,   default=12,     )
# parser.add_argument("--topk",                   type=int,   default=18      )

default_transform = transforms.Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


if __name__ == '__main__':
     
    args = parser.parse_args()
    
    test_transform = transforms.Compose([   # 请根据实际修改
        # transforms.CenterCrop(img_size),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    # 获取图像特征
    img = Image.open(args.image_path)
    transform = test_transform  # Or None, if none, it will adopt default_transform.
    input_tensor = test_transform(img).unsqueeze(0)
    input_tensor.requires_grad = True
    if args.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    input_tensor = input_tensor.to(device)
        
    # 获取prompt特征
    text_embedding = torch.load('text_features.pt', map_location=device)
    text_features = []
        # text = clip.tokenize(["a photo of "]).to(device)
    # with torch.no_grad():
    #     text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 构建模型
    cfg = setup_cfg(args)

    test_split = cfg.DATASET.TEST_SPLIT
    test_gzsi_dataset = build_dataset(cfg, test_split, cfg.DATASET.ZS_TEST)
    test_gzsl_split =  cfg.DATASET.TEST_GZSL_SPLIT
    test_unseen_dataset = build_dataset(cfg, test_gzsl_split, cfg.DATASET.ZS_TEST_UNSEEN)


    image_path = "/home/liangyiwen/dataset/nus_wide/images/dog/0591_760705044.jpg"
    category_index = 815

    classnames = test_gzsi_dataset.classnames

    model, arch_name = build_model(cfg, args, classnames)
    
    # 获取注意力 
    print("Doing Gradient Attention Rollout")
    R_image = interpret(model=model.image_encoder, image=input_tensor, texts=text_features[None, :, :], 
                        target_index = category_index, device=torch.device("cuda"), start_layer=-2)
    
    print(R_image.shape)
    show_image_relevance(R_image[0], input_tensor, orig_image=Image.open(image_path))