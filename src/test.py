# import torch
# from PIL import Image
# from torchvision.io import ImageReadMode, read_image
# from torchvision.transforms import v2
# from transformers import AutoProcessor
#
# from modeling_git import GitForCausalLMClipEmb
#
# url = "/media/SSD_1_2T/xt/data/natural-scenes-dataset/webdataset_avg_split/test/subj01/sample000000349.jpg"
# image = read_image(url, mode = ImageReadMode.RGB)
# # image = Image.open(url)
#
# preproc = v2.Compose(
#     [
#         v2.Resize(size = 224, interpolation = v2.InterpolationMode.BICUBIC, antialias = None),
#         v2.ToDtype(torch.float32, scale = True),
#         v2.CenterCrop(size = (224, 224)),
#         v2.Normalize(mean = (0.48145466, 0.4578275, 0.40821073), std = (0.26862954, 0.26130258, 0.27577711))
#     ]
# )
# # processor = AutoProcessor.from_pretrained("microsoft/git-large-coco", cache_dir = "/media/SSD_1_2T/xt/weights/")
# # model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco", cache_dir = "/media/SSD_1_2T/xt/weights/")
# processor = AutoProcessor.from_pretrained("microsoft/git-base-coco", cache_dir = "/media/SSD_1_2T/xt/weights/")
# model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-base-coco", cache_dir = "/media/SSD_1_2T/xt/weights/")
# vision_model = model.git.image_encoder
#
# pixel_values = processor(images = image, return_tensors = "pt").pixel_values  # [1, 3, 224, 224]
# pixel_values_2 = preproc(image).unsqueeze(0)
#
# image_embedding = vision_model(pixel_values).last_hidden_state
# image_embedding_2 = vision_model(pixel_values_2).last_hidden_state
#
# print(pixel_values.shape, image_embedding.shape)
#
# generated_ids = model.generate(pixel_values = image_embedding, max_length = 50)
# generated_ids_2 = model.generate(pixel_values = image_embedding_2, max_length = 50)
# generated_caption = processor.batch_decode(generated_ids, skip_special_tokens = True)
# generated_caption_2 = processor.batch_decode(generated_ids_2, skip_special_tokens = True)
# """
# ['a man standing in front of a pile of carrots.']
# ['a large pile of carrots and potatoes at a farmers market.']
#
#
# ['two chefs in a kitchen preparing food.']
# ['two chefs in a kitchen preparing food.']
# """
# print(generated_caption)
# print(generated_caption_2)

# import torch
# from diffusers import DDPMScheduler, StableUnCLIPPipeline, UnCLIPScheduler, UnCLIPPipeline
# from diffusers.models import PriorTransformer
# from transformers import CLIPTextModelWithProjection, CLIPTokenizer
#
# device = "cuda:1"
# prompt = "dramatic wave, the Oceans roar, Strong wave spiral across the oceans as the waves unfurl into roaring crests; perfect wave form; perfect wave shape; dramatic wave shape; wave shape unbelievable; wave; wave shape spectacular"
#
# text_encoder = CLIPTextModelWithProjection.from_pretrained(
#     "openai/clip-vit-large-patch14", cache_dir = "/media/SSD_1_2T/xt/weights/"
# ).to(device)
# text_tokenizer = CLIPTokenizer.from_pretrained(
#     "openai/clip-vit-large-patch14", cache_dir = "/media/SSD_1_2T/xt/weights/"
# )
#
# text_inputs = text_tokenizer(
#     prompt,
#     padding = "max_length",
#     max_length = text_tokenizer.model_max_length,
#     truncation = True,
#     return_tensors = "pt",
# )
# input_ids = text_inputs.input_ids.to(device)
#
# outputs = text_encoder(input_ids)
# print(outputs[0].shape, outputs[1].shape)
#
# prior = PriorTransformer.from_pretrained("kakaobrain/karlo-v1-alpha", subfolder = "prior", torch_dtype = torch.float16)
# prior_scheduler = UnCLIPScheduler.from_pretrained("kakaobrain/karlo-v1-alpha", subfolder = "prior_scheduler")
# prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)
#
# from diffusers import UnCLIPPipeline
# import torch
#
# pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16, cache_dir = "/media/SSD_1_2T/xt/weights/")
# pipe = pipe.to('cuda:1')
#
# prompt = "a big red frog on a green leaf."
# image = pipe(prompt).images[0]
# image.save("./frog.png")


