import torch

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


def get_openai_cifar(
    steps=100,
    model_path="res/openai-models/cifar10_uncond_50M_500K.pt",
    device="cuda",
):
    args = model_and_diffusion_defaults()
    # model flags
    args["image_size"] = 32
    args["num_channels"] = 128
    args["num_res_blocks"] = 3
    args["learn_sigma"] = True

    # diffusion flags
    args["diffusion_steps"] = 4000
    args["noise_schedule"] = "cosine"

    # sampling flags
    args["use_ddim"] = True
    args["timestep_respacing"] = f"ddim{steps}"

    # old models
    args["rescale_learned_sigmas"] = True
    args["rescale_timesteps"] = True

    model, diffusion = create_model_and_diffusion(**{k: args[k] for k in model_and_diffusion_defaults().keys()})
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model = model.eval()
    model = model.to(device)
    return model, diffusion, args


def get_openai_imagenet(
    steps=100,
    model_path="res/openai-models/imagenet64_uncond_100M_1500K.pt",
    device="cuda",
):
    args = model_and_diffusion_defaults()
    args["image_size"] = 64
    args["class_cond"] = False
    args["num_res_blocks"] = 3
    args["num_channels"] = 128
    args["learn_sigma"] = True
    args["noise_schedule"] = "cosine"
    args["use_ddim"] = True
    args["timestep_respacing"] = f"ddim{steps}"
    args["diffusion_steps"] = 4000
    args["rescale_learned_sigmas"] = True
    args["rescale_timesteps"] = True

    model, diffusion = create_model_and_diffusion(**{k: args[k] for k in model_and_diffusion_defaults().keys()})
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model = model.eval()
    model = model.to(device)
    return model, diffusion, args


def get_ls_cifar10(
    steps=100,
    model_path="<PROJECT_PATH>/res/openai_cifar10_checkpointed/model625000.pt",
    device="cuda",
    learn_sigma=True,
):
    args = model_and_diffusion_defaults()
    # model flags
    args["image_size"] = 32
    args["num_channels"] = 128
    args["num_res_blocks"] = 3
    args["learn_sigma"] = learn_sigma

    # diffusion flags
    args["diffusion_steps"] = 4000
    args["noise_schedule"] = "cosine"

    # sampling flags
    args["use_ddim"] = True
    args["timestep_respacing"] = f"ddim{steps}"

    # old models
    args["rescale_learned_sigmas"] = True
    args["rescale_timesteps"] = True

    model, diffusion = create_model_and_diffusion(**{k: args[k] for k in model_and_diffusion_defaults().keys()})
    model = model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.eval()
    return model, diffusion, args


def get_ls_imagenet(
    steps=100, model_path="res/imagenet_ckpt_multigpu/model1080000.pt", device="cuda", learn_sigma=True
):
    args = model_and_diffusion_defaults()
    args["image_size"] = 64
    args["num_channels"] = 128
    args["num_res_blocks"] = 3
    args["learn_sigma"] = learn_sigma

    args["diffusion_steps"] = 4000
    args["noise_schedule"] = "cosine"

    args["use_ddim"] = True
    args["timestep_respacing"] = f"ddim{steps}"

    args["rescale_learned_sigmas"] = True
    args["rescale_timesteps"] = True

    model, diffusion = create_model_and_diffusion(**{k: args[k] for k in model_and_diffusion_defaults().keys()})
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.eval()
    return model, diffusion, args


def get_ddpm_imagenet256(
    steps=100, model_path="res/openai-models/256x256_diffusion_uncond.pt", device="cuda", use_fp16=True
):
    """Guided Diffusion model for 256x256 ImageNet.
    ```
    --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True
    ```
    """

    args = model_and_diffusion_defaults()

    args["image_size"] = 256
    args["num_channels"] = 256
    args["num_head_channels"] = 64
    args["num_res_blocks"] = 2
    args["use_fp16"] = use_fp16
    args["use_scale_shift_norm"] = True
    args["resblock_updown"] = True
    args["attention_resolutions"] = "32,16,8"
    args["class_cond"] = False
    args["noise_schedule"] = "linear"
    args["diffusion_steps"] = 1000
    args["learn_sigma"] = True
    args["use_ddim"] = True
    args["timestep_respacing"] = f"ddim{steps}"
    arguments = {k: args[k] for k in model_and_diffusion_defaults().keys()}

    model, diffusion = create_model_and_diffusion(**arguments)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    if args["use_fp16"]:
        model.convert_to_fp16()
    model = model.eval()
    model = model.to(device)
    return model, diffusion, args
