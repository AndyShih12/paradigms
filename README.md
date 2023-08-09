# ParaDiGMS: Parallel Diffusion Generative Model Sampler

This repository contains code for the paper:

[Parallel Sampling of Diffusion Models](https://arxiv.org/abs/2305.16317) \
by Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, Nima Anari

-----
## Update

ParaDiGMS has been integrated into Huggingface Diffusers! 🥳🎉
```
pip install diffusers==0.19.3
```

```python
import torch
from diffusers import DDPMParallelScheduler
from diffusers import StableDiffusionParadigmsPipeline

scheduler = DDPMParallelScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", timestep_spacing="trailing")
pipe = StableDiffusionParadigmsPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", scheduler=scheduler, torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
ngpu, batch_per_device = torch.cuda.device_count(), 5
pipe.wrapped_unet = torch.nn.DataParallel(pipe.unet, device_ids=[d for d in range(ngpu)])
prompt = "a photo of an astronaut riding a horse on mars"

image = pipe(prompt, parallel=ngpu * batch_per_device, num_inference_steps=1000).images[0]
image.save("image.png")
```
-----

ParaDiGMs accelerates sampling of diffusion models without sacrificing quality by running denoising steps in parallel. ParaDiGMs is most useful when sampling with a large number of denoising steps on multiple GPUs, giving a 2-4x wallclock speedup.

![](imgs/paradigms.gif)

## Animation of Algorithm
![](imgs/method.gif)


## Sample Images
![](imgs/sample_images.png)

## Speedup of 1000-step DDPM on A100
![](imgs/paraddpm.png)

## Results on Image and Robotics Diffusion Models
![](imgs/results_table.png)

## Citation

If you find our work useful, consider citing:

```
@misc{shih2023paradigms,
    title = {Parallel Sampling of Diffusion Models},
    author={Shih, Andy and Belkhale, Suneel and Ermon, Stefano and Sadigh, Dorsa and Anari, Nima},
    publisher = {arXiv},
    year = {2023},
}
```
