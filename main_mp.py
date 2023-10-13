import imageio

import torch
import types
import torch.multiprocessing as mp


from diffusers import StableDiffusionPipeline
from paradigms.stablediffusion_paradigms_mp import paradigms_forward, paradigms_forward_worker
from paradigms.paraddpm_scheduler import ParaDDPMScheduler
from paradigms.paraddim_scheduler import ParaDDIMScheduler
from paradigms.paradpmsolver_scheduler import ParaDPMSolverMultistepScheduler


def run(rank, total_ranks, queues):

    model_str = 'stabilityai/stable-diffusion-2'
    prompts = ["beautiful castle, matte painting"]

    # scheduler = ParaDDPMScheduler.from_pretrained(model_str, subfolder="scheduler", timestep_spacing="trailing")
    # scheduler._is_ode_scheduler = False
    # num_inference_steps = 1000
    # tolerance = 1e-1
    # parallel = 48

    # scheduler = ParaDPMSolverMultistepScheduler.from_pretrained(model_str, subfolder="scheduler", timestep_spacing="trailing")
    # scheduler._is_ode_scheduler = True
    # num_inference_steps = 200
    # tolerance = 1e-1
    # parallel = 32

    scheduler = ParaDDIMScheduler.from_pretrained(model_str, subfolder="scheduler", timestep_spacing="trailing")
    scheduler._is_ode_scheduler = True
    num_inference_steps = 200
    tolerance = 1e-1
    parallel = 32

    num_consumers = total_ranks


    pipe = StableDiffusionPipeline.from_pretrained(
        model_str, scheduler=scheduler, torch_dtype=torch.float16
    )
    pipe.unet.eval()
    pipe.enable_xformers_memory_efficient_attention()
    if rank != -1:
        pipe = pipe.to(f"cuda:{rank}")
        pipe.paradigms_forward = types.MethodType(paradigms_forward_worker, pipe)
        pipe.paradigms_forward(mp_queues=queues, device=f"cuda:{rank}")
    else:
        pipe = pipe.to(f"cuda:0")
        pipe.paradigms_forward = types.MethodType(paradigms_forward, pipe)

        # warmup
        _, _ = pipe.paradigms_forward(prompts, parallel=num_consumers, num_inference_steps=5*num_consumers, num_images_per_prompt=1, tolerance=tolerance, full_return=False, mp_queues=queues, device=f"cuda:0", num_consumers=num_consumers)

        output, stats = pipe.paradigms_forward(prompts, parallel=parallel, num_inference_steps=num_inference_steps, num_images_per_prompt=1, tolerance=tolerance, full_return=False, mp_queues=queues, device=f"cuda:0", num_consumers=num_consumers)

        images = output.images
        images[0].save(f"image_{num_inference_steps}_{tolerance}.png")
        print("Prompt (type \"exit\" to quit): ", end='', flush=True)

        while (user_input := queues[2].get()) is not None:
            output, stats = pipe.paradigms_forward([user_input], parallel=parallel, num_inference_steps=num_inference_steps, num_images_per_prompt=1, tolerance=tolerance, full_return=False, mp_queues=queues, device=f"cuda:0", num_consumers=num_consumers)
                
            images = output.images
            images[0].save(f"image_{num_inference_steps}_{tolerance}.png")
            print("Prompt (type \"exit\" to quit): ", end='', flush=True)

        # shutdown workers
        for _ in range(total_ranks):
            queues[0].put(None)

def main():
    torch.autograd.set_detect_anomaly(True)
    mp.set_start_method('spawn', force=True)
    queues = mp.Queue(), mp.Queue(), mp.Queue()

    processes = []
    num_processes = torch.cuda.device_count()

    for rank in range(-1, num_processes):
        p = mp.Process(target=run, args=(rank, num_processes, queues))
        p.start()
        processes.append(p)

    while (user_input := input()) != 'exit':
        queues[2].put(user_input)
    queues[2].put(None)

    for p in processes:
        p.join()    # wait for all subprocesses to finish

if __name__ == "__main__":
    main()