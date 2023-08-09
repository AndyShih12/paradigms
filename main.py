import os
import torch
import pandas as pd
import types
from collections import defaultdict

from diffusers import DDIMScheduler
from diffusers import DDPMScheduler
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline

from paradigms.paraddpm_scheduler import ParaDDPMScheduler
from paradigms.paraddim_scheduler import ParaDDIMScheduler
from paradigms.paradpmsolver_scheduler import ParaDPMSolverMultistepScheduler
from paradigms.stablediffusion_paradigms import paradigms_forward

TOPIC = "stablediffusion"
MODEL_ID = "stabilityai/stable-diffusion-2"
HOME_DIR = "./"

SCHEDULER_CONFIGS = [
    # class, num_inference_steps, fname, method, is_parallel, is_ode
    # [DDIMScheduler, 200, f"{HOME_DIR}/imgs/{TOPIC}/warmup%s.png", "warmup", False, True],
    # [DDPMScheduler, 1000, f"{HOME_DIR}/imgs/{TOPIC}/ddpm%s.png", "ddpm", False, False],
    # [DDIMScheduler, 200, f"{HOME_DIR}/imgs/{TOPIC}/ddim%s.png", "ddim", False, True],
    # [DPMSolverMultistepScheduler, 200, f"{HOME_DIR}/imgs/{TOPIC}/dpmsolver%s.png", "dpmsolver", False, True],
    [ParaDDIMScheduler, 200, f"{HOME_DIR}/imgs/{TOPIC}/parawarmup%s.png", "parawarmup", True, True],
    [ParaDDPMScheduler, 1000, f"{HOME_DIR}/imgs/{TOPIC}/paraddpm%s.png", "paraddpm", True, False],
    [ParaDDIMScheduler, 200, f"{HOME_DIR}/imgs/{TOPIC}/paraddim%s.png", "paraddim", True, True],
    [ParaDPMSolverMultistepScheduler, 200, f"{HOME_DIR}/imgs/{TOPIC}/paradpmsolver%s.png", "paradpmsolver", True, True],
]

def prepare_pipe(scfg):
    scheduler_cls, _, _, name, is_parallel, is_ode = scfg

    scheduler = scheduler_cls.from_pretrained(MODEL_ID, subfolder="scheduler", timestep_spacing="trailing")
    scheduler._is_ode_scheduler = is_ode
    scheduler._is_parallel = is_parallel
    scheduler.config.thresholding = False

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.paradigms_forward = types.MethodType(paradigms_forward, pipe)
    pipe.enable_xformers_memory_efficient_attention() 

    return pipe

def run_stable_diffusion(pipe, ngpu, parallel, num_inference_steps, prompts):
    with torch.inference_mode():
        if ngpu > 1:
            pipe.wrapped_unet = torch.nn.DataParallel(pipe.unet, device_ids=[d for d in range(ngpu)])
        else:
            pipe.wrapped_unet = pipe.unet

        stats = defaultdict(float)
        if pipe.scheduler._is_parallel:
            options = {
                'parallel': parallel*ngpu,
                'tolerance': 0.1,
            }
            output, stats = pipe.paradigms_forward(prompts, num_inference_steps=num_inference_steps, full_return=False, **options)
        else:
            output = pipe(prompts, num_inference_steps=num_inference_steps)

    return output, stats

def main():
    prompts = ["A beautiful castle, matte painting"]
    ngpu_sweep = [(i+1) for i in range(torch.cuda.device_count())]
    parallel_sweep = [1,5,10]

    # prepare a dict for storing the results
    time_results = {scfg[3] : torch.zeros(max(ngpu_sweep)+1, max(parallel_sweep)+1) for scfg in SCHEDULER_CONFIGS}
    pass_results = {scfg[3] : torch.zeros(max(ngpu_sweep)+1, max(parallel_sweep)+1) for scfg in SCHEDULER_CONFIGS}
    flops_results = {scfg[3] : torch.zeros(max(ngpu_sweep)+1, max(parallel_sweep)+1) for scfg in SCHEDULER_CONFIGS}
    pipes = {scfg[3]: prepare_pipe(scfg) for scfg in SCHEDULER_CONFIGS}

    for ngpu in ngpu_sweep:
        for parallel in parallel_sweep:
            for scfg in SCHEDULER_CONFIGS:
                scheduler_cls, num_inference_steps, fname, name, is_parallel, is_ode = scfg

                output, stats = run_stable_diffusion(pipes[name], ngpu, parallel, num_inference_steps, prompts)
                image_savepath = fname % ("_%u_%u" % (ngpu, parallel))
                os.makedirs(os.path.dirname(image_savepath), exist_ok=True)
                output.images[0].save(image_savepath)

                print(f"ngpu={ngpu}, parallel={parallel}, scheduler={name}, time={stats['time']}")

                # store the result tm in a dict with key (ngpu, parallel, scheduler)
                time_results[name][ngpu, parallel] = stats['time']
                pass_results[name][ngpu, parallel] = stats['pass_count']
                flops_results[name][ngpu, parallel] = stats['flops_count']

    # convert results to a dataframe
    stat_dfs = [time_results, pass_results, flops_results]
    stat_names = ['time', 'pass', 'flops']
    for stat_df, stat_name in zip(stat_dfs, stat_names):
        for scheduler_name in stat_df:
            df = pd.DataFrame(stat_df[scheduler_name].numpy())
            print(scheduler_name)
            print(df.to_string())

            df_savepath = f'{HOME_DIR}/stats/{TOPIC}/{stat_name}_{scheduler_name}.csv'
            os.makedirs(os.path.dirname(df_savepath), exist_ok=True)
            df.to_csv(df_savepath, index=True)


if __name__ == "__main__":
    main()
