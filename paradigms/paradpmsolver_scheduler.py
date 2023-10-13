from typing import List, Tuple, Union
import torch

from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput

class ParaDPMSolverMultistepScheduler(DPMSolverMultistepScheduler):
    # careful when overriding __init__ function, can break things due to expected_keys parameter in configuration_utils
    # if necessary copy the whole init statement from parent class
    
    def _get_variance(self, timestep, prev_timestep=None):
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def batch_step_no_noise(
        self,
        model_output: torch.FloatTensor,
        timesteps: List[int],
        sample: torch.FloatTensor,
        generator: torch.Generator = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the multistep DPM-Solver.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        self.lambda_t = self.lambda_t.to(model_output.device)
        self.alpha_t = self.alpha_t.to(model_output.device)
        self.sigma_t = self.sigma_t.to(model_output.device)

        t = timesteps
        matches = (self.timesteps[None, :] == t[:, None])
        edgecases = ~matches.any(dim=1)
        step_index = torch.argmax(matches.int(), dim=1)
        step_index[edgecases] = len(self.timesteps) - 1 # if no match, then set to len(self.timesteps) - 1
        edgecases = (step_index == len(self.timesteps) - 1)

        prev_t = self.timesteps[ torch.clip(step_index+1, max=len(self.timesteps) - 1) ]
        prev_t[edgecases] = 0

        t = t.view(-1, *([1]*(model_output.ndim - 1)))
        prev_t = prev_t.view(-1, *([1]*(model_output.ndim - 1)))

        model_output = self.convert_model_output(model_output, t, sample)
        model_output = model_output.clamp(-1, 1) # important


        if self.config.solver_order == 1 or len(t) == 1:
            prev_sample = self.dpm_solver_first_order_update(model_output, t, prev_t, sample)
        elif self.config.solver_order == 2 or len(t) == 2:
            # first element in batch must do first_order_update
            prev_sample1 = self.dpm_solver_first_order_update(model_output[:1], t[:1], prev_t[:1], sample[:1])

            model_outputs_list = [model_output[:-1], model_output[1:]]
            timestep_list = [t[:-1], t[1:]]
            prev_sample2 = self.multistep_dpm_solver_second_order_update(
                model_outputs_list, timestep_list, prev_t[1:], sample[1:]
            )

            prev_sample = torch.cat([prev_sample1, prev_sample2], dim=0)
        else:
            # first element in batch must do first_order_update
            prev_sample1 = self.dpm_solver_first_order_update(model_output[:1], t[:1], prev_t[:1], sample[:1])

            # second element in batch must do second_order update
            model_outputs_list = [model_output[:1], model_output[1:2]]
            timestep_list = [t[:1], t[1:2]]
            prev_sample2 = self.multistep_dpm_solver_second_order_update(
                model_outputs_list, timestep_list, prev_t[1:2], sample[1:2]
            )

            model_outputs_list = [model_output[:-2], model_output[1:-1], model_output[2:]]
            timestep_list = [t[:-2], t[1:-1], t[2:]]
            prev_sample3 = self.multistep_dpm_solver_third_order_update(
                model_outputs_list, timestep_list, prev_t[2:], sample[2:]
            )

            prev_sample = torch.cat([prev_sample1, prev_sample2, prev_sample3], dim=0)

        # doing this otherwise set_timesteps throws an error
        # if worried about efficiency, can override the set_timesteps function
        self.lambda_t = self.lambda_t.to('cpu')

        return prev_sample