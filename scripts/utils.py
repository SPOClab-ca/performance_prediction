import wandb 
from typing import Optional, Dict  
from pathlib import Path 


def init_or_resume_wandb_run(wandb_id_file_path: Path,
                             project_name: Optional[str] = None,
                             run_name: Optional[str] = None,
                             config: Optional[Dict] = None):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file. 
        
        Returns the config, if it's not None it will also update it first
        
        NOTE:
            Make sure that wandb_id_file_path.parent exists before calling this function
    """
    # if the run_id was previously saved, resume from there
    if wandb_id_file_path.exists():
        resume_id = wandb_id_file_path.read_text()
        wandb.init(project=project_name,
                   name=run_name,
                   resume=resume_id,
                   config=config)
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        run = wandb.init(project=project_name, name=run_name, config=config)
        wandb_id_file_path.write_text(str(run.id))

    wandb_config = wandb.config
    if config is not None:
        # update the current passed in config with the wandb_config
        config.update(wandb_config)

    return config


import time
from functools import wraps

def timed_func(foo):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        results = foo(*args, **kwargs)
        print ("{} done in {:.2f} seconds.".format(foo.__name__, time.time() - start_time))
        return results
    return wrapper