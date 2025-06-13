from pydantic import BaseModel
from transformers import pipeline
import os

import ogpu.service


import warnings
warnings.simplefilter("ignore", category=FutureWarning)


PIPE_ARGS_STR = os.getenv("PIPE_ARGS_STR")

PIPE_ARGS = eval(PIPE_ARGS_STR)

pipe = None

@ogpu.service.init()
def lifespan():
    global pipe
    ogpu.service.logger.info("Loading pipeline...")
    pipe = pipeline(**PIPE_ARGS)
    ogpu.service.logger.info("Pipeline loaded.")



class Inference(BaseModel):
    input: str | dict
    args : dict  = {}


class InferenceResult(BaseModel):
    result: dict


def convert_result_to_dict(result) -> dict:
    """Convert inference result to dict format"""
    if not isinstance(result, dict):
        if hasattr(result, '__dict__'):
            return result.__dict__
        elif hasattr(result, 'to_dict'):
            return result.to_dict()
        else:
            # For lists or other types, wrap in a dict
            return {"output": result}
    else:
        return result

@ogpu.service.expose()
def inference(inference: Inference) -> InferenceResult:
    try:
        ogpu.service.logger.info(f"Running inference...")
        ogpu.service.logger.info(f"Input: {inference.input}, Args: {inference.args}")
        result = pipe(inference.input, **inference.args)
        ogpu.service.logger.info(f"Inference result: {result}")
        
        result_dict = convert_result_to_dict(result)
        
        return InferenceResult(result=result_dict)
    except Exception as e:
        ogpu.service.logger.error(f"Error during inference: {e}")

ogpu.service.start()