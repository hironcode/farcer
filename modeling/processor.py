from transformers import pipeline
import torch

try:
    import util
except ImportError:
    import modeling.util


class ProcessorParams:
    # sttm: Speech-to-Text Model
    freeze_sttm: bool = True
    sttm_name: str = "openai/whisper-medium"
    sttm_batch_size: int = 16
    time_stamps: bool = True

    device = util.device
    torch_dtype = torch.float16 if device == 'cuda:0' else torch.float32


class Processor:
    def __init__(self, params: ProcessorParams):
        self.params = params
        self.sttm = pipeline(
            "automatic-speech-recognition",
            model=params.sttm_name,
            device=params.device,
            batch_size=params.sttm_batch_size,
            return_timestamps=params.time_stamps,
            torch_dtype=params.torch_dtype
        )