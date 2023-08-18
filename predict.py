

from cog import BasePredictor, Input, ConcatenateIterator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from stream_search import stream_search
from datetime import datetime
from threading import Thread

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "vonjack/Qwen-LLaMAfied-HFTok-7B-Chat",
            use_cache="cache"
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = AutoModelForCausalLM.from_pretrained(
            "vonjack/Qwen-LLaMAfied-HFTok-7B-Chat",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            use_cache="cache"
        ).to(self.device)

        self.model = model


    def predict(
        self,
        prompt: str = Input(description="prompt", default="Can ducks fly?"),
        max_new_tokens: int = Input(description="max_new_tokens", default=1000),
        temperature: float = Input(description="temperature", default=0.9),
        seed: int = Input(description="random number seed, -1=generate", default=-1),
        repetition_penalty: float = Input(description="repetition_penalty", default=1.1),
    ) -> ConcatenateIterator[str]:
        trimmed_prompt = prompt.strip()
        inputs = self.tokenizer.encode(trimmed_prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer)
        if seed == -1:
            seed = int(datetime.now().timestamp())
        print("seed", seed)

        torch.manual_seed(seed)
        generation_kwargs = dict(inputs=inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                                 repetition_penalty=repetition_penalty, streamer=streamer, do_sample=True)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        counter = len(trimmed_prompt)
        prompt_still_running = True
        for new_text in stream_search(['<s>','</s>'],streamer):
            if prompt_still_running:
                counter -= len(new_text)
                if counter <= 0:
                    if counter < 0:
                        ## remove extra spaces which llm created after the prompt
                        yield new_text[counter:].lstrip()
                    prompt_still_running=False
            else:
                yield new_text
