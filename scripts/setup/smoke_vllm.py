"""Load the Search-R1 checkpoint with vLLM and check tag-structured generation."""

from vllm import LLM, SamplingParams

from scdn import paths
from scdn.gen_traces import PROMPT_TEMPLATE

QUESTION = (
    "Mike Barnett negotiated many contracts including which player that went on "
    "to become general manager of CSKA Moscow of the Kontinental Hockey League?"
)

llm = LLM(
    model=str(paths.SEARCH_R1_7B),
    dtype="bfloat16",
    max_model_len=16384,
    gpu_memory_utilization=0.85,
)
tok = llm.get_tokenizer()
prompt = PROMPT_TEMPLATE.format(question=QUESTION)
if tok.chat_template:
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )

sampling = SamplingParams(
    temperature=0.0,
    max_tokens=1024,
    stop=["</search>", "</answer>"],
    include_stop_str_in_output=True,
)
out = llm.generate([prompt], sampling)[0].outputs[0]
print("=" * 60)
print(out.text)
print("=" * 60)
print("finish_reason:", out.finish_reason, "| stop_reason:", out.stop_reason)
assert "<think>" in out.text, "missing <think> tag"
assert "<search>" in out.text or "<answer>" in out.text, "missing search/answer tag"
print("SMOKE OK")
