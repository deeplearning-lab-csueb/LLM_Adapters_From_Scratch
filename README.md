# LLM_Adapters_From_Scratch

1. Use Python 3.10.14 by `conda create --name LLM_Adapters python=3.10.14` 
2. Activate by `conda activate LLM_Adapters`
3. Install required python libraries:
    ```
    peft
    transformers
    bitsandbytes
    datasets
    fire
    evaluate
    scikit-learn
    ```
4. Install pytorch: `pip install torch torchvision torchaudio`

References:
1. https://huggingface.co/docs/transformers/en/model_doc/llama2

## Problems
On donny_evaluate2.py
```
/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:601: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.1` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:606: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.75` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:623: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `40` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:601: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.1` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:606: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.75` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:623: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `40` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
From v4.47 onwards, when a model cache is to be returned, `generate` will return a `Cache` instance instead by default (as opposed to the legacy tuple of tuples format). If you want to keep returning the legacy format, please set `return_legacy_cache=True`.

The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
```

new evaluate
```
Traceback (most recent call last):
  File "/media/dongping/new_space/donny/LLM_Adapters_From_Scratch/donny_evaluate_new.py", line 81, in <module>
    fire.Fire(main)
  File "/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/fire/core.py", line 135, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
  File "/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/fire/core.py", line 468, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
  File "/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/fire/core.py", line 684, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
  File "/media/dongping/new_space/donny/LLM_Adapters_From_Scratch/donny_evaluate_new.py", line 44, in main
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
  File "/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/datasets/arrow_dataset.py", line 560, in wrapper
    out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
  File "/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/datasets/arrow_dataset.py", line 3035, in map
    for rank, done, content in Dataset._map_single(**dataset_kwargs):
  File "/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/datasets/arrow_dataset.py", line 3461, in _map_single
    writer.write_batch(batch)
  File "/home/dongping/miniconda3/envs/LLM_Adapters/lib/python3.10/site-packages/datasets/arrow_writer.py", line 566, in write_batch
    pa_table = pa.Table.from_arrays(arrays, schema=schema)
  File "pyarrow/table.pxi", line 4740, in pyarrow.lib.Table.from_arrays
  File "pyarrow/table.pxi", line 4092, in pyarrow.lib.Table.validate
  File "pyarrow/error.pxi", line 92, in pyarrow.lib.check_status
pyarrow.lib.ArrowInvalid: Column 4 named input_ids expected length 1000 but got length 1
Script completed
Total runtime: -1728018508 s


```
