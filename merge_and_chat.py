import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# -------------------------------------------------------------
# 1. MERGE ADAPTER DIRECTLY INTO BASE WEIGHTS (ON CPU)
# -------------------------------------------------------------
def merge_and_export_model(
    base_model_id: str = "Qwen/Qwen2.5-1.5B",
    adapter_dir: str = "./control_tuner_final_adapter",
    output_dir: str = "./control_tuner_standalone_model"
):
    """
    Bakes LoRA matrices B*A directly into base weights W0 on CPU.
    Produces a single standalone model folder with zero PEFT dependency.
    """
    if os.path.exists(output_dir) and (
        os.path.exists(os.path.join(output_dir, "model.safetensors")) or
        os.path.exists(os.path.join(output_dir, "model-00001-of-00002.safetensors"))
    ):
        print(f"Standalone model already exists at: {output_dir}. Skipping merge step.\n")
        return

    print("=" * 60)
    print("STEP 5: MERGING LORA WEIGHTS (merge_and_unload)")
    print("=" * 60)

    # 1. Load tokenizer
    print(f"[1/4] Loading tokenizer from: {adapter_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    # 2. Load unquantized base model in 16-bit on CPU (low_cpu_mem_usage=True keeps RAM under 4 GB)
    print(f"[2/4] Loading base model on CPU: {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float16,               # Modern transformers parameter
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    # 3. Attach LoRA adapter
    print(f"[3/4] Attaching LoRA adapter from: {adapter_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    # 4. Merge W_final = W_0 + (alpha/r) * (B * A) and discard adapter layers
    print(f"[4/4] Executing merge_and_unload()...")
    merged_model = peft_model.merge_and_unload()

    # 5. Save standalone model and tokenizer to disk
    print(f"Saving standalone model to: {output_dir}...")
    merged_model.save_pretrained(output_dir, max_shard_size="4GB")
    tokenizer.save_pretrained(output_dir)

    print(f"[SUCCESS] Standalone model saved cleanly to: {output_dir}!\n")


# -------------------------------------------------------------
# 2. INTERACTIVE TERMINAL CHAT SESSION (ON GPU)
# -------------------------------------------------------------
def interactive_control_chat(model_dir: str = "./control_tuner_standalone_model"):
    """
    Loads your merged model onto your RTX 2060 and starts a live terminal chat.
    Notice: No PEFT or quantization config needed here!
    """
    print("=" * 60)
    print("STARTING LIVE CHAT WITH CONTROLTUNER (Type 'exit' to quit)")
    print("=" * 60)

    print("Loading standalone model onto RTX 2060 GPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,               # Clean 16-bit float
        device_map="auto"
    )
    model.eval()

    # Setup explicit stop tokens (<|endoftext|> and <|im_end|>)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [tokenizer.eos_token_id, im_end_id]

    print("\nControlTuner is online! Ask any control engineering question.")
    print("Example: 'Given A=[[0, 1], [-2, -3]] and B=[[0], [1]], is it controllable?'\n")

    while True:
        try:
            prompt = input("\nYou > ").strip()
            if not prompt:
                continue

            if prompt.lower() in ["exit", "quit", "q"]:
                print("Exiting chat. Goodbye!")
                break

            messages = [
                {"role": "system", "content": "You are ControlTuner, a precision control systems design assistant."},
                {"role": "user", "content": prompt}
            ]

            # Format in ChatML and extract dictionary tensors
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    eos_token_id=stop_ids,
                    pad_token_id=tokenizer.eos_token_id
                )

            prompt_len = inputs["input_ids"].shape[1]
            response_tokens = output_ids[0][prompt_len:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=False)

            # Strip end-of-turn tokens
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
            if "<|im_start|>" in response:
                response = response.split("<|im_start|>")[0]

            print(f"\nControlTuner >\n{response.strip()}")

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break


if __name__ == "__main__":
    # 1. Merge and save the standalone model
    merge_and_export_model()

    # 2. Launch interactive live chat
    interactive_control_chat()