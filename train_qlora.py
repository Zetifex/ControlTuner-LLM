import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Import our helper functions from Step 1 and Step 2
from chat_formater import setup_tokenizer
from dataset_and_config import build_raw_control_records, create_hf_dataset_splits, get_sft_config


def load_qlora_model(model_id: str = "Qwen/Qwen2.5-1.5B"):
    """
    Loads base model in 4-bit NF4 precision and attaches LoRA adapters.
    Ensures all adapter modules are strictly in float32 for Turing (RTX 2060).
    """
    # 1. 4-bit NormalFloat quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2. Load model (passing float16 explicitly)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 3. Prepare model for 4-bit training
    model = prepare_model_for_kbit_training(model)

    # 4. Define LoRA parameters targeting attention and MLP projections
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # Attention projections
            "gate_proj", "up_proj", "down_proj"       # MLP projections
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 5. Wrap the base model with PEFT adapters
    model = get_peft_model(model, lora_config)

    # 6. CRITICAL FIX FOR RTX 2060:
    # Convert all LoRA modules to float32 at the Module level (properly updates dtype)
    for name, module in model.named_modules():
        if "lora_" in name:
            module.to(torch.float32)

    return model, lora_config


def print_trainable_parameters(model):
    """
    Prints the number and percentage of trainable LoRA parameters.
    """
    trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"Total Base Parameters:     {total_params:,}")
    print(f"Trainable LoRA Parameters: {trainable_params:,}")
    print(f"Trainable Percentage:      {100 * (trainable_params / total_params):.2f}%")


def run_training_pipeline():
    """
    Orchestrates the complete SFT training loop and saves the final adapter.
    """
    model_id = "Qwen/Qwen2.5-1.5B"

    # 1. Setup tokenizer
    print("\n[1/5] Loading tokenizer and base model in 4-bit...")
    tokenizer = setup_tokenizer(model_id)
    tokenizer.model_max_length = 512
    model, lora_config = load_qlora_model(model_id)

    # 2. Print parameter breakdown
    print("\n[2/5] Inspecting parameter counts...")
    print_trainable_parameters(model)

    # 3. Prepare dataset splits
    print("\n[3/5] Preparing dataset splits...")
    records = build_raw_control_records()
    splits = create_hf_dataset_splits(records)
    print(f"      Train samples: {len(splits['train'])} | Test samples: {len(splits['test'])}")

    # 4. Training configuration: Disable external GradScaler to prevent BFloat16 conflict
    training_args = get_sft_config()
    training_args.fp16 = False   # bitsandbytes handles 16-bit compute internally!
    training_args.bf16 = False

    # 5. Instantiate SFTTrainer
    print("\n[4/5] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        peft_config=None,                    # Model is already a PeftModel
        processing_class=tokenizer,
        args=training_args
    )

    # 6. Execute training
    print("\n[5/5] Launching training loop...")
    print("=" * 60)
    print("TRAINING STARTED (Watch loss drop over steps)")
    print("=" * 60)
    trainer.train()

    # 7. Save the trained LoRA adapter and tokenizer
    output_dir = "./control_tuner_final_adapter"
    print(f"\nTraining complete! Saving LoRA adapter to: {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[PASS] STEP 3 COMPLETE: Adapter saved successfully to {output_dir}!")


if __name__ == "__main__":
    run_training_pipeline()