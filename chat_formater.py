from transformers import AutoTokenizer


def setup_tokenizer(model_name: str = "Qwen/Qwen2.5-1.5B"):
    """
    Loads the tokenizer and handles padding token configuration.
    """
    # TODO: Load tokenizer from pretrained
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # TODO: Check if pad_token is None, and if so, map it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
    return tokenizer
def format_conversation(tokenizer, messages: list, is_training: bool = True) -> str:
    """
    Formats multi-turn messages into a ChatML string.
    - If is_training=True: keeps the assistant target and omits generation prompt.
    - If is_training=False: strips trailing assistant turn and adds generation prompt.
    """
    
    if is_training:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        prompt_only = [msg for msg in messages if msg["role"] != "assistant"]
    return tokenizer.apply_chat_template(
        prompt_only,
        tokenize=False,
        add_generation_prompt=True
    )


def find_assistant_start_index(tokenizer, formatted_train_text: str) -> int:
    """
    Identifies the exact token index where the assistant's reply begins.
    Used for response-only loss masking (masking prompt tokens with -100).
    """
    full_tokens = tokenizer.encode(formatted_train_text, add_special_tokens=False)

    delimiter_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    del_len = len(delimiter_ids)

    for i in range (len(full_tokens) - del_len + 1):
        if full_tokens[i : i + del_len] == delimiter_ids:
            return i + del_len

    return -1

if __name__ == "__main__":
    # --- Verification Run ---
    print("=" * 60)
    print("STEP 1 VERIFICATION: CHAT TEMPLATES & TOKENIZATION")
    print("=" * 60)

    tokenizer = setup_tokenizer("Qwen/Qwen2.5-1.5B")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"Padding side: {tokenizer.padding_side}")

    sample_dialogue = [
        {
            "role": "system",
            "content": "You are ControlTuner, a precision control systems design assistant."
        },
        {
            "role": "user",
            "content": "What is the condition for a linear continuous-time system to be controllable?"
        },
        {
            "role": "assistant",
            "content": "A system (A, B) is controllable if the controllability matrix C = [B, AB, ..., A^(n-1)B] has full row rank (rank = n)."
        }
    ]

    # Training format check
    train_text = format_conversation(tokenizer, sample_dialogue, is_training=True)
    print("\n--- Training String Output ---")
    print(train_text)

    # Inference format check
    infer_text = format_conversation(tokenizer, sample_dialogue, is_training=False)
    print("--- Inference String Output ---")
    print(infer_text)

    # Token index boundary check
    start_idx = find_assistant_start_index(tokenizer, train_text)
    full_tokens = tokenizer.encode(train_text, add_special_tokens=False)
    print("--- Boundary Check ---")
    print(f"Assistant response starts at token index: {start_idx}")
    print(f"Decoded starting tokens: '{tokenizer.decode(full_tokens[start_idx : start_idx + 6])}'")