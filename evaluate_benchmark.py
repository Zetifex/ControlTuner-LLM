import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel


# -------------------------------------------------------------
# 1. LOAD INFERENCE PIPELINE
# -------------------------------------------------------------
def load_inference_pipeline(adapter_dir: str = "./control_tuner_final_adapter"):
    print(f"Loading tokenizer from: {adapter_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    print("Configuring 4-bit NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    print("Loading base model: Qwen/Qwen2.5-1.5B...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("Attaching trained LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    return model, tokenizer


# -------------------------------------------------------------
# 2. GENERATE SOLUTION WITH CLEAN TURN TRUNCATION
# -------------------------------------------------------------
def generate_control_solution(model, tokenizer, user_question: str) -> str:
    messages = [
        {"role": "system", "content": "You are ControlTuner, a precision control systems design assistant."},
        {"role": "user", "content": user_question}
    ]

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
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_len = inputs["input_ids"].shape[1]
    response_tokens = output_ids[0][prompt_len:]
    
    # Decode raw text and split cleanly at the end-of-turn token
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=False)
    if "<|im_end|>" in response_text:
        response_text = response_text.split("<|im_end|>")[0]
    if "<|im_start|>" in response_text:
        response_text = response_text.split("<|im_start|>")[0]

    return response_text.strip()


# -------------------------------------------------------------
# 3. DOMAIN BENCHMARK VERIFIER
# -------------------------------------------------------------
def verify_control_response(response_text: str) -> dict:
    text_lower = response_text.lower()

    # 1. Format check (all 3 sections present)
    has_sec1 = "1." in response_text and ("state-space" in text_lower or "matrix" in text_lower)
    has_sec2 = "2." in response_text and "controllab" in text_lower
    has_sec3 = "3." in response_text and "stability" in text_lower
    format_pass = bool(has_sec1 and has_sec2 and has_sec3)

    # 2. Controllability check
    has_rank = "rank" in text_lower
    has_ctrl = "controllable" in text_lower
    controllability_pass = bool(has_rank and has_ctrl)

    # 3. Stability check (handles 'stable', 'unstable', and 'instability')
    has_poles = any(term in text_lower for term in ["eigenvalue", "pole", "det(si", "s_1", "s_2", "re(s)", "roots"])
    has_verdict = any(term in text_lower for term in ["stable", "unstable", "instab"])
    stability_pass = bool(has_poles and has_verdict)

    all_passed = bool(format_pass and controllability_pass and stability_pass)

    return {
        "format_pass": format_pass,
        "controllability_pass": controllability_pass,
        "stability_pass": stability_pass,
        "all_passed": all_passed
    }


# -------------------------------------------------------------
# 4. BENCHMARK EXECUTION
# -------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4: AUTOMATED EVALUATION & DOMAIN BENCHMARK")
    print("=" * 60)

    model, tokenizer = load_inference_pipeline("./control_tuner_final_adapter")

    test_cases = [
        "Given continuous state-space system dx/dt = A*x + B*u with A = [[0, 1], [-2, -3]] and B = [[0], [1]]. Is the system controllable and open-loop stable?",
        "Linearize an inverted pendulum on a cart around upright equilibrium (theta=0). M=1.0kg, m=0.1kg, l=0.5m, g=9.81m/s^2. Give state-space A and B matrices for state x = [p, p_dot, theta, theta_dot]^T.",
        "For a DC motor position control system with states x = [theta, omega]^T, J=0.01, b=0.1, Kt=0.01, R_a=1.0. State the A and B matrices and evaluate open-loop stability."
    ]

    total_tests = len(test_cases)
    passed_tests = 0

    for i, test_question in enumerate(test_cases, 1):
        print("\n" + "-" * 60)
        print(f"TEST CASE {i}/{total_tests}")
        print(f"Prompt: {test_question}\n")

        solution = generate_control_solution(model, tokenizer, test_question)
        print("Generated Solution:")
        print(solution)

        results = verify_control_response(solution)
        print("\nVerification Results:")
        for metric, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  - {metric}: [{status}]")

        if results["all_passed"]:
            passed_tests += 1

    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETED: {passed_tests}/{total_tests} Tests Passed ({100 * passed_tests / total_tests:.1f}%)")
    print("=" * 60)