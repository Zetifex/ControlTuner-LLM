import re
import torch
from datasets import load_dataset
from evaluate_benchmark import load_inference_pipeline


def extract_numerical_answer(text: str) -> str:
    """
    Extracts the final numerical answer from an LLM's solution.
    GSM8K ground-truth solutions end with '#### X'.
    """
    match = re.findall(r"####\s*(-?\d+)", text)
    if match:
        return match[-1]

    match = re.findall(r"answer is\s*(-?\d+)", text, re.IGNORECASE)
    if match:
        return match[-1]

    # Fallback: find the last integer in the output
    numbers = re.findall(r"-?\d+", text)
    return numbers[-1] if numbers else ""


def run_gsm8k_benchmark(num_samples: int = 10):
    """
    Evaluates our model on real GSM8K mathematical reasoning problems from OpenAI.
    """
    print("=" * 60)
    print(f"EVALUATING MODEL ON GSM8K MATH BENCHMARK ({num_samples} SAMPLES)")
    print("=" * 60)

    # 1. Load fine-tuned model and tokenizer
    model, tokenizer = load_inference_pipeline("./control_tuner_final_adapter")

    # 2. Download the official GSM8K dataset using full namespace "openai/gsm8k"
    print("Downloading GSM8K test split from openai/gsm8k...")
    gsm8k = load_dataset("openai/gsm8k", "main", split=f"test[:{num_samples}]")

    correct = 0

    for i, example in enumerate(gsm8k, 1):
        question = example["question"]
        ground_truth = extract_numerical_answer(example["answer"])

        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve this problem step-by-step and conclude with 'The answer is X'."},
            {"role": "user", "content": question}
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
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        prompt_len = inputs["input_ids"].shape[1]
        response_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        predicted_answer = extract_numerical_answer(response_text)

        is_correct = (predicted_answer == ground_truth)
        if is_correct:
            correct += 1

        print(f"\n[Problem {i}/{num_samples}]")
        print(f"Q: {question[:80]}...")
        print(f"Predicted: '{predicted_answer}' | Ground Truth: '{ground_truth}' -> [{'CORRECT' if is_correct else 'INCORRECT'}]")

    score = 100 * (correct / num_samples)
    print("\n" + "=" * 60)
    print(f"GSM8K BENCHMARK SCORE: {correct}/{num_samples} ({score:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_gsm8k_benchmark(num_samples=10)