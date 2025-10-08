"""Manual test for TinkerInferenceWrapper - run this to verify it works."""
import asyncio
import os
import tinker
from tinker_atropos.inference.wrapper import TinkerInferenceWrapper


async def main():
    if not os.getenv("TINKER_API_KEY"):
        print("ERROR: Set TINKER_API_KEY environment variable")
        return

    print("Creating Tinker service client...")
    service_client = tinker.ServiceClient()

    base_model = "meta-llama/Llama-3.1-8B"
    print(f"Creating wrapper for {base_model}...")

    wrapper = TinkerInferenceWrapper(service_client, base_model)

    # Test generate
    print("\nTesting generation...")
    prompts = ["What is 2+2?", "What is the capital of France?"]

    try:
        completions = await wrapper.generate(
            prompts=prompts,
            max_tokens=50,
            temperature=0.7,
        )

        print("\nSuccess! Generated completions:")
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            print(f"\n[{i}] Prompt: {prompt}")
            print(f"    Completion: {completion}")

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
