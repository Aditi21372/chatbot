from transformers import pipeline

print("🔍 Loading TinyLlama model...")  # Debug message

# Load TinyLlama (lighter model, ~1.1GB)
free_generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-chat-v1.0",  # ~1.1GB model
    device=-1
)

print("✅ TinyLlama model loaded!")  # Debug message

def generate_free_response(prompt: str, max_new_tokens: int = 100, temperature: float = 0.7, top_p: float = 0.9, **kwargs) -> str:
    """
    Generate response from TinyLlama with controlled token limits.
    
    Args:
        prompt (str): Input prompt.
        max_new_tokens (int): Number of new tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
        **kwargs: Any additional keyword arguments.
    
    Returns:
        str: Generated response text.
    """
    print(f"🔍 Generating response for prompt: {prompt}")  # Debug message

    results = free_generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.2,
        return_full_text=False,  # Only return the generated portion
        truncation=True,         # Provided here only
        **kwargs
    )
    print("✅ Generation complete!")  # Debug message
    return results[0]['generated_text'].strip()

if __name__ == "__main__":
    sample_prompt = "What is the capital of France?"
    print(f"🔍 Testing model with prompt: {sample_prompt}")
    response = generate_free_response(sample_prompt)
    print(f"✅ Response: {response}")
