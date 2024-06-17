import openai
from helm import evaluate  # Hypothetical import, adjust according to actual CRFM-HELM library
import os
# Set your OpenAI API key
openai.api_key = os.getenv("OPENAIKEY')   #'your-api-key-here'

def get_openai_response(prompt, model="text-davinci-003"):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Example prompt
prompt = "Once upon a time,"

# Get response from OpenAI model
response = get_openai_response(prompt)
print(f"Original Response: {response}")

# Perform evaluation using CRFM-HELM
evaluation_result = evaluate(
    model_name="gpt-3",  # Replace with the actual model name in CRFM-HELM
    prompt=prompt,
    response=response
)

print("Evaluation Result:")
print(evaluation_result)
