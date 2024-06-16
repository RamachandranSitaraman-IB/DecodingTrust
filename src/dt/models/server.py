from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load the tokenizer and model
model_name = "./downloadedmodels"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


@app.post("/generate")
async def generate(request: Request):
    json = await request.json()
    prompt = json['prompt']
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Explicitly set the pad token ID and attention mask
    inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id  # Set the pad token ID to the end-of-sequence token ID
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}


@app.get("/")
async def read_root():
    return {"message": "Welcome to the GPT-2 FastAPI server!"}
