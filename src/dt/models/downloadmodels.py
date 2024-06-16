from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

# Download the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model and tokenizer
model.save_pretrained("./downloadedmodels")
tokenizer.save_pretrained("./downloadedmodels")
