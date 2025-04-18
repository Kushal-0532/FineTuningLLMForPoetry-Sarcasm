!pip install transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_sarcasm(prompt):
    instruction = f"Respond with sarcasm to: \"{prompt}\""
    input_ids = tokenizer(instruction, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.92,
            temperature=0.9
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.strip()

while True:
    user_input = input("Enter a prompt (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    response = generate_sarcasm(user_input)
    print(f"Sarcastic Response: {response}")