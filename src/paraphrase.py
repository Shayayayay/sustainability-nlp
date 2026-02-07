from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def paraphrase(text):

    prompt = "paraphrase: " + text + " </s>"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_length=80,
            num_beams=5,
            num_return_sequences=1,
            do_sample=False,
            early_stopping=True
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return result


if __name__ == "__main__":

    s = input("Enter sentence: ")

    print("\nParaphrased:")
    print(paraphrase(s))
