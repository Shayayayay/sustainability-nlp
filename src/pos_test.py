import spacy

nlp = spacy.load("en_core_web_sm")

text = "I am so cool"

doc = nlp(text)

for token in doc:
    print(token.text, token.pos_)
