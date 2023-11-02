import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')

with open('shakespeare.txt', 'r') as f:
    text = f.read()
    
sentences = sent_tokenize(text)  # Split the text into sentences
tokenized_sentences = [word_tokenize(s) for s in sentences]  # Split each sentence into words

processed_sentences = []
for sentence in tokenized_sentences:
    processed_sentence = []
    for token in sentence:
        token = token.lower()  # Convert to lowercase
        token = token.strip()  # Remove leading/trailing white spaces
        token = token.replace('\n', '')  # Remove newlines
        processed_sentence.append(token)
    processed_sentences.append(processed_sentence)
    
processed_sentences_str = [' '.join(sentence) for sentence in processed_sentences]

# Join all the sentences back into a single string
processed_text = '\n'.join(processed_sentences_str)

# Write the processed text to a new file
with open('processed_shakespeare.txt', 'w') as f:
    f.write(processed_text)