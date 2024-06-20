import re
import pickle
from flask import Flask, render_template, request
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import torch
from transformer_module import MultiHeadAttention, PositionalEncoding, PositionWiseFeedForward, DecoderLayer, Transformer, TextDataset
allprompt = []

def generate_text(model, tokenizer, input_text, max_length=50, do_sample=True, num_beams=5, no_repeat_ngram_size=2, early_stopping=True):
    inputs = tokenizer(input_text, return_tensors='pt')['input_ids']
    input_ids = inputs
    batch_size = input_ids.size(0)

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9  # Set initial scores for non-beam 0 elements to a large negative value
    beam_scores = beam_scores.view(-1)  # Flatten beam scores to [batch_size * num_beams]

    # Beam search setup
    beam_outputs = torch.zeros((batch_size * num_beams, max_length), dtype=torch.long, device=input_ids.device)
    beam_outputs[:, :input_ids.size(1)] = input_ids.repeat(1, num_beams).view(-1, input_ids.size(1))

    cur_len = input_ids.size(1)

    for _ in range(max_length - cur_len):
        model_inputs = beam_outputs[:, :cur_len]
        outputs = model(model_inputs)
        next_token_logits = outputs[:, -1, :]

        # Apply repetition penalty (no_repeat_ngram_size)
        if no_repeat_ngram_size > 0:
            for batch_idx in range(batch_size * num_beams):
                generated_tokens = beam_outputs[batch_idx].tolist()
                for ngram in zip(*[generated_tokens[i:] for i in range(no_repeat_ngram_size)]):
                    ngram_tokens = tuple(ngram)
                    ngram_indices = [i for i in range(len(generated_tokens) - no_repeat_ngram_size + 1)
                                     if tuple(generated_tokens[i:i + no_repeat_ngram_size]) == ngram_tokens]
                    for idx in ngram_indices:
                        next_token_logits[batch_idx, generated_tokens[idx + no_repeat_ngram_size - 1]] = -float('inf')

        # Apply softmax to get probabilities
        if do_sample:
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token_ids = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
        else:
            next_token_ids = torch.argmax(next_token_logits, dim=-1)

        # Update scores and outputs
        beam_scores = beam_scores + F.log_softmax(next_token_logits, dim=-1)[range(batch_size * num_beams), next_token_ids]
        next_token_ids = next_token_ids.view(batch_size, num_beams)
        beam_outputs[:, cur_len] = next_token_ids.view(-1)

        # Early stopping if all beams end with eos token
        if early_stopping:
            if all((tokenizer.eos_token_id in beam_outputs[i, :cur_len+1] for i in range(batch_size * num_beams))):
                break

        cur_len += 1

    output_texts = []
    for i in range(batch_size):
        best_beam_idx = torch.argmax(beam_scores[i*num_beams: (i+1)*num_beams])
        best_output = beam_outputs[i*num_beams + best_beam_idx]
        output_texts.append(tokenizer.decode(best_output[len(input_text):], skip_special_tokens=True))  # exclude input_text from the output

    return output_texts[0] if len(output_texts) == 1 else output_texts

app = Flask(__name__)

model = torch.load("checkpoint_openwebtext.pth", map_location=torch.device('cpu'))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text2():
    prompt = request.form['prompt']
    allprompt.append(prompt)
    output_text = generate_text(model, tokenizer, " ".join(prompt))
    allprompt.append(output_text)
    # Return only the generated text without the input prompt
    return render_template('result.html', generated_text=output_text)

if __name__ == '__main__':
    app.run(debug=True)
