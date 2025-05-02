# enhanced_summarizer_app.py
import time
import torch
import matplotlib.pyplot as plt
from collections import Counter

from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr

# Device & model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
model.eval()

# Summarize + visualize
def summarize_and_visualize(text, max_length, num_beams):
    if not text.strip():
        return "⚠️ Please enter some text to summarize.", None, None

    # Prepare input
    input_str = "summarize: " + text.strip()
    inputs = tokenizer(
        input_str,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Summarize
    start = time.time()
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=int(max_length),
            num_beams=int(num_beams),
            early_stopping=True
        )
    end = time.time()
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    runtime = end - start

    # Token counts
    orig_tokens = input_ids.numel()
    summ_tokens = summary_ids.numel()

    # Word frequency for summary
    words = summary.lower().split()
    common = Counter(words).most_common(10)
    labels, freqs = zip(*common) if common else ([], [])

    # Plot 1: Token count bar chart
    plt.figure(figsize=(5,4))
    plt.bar(["Original", "Summary"], [orig_tokens, summ_tokens])
    plt.ylabel("Tokens")
    plt.title("Token Count: Original vs. Summary")
    plt.tight_layout()
    bar_chart = plt.gcf()

    # Plot 2: Top 10 word frequency
    plt.figure(figsize=(6,4))
    plt.barh(labels, freqs)
    plt.xlabel("Frequency")
    plt.title("Top 10 Words in Summary")
    plt.tight_layout()
    freq_chart = plt.gcf()

    # Prepare summary markdown
    summary_md = (
        f"**Summary:**\n\n{summary}\n\n"
        f"⏱ Time: {runtime:.2f}s   |   "
        f"Tokens in: {orig_tokens} → out: {summ_tokens}"
    )
    return summary_md, bar_chart, freq_chart

# Gradio interface
iface = gr.Interface(
    fn=summarize_and_visualize,
    inputs=[
        gr.Textbox(lines=8, label="Input Text", placeholder="Paste text here…"),
        gr.Slider(10, 200, step=10, value=60, label="Max Summary Length"),
        gr.Slider(1, 10, step=1, value=4, label="Beam Width")
    ],
    outputs=[
        gr.Markdown(label="Generated Summary"),
        gr.Plot(label="Token Count Chart"),
        gr.Plot(label="Word Frequency Chart")
    ],
    title="📝 T5 Summarizer + Visualizations",
    description="Enter text, adjust summary length & beam width, then view the summary along with token count and word-frequency charts.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
