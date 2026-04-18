import os
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "" # This forces pure CPU mode
os.environ["ACCELERATE_USE_CPU"] = "True" # Forces Hugging Face to strictly use CPU
# Add this line to link Poppler!
os.environ["PATH"] += os.pathsep + r"D:\uni sem 2\MultiMedia\A1\poppler\Library\bin"
import gradio as gr
import pandas as pd
from src.data_processing import setup_rag_index
from src.model import load_vision_model, answer_my_question

print("Initializing System...")
RAG = setup_rag_index()
model, processor = load_vision_model()

# Wrapper to pass model/RAG easily from UI
def process_query(question):
    return answer_my_question(question, RAG, model, processor)

def run_evaluation_suite():
    benchmarks = [
        {"modality": "Text", "query": "How does this report specifically define 'mental health'?"},
        {"modality": "Table/Data", "query": "What is the electronic ISBN of this report?"},
        {"modality": "Image/Chart", "query": "According to fig 2.4, what are the RISKS that undermine mental health?"},
        {"modality": "Anti-Hallucination", "query": "Based ONLY on the Acknowledgements section, how many people die by suicide each year?"}
    ]

    results_data = []
    print("Starting Visual Evaluation Suite...\n")
    
    for item in benchmarks:
        print(f"Running: {item['modality']}...")
        try:
            answer, _ = process_query(item["query"])
        except Exception as e:
            answer = f"Error: {str(e)}"

        results_data.append({
            "Target Modality": item["modality"],
            "Benchmark Query": item["query"],
            "Generated Answer": answer
        })

    print("\nEvaluation Complete!")
    return pd.DataFrame(results_data)

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Stress & Health: Book-Level Multi-Modal RAG (~300 Pages)")

    with gr.Tabs():
        with gr.Tab("Interactive QA"):
            with gr.Row():
                with gr.Column():
                    user_input = gr.Textbox(label="Type your question here:")
                    ask_button = gr.Button("Ask Question", variant="primary")
                    answer_box = gr.Markdown(label="AI Answer")
                with gr.Column():
                    image_box = gr.Image(label="Page Used for Answer", type="pil")

            ask_button.click(fn=process_query, inputs=user_input, outputs=[answer_box, image_box])

        with gr.Tab("Evaluation Suite"):
            gr.Markdown("### Automated Benchmarks")
            eval_btn = gr.Button("Run Benchmark", variant="secondary")
            eval_output = gr.Dataframe(label="Evaluation Results", wrap=True)
            eval_btn.click(fn=run_evaluation_suite, inputs=None, outputs=eval_output)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)