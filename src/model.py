import torch
import base64
from io import BytesIO
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def load_vision_model():
    print("Loading AI model in CPU Mode (This will use more RAM and be slower)...")
    
    # We removed BitsAndBytes Config since it strictly requires NVIDIA GPUs
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        device_map="cpu", # Forced to CPU
        torch_dtype=torch.float32, # Standard CPU math format
        low_cpu_mem_usage=True
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    print("Model loaded successfully on CPU.")
    return model, processor

def answer_my_question(user_question, RAG, model, processor):
    print(f"Searching PDF for: {user_question}")
    results = RAG.search(user_question, k=3)

    cot_instructions = (
        f"Task: Answer the question '{user_question}' using ONLY the provided image.\n\n"
        "STEPS TO FOLLOW:\n"
        "1. SCAN FOR FIGURES: Carefully look for any charts, graphs, flowcharts, tables, or diagrams on this page.\n"
        "2. DECONSTRUCT THE VISUAL: If there is a figure or table, explicitly write out its Title, the labels on its axes (X and Y), column headers, and the legend categories before doing anything else. If there are no figures, summarize the bold text headings.\n"
        "3. EXTRACT DATA: Read the specific text, data points, bars, or lines in the figure or paragraphs that relate to the question.\n"
        "4. ANSWER & PROOF: Provide the final answer based ONLY on the data you just extracted. You MUST provide a direct quote or specific visual reference (e.g., 'The blue bar for 2019 shows...') to prove it.\n"
        "5. IRRELEVANT: If the exact information to answer the question is NOT visually printed on this specific page, reply exactly with: NOT_FOUND"
    )

    for index, result in enumerate(results):
        print(f"Checking Page {result.page_num} for visual evidence...")
        
        image_bytes = base64.b64decode(result.base64)
        page_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        page_number = result.page_num
        
        page_image.thumbnail((1024, 1024))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image},
                    {"type": "text", "text": cot_instructions},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Changed return_tensors to send to 'cpu' instead of 'cuda'
        inputs = processor(text=[text], images=[page_image], padding=True, return_tensors="pt").to("cpu")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=400, 
                do_sample=False 
            )
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        ai_answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        if "NOT_FOUND" not in ai_answer:
            print(f"Verified Answer Found on Page {page_number}")
            return f"{ai_answer}\n\n**[Source: Verified Visual Extraction - Page {page_number}]**", page_image
        
        print(f"Page {page_number} did not have visual proof. Moving to next page...")

    return "The information was not visually present on the retrieved pages. Please try a different query.", None