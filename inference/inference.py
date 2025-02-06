import argparse
import json
import os
from typing import Dict, List
from tqdm import tqdm
import torch
from PIL import Image
from abc import ABC, abstractmethod
import sys
import base64
from io import BytesIO
from openai import OpenAI

from emu3.mllm.processing_emu3 import Emu3Processor

class BaseModel(ABC):
    """Abstract base class for all models"""
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, question: str, image_path: str, choices: List[str]) -> str:
        """
        Make prediction for a single question
        Returns: answer string (one of "A", "B", "C", "D", "E")
        """
        pass

class GPTModel(BaseModel):
    """Implementation for GPT model using API"""
    def __init__(self):
        # Initialize OpenAI client with provided configuration
        self.client = OpenAI(
            base_url='https://api.gptsapi.net/v1',
            api_key='sk-Dsc5e485f92f56eb626b0a626e3315ee619c537db6e6zQ7Z'
        )

    def encode_image(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def predict(self, question: str, image_path: str, choices: List[str]) -> str:
        try:
            # Encode image to base64
            base64_image = self.encode_image(image_path)

            # Construct the system prompt to enforce the format
            system_prompt = """You are a visual question answering system. Given an image and a multiple choice question:
1. Analyze the image carefully
2. Read the question and all choices
3. Select the most appropriate answer
4. Respond ONLY with the letter (A, B, C, D, or E) of your chosen answer, nothing else."""

            # Construct user prompt with question and choices
            user_prompt = f"Question: {question}\n\nChoices:\n"
            for choice in choices:
                user_prompt += f"{choice}\n"

            # Make API call with both image and text
            try:
                chat_completion = self.client.chat.completions.create(
                    model="gpt-4o",  # Use vision model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]}
                    ],
                    max_tokens=10,  # Keep response short since we only need a letter
                    temperature=0.3  # Lower temperature for more consistent responses
                )
                
                # Extract the answer
                response = chat_completion.choices[0].message.content.strip().upper()
                
                # Return the first valid answer letter found
                for char in response:
                    if char in "ABCDE":
                        return char
                
                return ""  # Return empty string if no valid answer found

            except Exception as api_error:
                print(f"API call error: {api_error}")
                return ""
                
        except Exception as e:
            print(f"Error in GPT prediction: {e}")
            return ""


def process_dataset(data: List[Dict], model: BaseModel, model_id: str, base_path: str, output_dir: str) -> List[Dict]:
    """Process the entire dataset"""
    results = []
    IMAGE_PREFIX = "/DATA/disk0/nby/xwl/decode_images"
    
    for item in tqdm(data, desc=f"Processing with {model_id}"):
        try:
            # Get image path and normalize it
            image_path = os.path.join(IMAGE_PREFIX, item['image'][0])
            # Convert Windows path separators to Unix and fix spaces
            image_path = image_path.replace('\\', '/').replace(' ', '_')
            
            print(f"Processing image: {image_path}")  # Debug print

            #/DATA/disk0/nby/xwl/decode_images/Single_Image_Perception_and_Understanding/MMBench/143252.png

            # Verify file exists
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Get prediction
            answer = model.predict(
                question=item['question'],
                image_path=image_path,
                choices=item['choice']
            )
            
            # Create result entry
            result = {
                "question_id": item["question_id"],
                "category": item["category"],
                "output": answer,
                "model_id": model_id
            }
            results.append(result)
            
            # Save intermediate results after each prediction
            output_file = os.path.join(output_dir, f"result_{model_id}.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error processing item {item.get('question_id', 'unknown')}: {e}")
            print(f"Question: {item.get('question', '')}")
            print(f"Image path: {image_path if 'image_path' in locals() else 'Not available'}")
            # Add error result
            result = {
                "question_id": item.get("question_id", "unknown"),
                "category": item.get("category", "unknown"),
                "answer": "",
                "model_id": model_id
            }
            results.append(result)

    return results

class Emu3Model(BaseModel):
    """Implementation for Emu3 model"""
    def __init__(self):
        from transformers import (
            AutoTokenizer, 
            AutoModel, 
            AutoImageProcessor, 
            AutoModelForCausalLM,
            GenerationConfig
        )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            "BAAI/Emu3-Chat",
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/Emu3-Chat",
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Initialize image components
        self.image_processor = AutoImageProcessor.from_pretrained(
            "BAAI/Emu3-VisionTokenier",
            trust_remote_code=True
        )
        self.image_tokenizer = AutoModel.from_pretrained(
            "BAAI/Emu3-VisionTokenier",
            device_map=self.device,
            trust_remote_code=True
        ).eval()
        
        # Initialize processor
        self.processor = Emu3Processor(
            self.image_processor,
            self.image_tokenizer,
            self.tokenizer
        )
        
        # Initialize generation config
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=128,
        )

    def predict(self, question: str, image_path: str, choices: List[str]) -> str:
        try:
            # Load image
            image = Image.open(image_path)
            
            # Construct prompt
            prompt = f"Question: {question}\n\nChoices:\n"
            for choice in choices:
                prompt += f"{choice}\n"
            prompt += "\nPlease select the best answer. Only output a single letter (A, B, C, D, or E)."

            # Process inputs
            inputs = self.processor(
                text=prompt,
                image=image,
                mode='U',
                return_tensors="pt",
                padding="longest",
            )

            # Generate output
            outputs = self.model.generate(
                inputs.input_ids.to(self.device),
                self.generation_config,
                attention_mask=inputs.attention_mask.to(self.device),
            )
            
            # Extract answer
            outputs = outputs[:, inputs.input_ids.shape[-1]:]
            output_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Get answer letter
            output_text = output_text.strip().upper()
            for char in output_text:
                if char in "ABCDE":
                    return char
            return ""
            
        except Exception as e:
            print(f"Error in Emu3 prediction: {e}")
            return ""


def get_model(model_id: str) -> BaseModel:
    """Factory function to get the appropriate model"""
    model_map = {
        "emu3": Emu3Model,
        "gpt": GPTModel,
    }
    
    model_id = model_id.lower()
    if model_id not in model_map:
        raise ValueError(f"Unknown model_id: {model_id}")
    
    return model_map[model_id]()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    args = parser.parse_args()
    
    # Load input data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get base path for images
    base_path = os.path.dirname(args.input_file)
    
    # Get output directory
    output_dir = os.path.dirname(args.output_file)
    
    try:
        # Initialize model
        model = get_model(args.model_id)
        
        # Process dataset
        results = process_dataset(data, model, args.model_id, base_path, output_dir)
        
        # Save final results
        output_file = os.path.join(output_dir, f"result_{args.model_id}.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()