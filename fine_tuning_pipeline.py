import os
import re
import pandas as pd
from openai import OpenAI
import random
import json
from typing import List, Dict, Tuple, Optional

class FineTuningPipeline:
    """
    A pipeline for fine-tuning GPT models on LinkedIn posts data.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4.1"):
        """
        Initialize the fine-tuning pipeline.
        
        Args:
            api_key: OpenAI API key
            model: Base model to use for fine-tuning
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = """
        You are an expert prompt Engineer and content creator.
        Analyze the posts and draft a prompt that is composed of the main topic plus any references, if available.
        Here is the structure of your output:
        Topic: [topic]
        References: [references]
        """
        self.system_message_posts = """
        You are Diogo, you create content in Data and AI.
        You are an expert in writing LinkedIn Posts.
        You write punchy, spicy, no-bs posts with simple and informal vocabulary.
        You start the posts with a one sentence provocative hook.
        Your paragraphs are 1 sentence long.
        """

    def read_linkedin_posts(self, path: str = "LinkedIn Posts") -> List[Dict[str, str]]:
        """
        Read LinkedIn posts from text files.
        
        Args:
            path: Directory containing LinkedIn post files
            
        Returns:
            List of dictionaries with post content
        """
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        posts = []
        
        for file in files:
            with open(os.path.join(path, file), "r") as f:
                post = f.read()
            
            posts.append({'content': f"Post: {post}"})
        
        return posts

    def extract_prompts(self, posts: List[Dict[str, str]]) -> List[str]:
        """
        Extract prompts from posts using GPT model.
        
        Args:
            posts: List of post dictionaries
            
        Returns:
            List of extracted prompts
        """
        prompts = []
        
        for post in posts:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": post['content']}
                ]
            )
            prompts.append(completion.choices[0].message.content)
        
        return prompts

    def prepare_training_data(self, posts: List[Dict[str, str]], prompts: List[str], 
                            train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare training and validation data.
        
        Args:
            posts: List of post dictionaries
            prompts: List of extracted prompts
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (training_data, validation_data)
        """
        # Combine posts and prompts
        combined_data = list(zip(posts, prompts))
        random.shuffle(combined_data)
        
        # Split into train and test
        train_size = int(train_ratio * len(combined_data))
        train = combined_data[:train_size]
        test = combined_data[train_size:]
        
        # Prepare data
        train_data = []
        validation_data = []
        
        for post, prompt in train:
            train_data.append(self._prepare_data(self.system_message_posts, prompt, post['content']))
        
        for post, prompt in test:
            validation_data.append(self._prepare_data(self.system_message_posts, prompt, post['content']))
        
        return train_data, validation_data

    def _prepare_data(self, system_message: str, prompt: str, output: str) -> Dict:
        """
        Prepare a single training example.
        
        Args:
            system_message: System message for the conversation
            prompt: User prompt
            output: Expected assistant response
            
        Returns:
            Dictionary with messages for training
        """
        return {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output}
            ]
        }

    def write_jsonl(self, data_list: List[Dict], filename: str) -> None:
        """
        Write data to JSONL format.
        
        Args:
            data_list: List of training examples
            filename: Output filename
        """
        with open(filename, "w") as out:
            for ddict in data_list:
                jout = json.dumps(ddict) + "\n"
                out.write(jout)

    def upload_file(self, file_name: str, purpose: str = "fine-tune") -> str:
        """
        Upload file to OpenAI API.
        
        Args:
            file_name: Path to file to upload
            purpose: Purpose of the file upload
            
        Returns:
            File ID from OpenAI
        """
        with open(file_name, "rb") as file:
            response = self.client.files.create(file=file, purpose=purpose)
        return response.id

    def create_fine_tuning_job(self, training_file_id: str, validation_file_id: str, 
                              model: str = "gpt-4.1-2025-04-14", suffix: str = "linkedin-ztm") -> str:
        """
        Create a fine-tuning job.
        
        Args:
            training_file_id: ID of training file
            validation_file_id: ID of validation file
            model: Model to fine-tune
            suffix: Suffix for the fine-tuned model
            
        Returns:
            Fine-tuning job ID
        """
        response = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=model,
            suffix=suffix
        )
        return response.id

    def get_fine_tuned_model_id(self, job_id: str) -> Optional[str]:
        """
        Get the fine-tuned model ID from a job.
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            Fine-tuned model ID or None if not ready
        """
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        return job.fine_tuned_model

    def generate_with_fine_tuned_model(self, model_id: str, user_prompt: str, 
                                     temperature: float = 1.1) -> str:
        """
        Generate text using a fine-tuned model.
        
        Args:
            model_id: Fine-tuned model ID
            user_prompt: User prompt
            temperature: Generation temperature
            
        Returns:
            Generated text
        """
        messages = [
            {"role": "system", "content": self.system_message_posts},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature
        )
        
        return response.choices[0].message.content

    def run_full_pipeline(self, linkedin_posts_path: str = "LinkedIn Posts", 
                         output_train_file: str = "train.jsonl",
                         output_validation_file: str = "validation.jsonl") -> Tuple[str, str]:
        """
        Run the complete fine-tuning pipeline.
        
        Args:
            linkedin_posts_path: Path to LinkedIn posts directory
            output_train_file: Output filename for training data
            output_validation_file: Output filename for validation data
            
        Returns:
            Tuple of (training_file_id, validation_file_id)
        """
        # Read LinkedIn posts
        print("Reading LinkedIn posts...")
        posts = self.read_linkedin_posts(linkedin_posts_path)
        
        # Extract prompts
        print("Extracting prompts...")
        prompts = self.extract_prompts(posts)
        
        # Prepare training data
        print("Preparing training data...")
        train_data, validation_data = self.prepare_training_data(posts, prompts)
        
        # Write JSONL files
        print("Writing JSONL files...")
        self.write_jsonl(train_data, output_train_file)
        self.write_jsonl(validation_data, output_validation_file)
        
        # Upload files
        print("Uploading files to OpenAI...")
        train_file_id = self.upload_file(output_train_file)
        validation_file_id = self.upload_file(output_validation_file)
        
        print(f"Training file ID: {train_file_id}")
        print(f"Validation file ID: {validation_file_id}")
        
        return train_file_id, validation_file_id

def main():
    """
    Example usage of the FineTuningPipeline.
    """
    # Initialize the pipeline
    api_key = "your-openai-api-key-here"  # Replace with your actual API key
    pipeline = FineTuningPipeline(api_key)
    
    # Run the full pipeline
    train_file_id, validation_file_id = pipeline.run_full_pipeline()
    
    # Create fine-tuning job
    job_id = pipeline.create_fine_tuning_job(train_file_id, validation_file_id)
    print(f"Fine-tuning job ID: {job_id}")
    
    # Example of using the fine-tuned model (after training is complete)
    # model_id = pipeline.get_fine_tuned_model_id(job_id)
    # if model_id:
    #     user_prompt = """
    #     TOPIC: Launch of new ZTM course on Fine Tuning a GPT Model
    #     
    #     REFERENCES: GPT-4.1 model fine tuning, building a kick ass model for linkedIn posts, reference that we are the Henry Ford of the AI world
    #     """
    #     generated_text = pipeline.generate_with_fine_tuned_model(model_id, user_prompt)
    #     print("Generated text:")
    #     print(generated_text)

if __name__ == "__main__":
    main() 