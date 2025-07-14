#!/usr/bin/env python3
"""
Example usage of the FineTuningPipeline module.
"""

import os
from fine_tuning_pipeline import FineTuningPipeline

def main():
    """
    Example of how to use the FineTuningPipeline in your main application.
    """
    # Get API key from environment variable (recommended for security)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Initialize the pipeline
    pipeline = FineTuningPipeline(api_key)
    
    # Option 1: Run the complete pipeline
    print("=== Running Complete Pipeline ===")
    train_file_id, validation_file_id = pipeline.run_full_pipeline()
    
    # Create fine-tuning job
    job_id = pipeline.create_fine_tuning_job(train_file_id, validation_file_id)
    print(f"Fine-tuning job created: {job_id}")
    
    # Option 2: Use individual components
    print("\n=== Using Individual Components ===")
    
    # Read posts
    posts = pipeline.read_linkedin_posts("LinkedIn Posts")
    print(f"Read {len(posts)} posts")
    
    # Extract prompts
    prompts = pipeline.extract_prompts(posts)
    print(f"Extracted {len(prompts)} prompts")
    
    # Prepare training data
    train_data, validation_data = pipeline.prepare_training_data(posts, prompts)
    print(f"Prepared {len(train_data)} training examples and {len(validation_data)} validation examples")
    
    # Option 3: Use a fine-tuned model (after training is complete)
    print("\n=== Using Fine-tuned Model ===")
    
    # Uncomment and modify these lines after your fine-tuning job is complete
    # model_id = pipeline.get_fine_tuned_model_id(job_id)
    # if model_id:
    #     user_prompt = """
    #     TOPIC: Launch of new ZTM course on Fine Tuning a GPT Model
    #     
    #     REFERENCES: GPT-4.1 model fine tuning, building a kick ass model for linkedIn posts, reference that we are the Henry Ford of the AI world
    #     """
    #     generated_text = pipeline.generate_with_fine_tuned_model(model_id, user_prompt)
    #     print("Generated LinkedIn post:")
    #     print(generated_text)
    # else:
    #     print("Fine-tuned model not ready yet. Check job status.")

def check_fine_tuning_status(job_id: str):
    """
    Check the status of a fine-tuning job.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    pipeline = FineTuningPipeline(api_key)
    
    # Get job status
    job = pipeline.client.fine_tuning.jobs.retrieve(job_id)
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Created at: {job.created_at}")
    print(f"Finished at: {job.finished_at}")
    
    if job.fine_tuned_model:
        print(f"Fine-tuned model: {job.fine_tuned_model}")
    else:
        print("Fine-tuned model not ready yet")

if __name__ == "__main__":
    main()
    
    # Example of checking job status (uncomment and provide job_id)
    # check_fine_tuning_status("your-job-id-here") 