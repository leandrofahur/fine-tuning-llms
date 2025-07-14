# Fine-Tuning Pipeline for LinkedIn Posts

This module provides a complete pipeline for fine-tuning GPT models on LinkedIn posts data. It extracts the functionality from the Jupyter notebook and organizes it into a reusable Python class.

## Features

- **LinkedIn Posts Processing**: Read and process LinkedIn posts from text files
- **Prompt Extraction**: Use GPT to extract prompts from posts
- **Training Data Preparation**: Prepare training and validation datasets
- **Fine-tuning Job Management**: Create and monitor fine-tuning jobs
- **Text Generation**: Generate LinkedIn-style posts using fine-tuned models

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from fine_tuning_pipeline import FineTuningPipeline

# Initialize the pipeline
pipeline = FineTuningPipeline(api_key="your-api-key")

# Run the complete pipeline
train_file_id, validation_file_id = pipeline.run_full_pipeline()

# Create a fine-tuning job
job_id = pipeline.create_fine_tuning_job(train_file_id, validation_file_id)
```

### Step-by-Step Usage

```python
# Read LinkedIn posts
posts = pipeline.read_linkedin_posts("LinkedIn Posts")

# Extract prompts
prompts = pipeline.extract_prompts(posts)

# Prepare training data
train_data, validation_data = pipeline.prepare_training_data(posts, prompts)

# Write JSONL files
pipeline.write_jsonl(train_data, "train.jsonl")
pipeline.write_jsonl(validation_data, "validation.jsonl")

# Upload files
train_file_id = pipeline.upload_file("train.jsonl")
validation_file_id = pipeline.upload_file("validation.jsonl")

# Create fine-tuning job
job_id = pipeline.create_fine_tuning_job(train_file_id, validation_file_id)
```

### Using Fine-tuned Models

```python
# Get the fine-tuned model ID (after training is complete)
model_id = pipeline.get_fine_tuned_model_id(job_id)

# Generate text
user_prompt = """
TOPIC: Launch of new ZTM course on Fine Tuning a GPT Model

REFERENCES: GPT-4.1 model fine tuning, building a kick ass model for linkedIn posts, reference that we are the Henry Ford of the AI world
"""

generated_text = pipeline.generate_with_fine_tuned_model(model_id, user_prompt)
print(generated_text)
```

## File Structure

```
├── fine_tuning_pipeline.py    # Main pipeline module
├── example_usage.py           # Example usage script
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── LinkedIn Posts/           # Directory containing LinkedIn post files
    ├── post1.txt
    ├── post2.txt
    └── ...
```

## Configuration

### System Prompts

The pipeline uses two main system prompts:

1. **Prompt Extraction**: Extracts topics and references from posts
2. **Post Generation**: Defines the style for generating LinkedIn posts

You can modify these in the `FineTuningPipeline` class:

```python
pipeline = FineTuningPipeline(api_key)
pipeline.system_prompt = "Your custom prompt extraction prompt"
pipeline.system_message_posts = "Your custom post generation prompt"
```

### Model Configuration

- **Base Model**: Default is "gpt-4.1"
- **Fine-tuning Model**: Default is "gpt-4.1-2025-04-14"
- **Temperature**: Default is 1.1 for generation

## Monitoring Fine-tuning Jobs

```python
# Check job status
job = pipeline.client.fine_tuning.jobs.retrieve(job_id)
print(f"Status: {job.status}")
print(f"Fine-tuned model: {job.fine_tuned_model}")
```

## Error Handling

The pipeline includes basic error handling, but you should add additional error handling for production use:

```python
try:
    train_file_id, validation_file_id = pipeline.run_full_pipeline()
except Exception as e:
    print(f"Error running pipeline: {e}")
```

## Security Notes

- Always use environment variables for API keys
- Never commit API keys to version control
- Consider rate limiting for large datasets

## Example Output

The fine-tuned model generates LinkedIn posts in the style of Diogo, with:
- Punchy, spicy content
- Simple and informal vocabulary
- One-sentence paragraphs
- Provocative hooks
- Focus on Data and AI topics