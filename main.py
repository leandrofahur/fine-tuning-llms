# In your main.py or any other file
from fine_tuning_pipeline import FineTuningPipeline
import os

# Initialize
api_key = os.getenv("OPENAI_API_KEY")
pipeline = FineTuningPipeline(api_key)

# Run the complete pipeline
train_file_id, validation_file_id = pipeline.run_full_pipeline()

# Create fine-tuning job
job_id = pipeline.create_fine_tuning_job(train_file_id, validation_file_id)

# Later, use the fine-tuned model
model_id = pipeline.get_fine_tuned_model_id(job_id)
generated_text = pipeline.generate_with_fine_tuned_model(model_id, user_prompt)






