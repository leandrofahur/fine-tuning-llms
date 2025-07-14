# Import libs:
import os
from os.path import isfile, join
import re
from openai import OpenAI
import json
from dotenv import load_dotenv

# Load env variables:
load_dotenv()

# Connect to the OpenAI API:
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1"


