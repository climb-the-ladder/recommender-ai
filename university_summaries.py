import os
import json
from typing import Dict, Optional, TypedDict
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

class UniversitySummary(TypedDict):
    overview: str
    academic_programs: str
    campus_life: str
    achievements: str
    unique_features: str

class UniversitySummaryGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_summary(self, university_name: str, additional_info: Optional[Dict] = None) -> UniversitySummary:
        """
        Generate a structured summary for a university using GPT-4.
        """
        # Build a flexible prompt
        prompt_parts = [
            f"Generate a structured university summary for '{university_name}'.",
            "Use the following sections: Overview, Academic Programs, Campus Life, Achievements, Unique Features.",
            "Return your output as a valid JSON object with keys: overview, academic_programs, campus_life, achievements, unique_features.",
            "Each section should be around 2-3 sentences long, objective, and helpful to a high school student.",
            "Focus on recent developments and current information.",
            "Include specific details about programs, facilities, and student life.",
        ]

        if additional_info:
            prompt_parts.append("Include the following context in the summary:")
            for key, value in additional_info.items():
                if value:  # Only include non-empty values
                    prompt_parts.append(f"{key}: {value}")

        prompt = "\n".join(prompt_parts)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Using the latest GPT-4 model
                messages=[
                    {"role": "system", "content": "You are an expert education advisor. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}  # Ensure JSON response
            )

            content = response.choices[0].message.content
            summary = json.loads(content)

            # Validate the summary has all required fields
            expected_fields = ["overview", "academic_programs", "campus_life", "achievements", "unique_features"]
            if not all(field in summary for field in expected_fields):
                raise ValueError("Missing required fields in the summary.")

            # Validate field types and content
            for field in expected_fields:
                if not isinstance(summary[field], str) or not summary[field].strip():
                    raise ValueError(f"Invalid content in {field} field")

            return summary

        except json.JSONDecodeError as e:
            print("❌ JSON parse error:", str(e))
            raise ValueError("Model returned invalid JSON.")

        except Exception as e:
            print("❌ Error generating university summary:", str(e))
            return {
                "overview": f"Unable to generate summary for {university_name}.",
                "academic_programs": "N/A",
                "campus_life": "N/A",
                "achievements": "N/A",
                "unique_features": "N/A"
            }

# Singleton usage
university_summary_generator = UniversitySummaryGenerator()
