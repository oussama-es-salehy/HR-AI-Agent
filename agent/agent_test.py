import json
import re
import os
import warnings
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from helper import get_candidate
from tools import tools
from prompt import system_prompt  # your system prompt string
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -----------------------------
# Load a small HF model
# -----------------------------
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1"

load_dotenv()
token = os.environ.get("GITHUB_TOKEN")
if not token:
    raise ValueError("GITHUB_TOKEN is not set. Add it to your .env or environment.")
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

def generate_response(prompt, max_length=200):
    response = client.complete(
        messages=[
            SystemMessage(system_prompt),
            UserMessage(prompt),
        ],
        temperature=1,
        top_p=1,
        model=model_name,
    )
    return response.choices[0].message.content

# -----------------------------
# Agent
# -----------------------------
def hr_ai_agent():
    print("🤖 HR AI Agent: Ask me anything about candidates or jobs! (type 'exit' to quit)")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Compose prompt for LLM
        prompt = f"""
User question: "{user_input}"

Available ML models:
1. Job Fit Prediction -> Inputs: candidate_skills, required_skills, degree, years_experience
2. Salary Prediction -> Inputs: years_experience, role, degree, company_size, location, level
3. Resume Screening -> Inputs: resume_text, job_description_text
4. Candidate Priority -> Inputs: experience_band, skills_coverage, referral_flag, english_level, location_match

Instructions:
- If the question REQUIRES one of these predictions, respond ONLY with a JSON object with keys:
  - model: chosen model name
  - params: dictionary of input parameters for the model
  - note: optional explanation
- Otherwise, answer normally in the same language as the user without JSON.
"""

        llm_output = generate_response(prompt)

        if "{" not in llm_output:
            print("HR AI Agent: I'm not sure which model to use. Please ask about a candidate, job, or salary.")
            continue
        
        try:
            llm_data = json.loads(llm_output)
        except json.JSONDecodeError:
            # Try to extract the first JSON object from the text
            match = re.search(r"\{[\s\S]*\}", llm_output)
            if not match:
                # No JSON -> treat as a normal assistant answer
                print("HR AI Agent Output:")
                print(llm_output.strip())
                continue
            try:
                llm_data = json.loads(match.group(0))
            except Exception:
                print("HR AI Agent Output:")
                print(llm_output.strip())
                continue

        model_name = llm_data.get("model")
        params = llm_data.get("params", {})

        # If no model was chosen, treat as a normal answer when available
        if not model_name:
            direct_answer = llm_data.get("answer") or llm_data.get("response") or llm_data.get("message")
            if direct_answer:
                print("HR AI Agent Output:")
                print(str(direct_answer).strip())
                continue

        # Normalize model name and provide aliases
        aliases = {
            "job fit prediction": "Job Fit Prediction",
            "job fit": "Job Fit Prediction",
            "salary prediction": "Salary Prediction",
            "salary": "Salary Prediction",
            "resume screening": "Resume Screening",
            "resume": "Resume Screening",
            "candidate priority": "Candidate Priority",
            "priority": "Candidate Priority",
        }
        if isinstance(model_name, str):
            key = model_name.strip().lower()
            model_name = aliases.get(key, model_name)

        # Backfill defaults per model
        def _safe_num(x):
            try:
                v = float(x)
                if isinstance(v, float) and (v != v):  # NaN check
                    return 0.0
                return v
            except Exception:
                return 0.0

        def _safe_str(x):
            if x is None:
                return ""
            try:
                # treat NaN float
                if isinstance(x, float) and (x != x):
                    return ""
            except Exception:
                pass
            return str(x)

        if model_name == "Salary Prediction":
            params = {
                "years_experience": _safe_num(params.get("years_experience", 0)),
                "role": _safe_str(params.get("role", "")),
                "degree": _safe_str(params.get("degree", "")),
                "company_size": _safe_str(params.get("company_size", "")),
                "location": _safe_str(params.get("location", "")),
                "level": _safe_str(params.get("level", "")),
            }
        elif model_name == "Job Fit Prediction":
            params = {
                "candidate_skills": _safe_str(params.get("candidate_skills", "")),
                "required_skills": _safe_str(params.get("required_skills", "")),
                "degree": _safe_str(params.get("degree", "")),
                "years_experience": _safe_num(params.get("years_experience", 0)),
            }
        elif model_name == "Resume Screening":
            params = {
                "resume_text": _safe_str(params.get("resume_text", "")),
                "jd_text": _safe_str(params.get("job_description_text", params.get("jd_text", ""))),
            }
        elif model_name == "Candidate Priority":
            params = {
                "referral_flag": int(_safe_num(params.get("referral_flag", 0))),
                "years_exp_band": _safe_str(params.get("years_exp_band", params.get("experience_band", ""))),
                "skills_coverage_band": _safe_str(params.get("skills_coverage_band", params.get("skills_coverage", ""))),
                "english_level": _safe_str(params.get("english_level", "")),
                "location_match": _safe_str(params.get("location_match", "")),
            }

        # Fetch candidate info if candidate_id provided
        candidate_id = params.get("candidate_id")
        if candidate_id:
            candidate_info = get_candidate(candidate_id)
            if candidate_info:
                params.update(candidate_info)

        # Map model_name to your actual API functions
        api_map = {
            "Job Fit Prediction": "check_job_fit",
            "Salary Prediction": "predict_salary",
            "Resume Screening": "screen_resume",
            "Candidate Priority": "get_priority"
        }

        func_name = api_map.get(model_name)
        if func_name and func_name in tools:
            try:
                result = tools[func_name](**params)
            except Exception as e:
                result = {"error": str(e)}
        else:
            # If LLM returned a normal answer field, surface it; otherwise report the issue
            fallback_answer = llm_data.get("answer") or llm_data.get("response") or llm_data.get("message")
            if fallback_answer:
                print("HR AI Agent Output:")
                print(str(fallback_answer).strip())
                continue
            result = {"error": "Unknown model or missing tool"}

        print("HR AI Agent Output:")
        note = llm_data.get("note")
        if note:
            print(json.dumps({"result": result, "note": note}, indent=2))
        else:
            print(json.dumps(result, indent=2))


# -----------------------------
# Run the agent
# -----------------------------
if __name__ == "__main__":
    hr_ai_agent()
