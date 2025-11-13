import os
import re
import json
import warnings
from flask import Blueprint, request, jsonify, render_template
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
try:
    from agent.helper import get_candidate
    from agent.tools import tools
    from agent.prompt import system_prompt
except ModuleNotFoundError:
    import sys
    CURRENT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from helper import get_candidate
    from tools import tools
    from prompt import system_prompt

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Initialize Azure AI Inference client
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1"
load_dotenv()
_token = os.environ.get("GITHUB_TOKEN")
if not _token:
    raise ValueError("GITHUB_TOKEN is not set. Add it to your .env or environment.")
_client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(_token))

agent_bp = Blueprint("agent_bp", __name__)


def _generate_response(prompt: str) -> str:
    resp = _client.complete(
        messages=[
            SystemMessage(system_prompt),
            UserMessage(prompt),
        ],
        temperature=1,
        top_p=1,
        model=model_name,
    )
    return resp.choices[0].message.content


def _process_user_input(user_input: str):
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
    llm_output = _generate_response(prompt)

    try:
        llm_data = json.loads(llm_output)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", llm_output)
        if not match:
            return {"mode": "answer", "output": llm_output.strip()}
        try:
            llm_data = json.loads(match.group(0))
        except Exception:
            return {"mode": "answer", "output": llm_output.strip()}

    model_name_local = llm_data.get("model")
    params = llm_data.get("params", {})

    if not model_name_local:
        direct_answer = llm_data.get("answer") or llm_data.get("response") or llm_data.get("message")
        if direct_answer:
            return {"mode": "answer", "output": str(direct_answer).strip()}

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
    if isinstance(model_name_local, str):
        key = model_name_local.strip().lower()
        model_name_local = aliases.get(key, model_name_local)

    def _safe_num(x):
        try:
            v = float(x)
            if isinstance(v, float) and (v != v):
                return 0.0
            return v
        except Exception:
            return 0.0

    def _safe_str(x):
        if x is None:
            return ""
        try:
            if isinstance(x, float) and (x != x):
                return ""
        except Exception:
            pass
        return str(x)

    if model_name_local == "Salary Prediction":
        params = {
            "years_experience": _safe_num(params.get("years_experience", 0)),
            "role": _safe_str(params.get("role", "")),
            "degree": _safe_str(params.get("degree", "")),
            "company_size": _safe_str(params.get("company_size", "")),
            "location": _safe_str(params.get("location", "")),
            "level": _safe_str(params.get("level", "")),
        }
    elif model_name_local == "Job Fit Prediction":
        params = {
            "candidate_skills": _safe_str(params.get("candidate_skills", "")),
            "required_skills": _safe_str(params.get("required_skills", "")),
            "degree": _safe_str(params.get("degree", "")),
            "years_experience": _safe_num(params.get("years_experience", 0)),
        }
    elif model_name_local == "Resume Screening":
        params = {
            "resume_text": _safe_str(params.get("resume_text", "")),
            "jd_text": _safe_str(params.get("job_description_text", params.get("jd_text", ""))),
        }
    elif model_name_local == "Candidate Priority":
        params = {
            "referral_flag": int(_safe_num(params.get("referral_flag", 0))),
            "years_exp_band": _safe_str(params.get("years_exp_band", params.get("experience_band", ""))),
            "skills_coverage_band": _safe_str(params.get("skills_coverage_band", params.get("skills_coverage", ""))),
            "english_level": _safe_str(params.get("english_level", "")),
            "location_match": _safe_str(params.get("location_match", "")),
        }

    candidate_id = params.get("candidate_id")
    if candidate_id:
        candidate_info = get_candidate(candidate_id)
        if candidate_info:
            params.update(candidate_info)

    api_map = {
        "Job Fit Prediction": "check_job_fit",
        "Salary Prediction": "predict_salary",
        "Resume Screening": "screen_resume",
        "Candidate Priority": "get_priority"
    }

    func_name = api_map.get(model_name_local)
    if func_name and func_name in tools:
        try:
            result = tools[func_name](**params)
        except Exception as e:
            result = {"error": str(e)}
    else:
        fallback_answer = llm_data.get("answer") or llm_data.get("response") or llm_data.get("message")
        if fallback_answer:
            return {"mode": "answer", "output": str(fallback_answer).strip()}
        result = {"error": "Unknown model or missing tool"}

    note = llm_data.get("note")
    explain_input = {
        "user_question": user_input,
        "model": model_name_local,
        "params": params,
        "result": result,
        "note": note,
    }
    explanation_prompt = f"""
You are an assistant that reformulates model outputs into a concise, user-friendly explanation in the same language as the user.

User question:
{user_input}

Chosen model: {model_name_local}
Parameters (normalized):
{json.dumps(params, ensure_ascii=False)}

Model result:
{json.dumps(result, ensure_ascii=False)}

Note (optional): {note}

Instructions:
- Explain clearly and succinctly what the prediction/result means for the user.
- Include the key numbers or probabilities if present.
- Use the user's language and avoid JSON in the answer.
- If there is an error, explain it briefly and suggest how to fix the inputs.
"""
    explanation_text = _generate_response(explanation_prompt)
    return {"mode": "answer", "output": explanation_text.strip()}


@agent_bp.route('/', methods=['GET'])
def page():
    return render_template('agent.html')


@agent_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    if not message:
        return "message is required", 400, {"Content-Type": "text/plain; charset=utf-8"}
    result = _process_user_input(message)
    # Build a plain text response
    if isinstance(result, dict):
        mode = result.get("mode")
        if mode == "answer":
            text = str(result.get("output", "")).strip()
        elif mode == "tool":
            text = json.dumps(result.get("output", {}), ensure_ascii=False)
        else:
            text = json.dumps(result, ensure_ascii=False)
    else:
        text = str(result)
    return text, 200, {"Content-Type": "text/plain; charset=utf-8"}