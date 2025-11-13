import os
import sys
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import API modules to reuse their loaded models and constants
from apis import job_fit_api, salary_api, resume_api, hiring_api

# Pure-Python wrappers expected by the agent

def check_job_fit(required_skills: str = "", candidate_skills: str = "", degree: str = "", years_experience: float = 0):
    req_set = set([s.strip() for s in str(required_skills).split(",") if s.strip()])
    cand_set = set([s.strip() for s in str(candidate_skills).split(",") if s.strip()])
    overlap = len(req_set & cand_set)
    num_required_skills = len(req_set)
    overlap_ratio = overlap / num_required_skills if num_required_skills > 0 else 0

    df_input = pd.DataFrame([{
        "degree": degree,
        "years_experience": years_experience,
        "overlap": overlap,
        "num_required_skills": num_required_skills,
        "overlap_ratio": overlap_ratio
    }])

    prediction = job_fit_api.job_fit_model.predict(df_input)
    proba = job_fit_api.job_fit_model.predict_proba(df_input)[:, 1]
    return {
        "prediction": int(prediction[0]),
        "fit_probability": float(proba[0])
    }


def predict_salary(years_experience=0, role="", degree="", company_size="", location="", level=""):
    def _safe_num(x):
        try:
            v = float(x)
            if np.isnan(v):
                return 0.0
            return v
        except Exception:
            return 0.0

    def _safe_str(x):
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        return str(x)

    input_df = pd.DataFrame([{
        "years_experience": _safe_num(years_experience),
        "role": _safe_str(role),
        "degree": _safe_str(degree),
        "company_size": _safe_str(company_size),
        "location": _safe_str(location),
        "level": _safe_str(level),
    }])

    # Ensure no NaNs remain
    input_df = input_df.fillna({
        "years_experience": 0.0,
        "role": "",
        "degree": "",
        "company_size": "",
        "location": "",
        "level": "",
    })

    prediction = salary_api.salary_model.predict(input_df)
    return {"prediction": float(prediction[0])}


def screen_resume(resume_text: str = "", jd_text: str = ""):
    X_resume = resume_api.tfidf_resume.transform([resume_text]).toarray()
    X_jd = resume_api.tfidf_jd.transform([jd_text]).toarray()
    X = np.hstack((X_resume, X_jd))
    pred = resume_api.resume_model.predict(X)
    return {"advance_prediction": int(pred[0] > 0.5)}


def get_priority(referral_flag=0, years_exp_band="", skills_coverage_band="", english_level="", location_match=""):
    df = pd.DataFrame([{
        "referral_flag": referral_flag,
        "years_exp_band": years_exp_band,
        "skills_coverage_band": skills_coverage_band,
        "english_level": english_level,
        "location_match": location_match,
    }])

    # One-hot encode to align with training columns in the API
    df_encoded = pd.get_dummies(df, columns=["years_exp_band", "skills_coverage_band", "english_level", "location_match"]) 
    for col in hiring_api.X_train_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[hiring_api.X_train_columns]

    prediction = hiring_api.hiring_model.predict(df_encoded)[0]
    return {"prediction": int(prediction)}

tools = {
    "check_job_fit": check_job_fit,
    "predict_salary": predict_salary,
    "screen_resume": screen_resume,
    "get_priority": get_priority,
    # add other tools as needed
}
