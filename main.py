"""from flask import Flask, render_template
from apis.salary_api import salary_bp
from apis.job_fit_api import job_fit_bp
from apis.resume_api import resume_bp
from apis.hiring_api import hiring_bp
from agent.agent_api import agent_bp

app = Flask(__name__)

# Register blueprints
#app.register_blueprint(salary_bp, url_prefix='/salary')
#app.register_blueprint(job_fit_bp, url_prefix='/job_fit')
#app.register_blueprint(resume_bp, url_prefix='/resume_screen')
#app.register_blueprint(hiring_bp, url_prefix="/candidate_priority")
app.register_blueprint(agent_bp, url_prefix="/agent")

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)"""


import os
from flask import Flask, render_template
from agent.agent_api import agent_bp

app = Flask(__name__)

app.register_blueprint(agent_bp, url_prefix="/agent")

@app.route("/")
def index():
    return "HR AI Agent is running ✅"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    
