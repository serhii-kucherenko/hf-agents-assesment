import os
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import requests

from agent import GaiaAgent

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def download_task_file(api_url: str, task_id: str, file_name: str, download_dir: Path) -> str | None:
    if not file_name:
        return None

    file_url = f"{api_url}/files/{task_id}"
    destination = download_dir / file_name
    print(f"Downloading file for task {task_id}: {file_url}")

    response = requests.get(file_url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return str(destination.resolve())


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Fetch all questions, run the GAIA agent, submit answers, and display results."""
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = GaiaAgent()
    except Exception as error:
        print(f"Error instantiating agent: {error}")
        return f"Error initializing agent: {error}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as error:
        return f"Error fetching questions: {error}", None
    except requests.exceptions.JSONDecodeError as error:
        return f"Error decoding server response for questions: {error}", None
    except Exception as error:
        return f"An unexpected error occurred fetching questions: {error}", None

    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")

    with tempfile.TemporaryDirectory(prefix="gaia_files_") as temp_dir:
        download_dir = Path(temp_dir)
        for item in questions_data:
            task_id = item.get("task_id")
            question_text = item.get("question")
            file_name = item.get("file_name") or ""

            if not task_id or question_text is None:
                print(f"Skipping item with missing task_id or question: {item}")
                continue

            file_path = None
            try:
                if file_name:
                    file_path = download_task_file(api_url, task_id, file_name, download_dir)
                submitted_answer = agent(question_text, file_path=file_path)
                answers_payload.append(
                    {"task_id": task_id, "submitted_answer": submitted_answer}
                )
                results_log.append(
                    {
                        "Task ID": task_id,
                        "Question": question_text,
                        "Submitted Answer": submitted_answer,
                    }
                )
            except Exception as error:
                print(f"Error running agent on task {task_id}: {error}")
                results_log.append(
                    {
                        "Task ID": task_id,
                        "Question": question_text,
                        "Submitted Answer": f"AGENT ERROR: {error}",
                    }
                )

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    print(
        f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    )

    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as error:
        error_detail = f"Server responded with status {error.response.status_code}."
        try:
            error_json = error.response.json()
            error_detail += f" Detail: {error_json.get('detail', error.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {error.response.text[:500]}"
        return f"Submission Failed: {error_detail}", pd.DataFrame(results_log)
    except requests.exceptions.Timeout:
        return "Submission Failed: The request timed out.", pd.DataFrame(results_log)
    except requests.exceptions.RequestException as error:
        return f"Submission Failed: Network error - {error}", pd.DataFrame(results_log)
    except Exception as error:
        return f"An unexpected error occurred during submission: {error}", pd.DataFrame(
            results_log
        )


with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1. Add your `HF_TOKEN` secret in Space settings.
        2. Log in to Hugging Face with the button below.
        3. Click **Run Evaluation & Submit All Answers** to score your agent on the 20 GAIA questions.

        Your agent uses `smolagents` with web search, Wikipedia, file reading, audio transcription,
        image analysis, and Python code execution.
        """
    )

    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False
    )
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
    )


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"SPACE_HOST found: {space_host_startup}")
    if space_id_startup:
        print(f"SPACE_ID found: {space_id_startup}")
        print(f"Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")

    demo.launch(debug=True, share=False)
