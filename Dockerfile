FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies for headless OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y torchao || true

# Copy standalone submission notebook, documentation, and evaluation artifacts
COPY EdgeDiff_CV_Hackathon_P3_Submission.ipynb /app/
COPY EdgeDiff_CV_Hackathon_P3_Upgraded.ipynb /app/
COPY EdgeDiff_CV_Master_Upgraded.md /app/
COPY README.md /app/
COPY eval_P3_T01.md /app/

EXPOSE 7860

# Execute the standalone submission notebook (No Google Drive required)
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "EdgeDiff_CV_Hackathon_P3_Submission.ipynb", "--output", "EdgeDiff_Executed.ipynb"]
