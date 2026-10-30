FROM python:3.12-slim

# Hugging Face Spaces run the container as user 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

# Download the model at build time so the Space starts faster
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; \
M.from_pretrained('yzdvai/tiny-injection-guard'); AutoTokenizer.from_pretrained('yzdvai/tiny-injection-guard')"

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
