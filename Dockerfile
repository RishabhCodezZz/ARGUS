# Stage 7 deployment image. NOT `adk deploy cloud_run`'s auto-generated
# Dockerfile — see the "Why NOT adk deploy cloud_run" section of the Stage 7
# plan for the five landmines that command's copytree/gitignore assumptions
# would have hit (mock data resolved as a sibling of argus/, requirements.txt
# expected inside argus/, .gitignore read from the wrong directory, etc.).
# Writing this by hand means `COPY . .` from the repo root is correct by
# construction, and .dockerignore below is the ONLY thing standing between
# a visitor and argus/.env. Disclosed limitation: this was never verified
# with a local `docker build` (a Windows-specific Docker Desktop bug made
# local builds impossible in this environment) — the first real build of
# this Dockerfile happens on the deploy platform itself.

FROM python:3.12-slim

WORKDIR /app

# Dependencies first so Docker's layer cache skips reinstalling on every
# code-only change — only requirements.txt changes invalidate this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# argus/.env is excluded by .dockerignore below — this image carries no
# Gemini key of its own by design (see server.py's module docstring).
COPY . .

# EXPOSE is informational only — server.py reads the $PORT environment
# variable at runtime (falling back to 7860 only if unset), so this image
# runs correctly regardless of which port a given platform actually
# assigns (e.g. Render defaults to 10000, Cloud Run assigns its own).
EXPOSE 7860

CMD ["python", "server.py"]
