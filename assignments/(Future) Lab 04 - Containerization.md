# Lab Assignment 04: Docker Containerization

## Overview

In this lab, we will containerize your Flask API and MLflow tracking server using Docker. Up until now, your project has run directly on your machine (or in a Codespace), requiring manual setup of virtual environments, dependencies, and configurations. This works for development, but it creates problems in production:

- **"It works on my machine"**: Your project might work perfectly on your laptop but fail on a teammate's computer or a deployment server due to different Python versions, missing system libraries, or conflicting dependencies.
- **Manual setup is error-prone**: Every new environment requires running `python -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt`, `dvc pull`, etc. If any step is missed or done incorrectly, the application breaks.
- **Reproducibility is fragile**: Six months from now, will your project still run? What if a dependency updates and breaks compatibility?

**Docker solves these problems** by packaging your entire application—code, dependencies, Python runtime, system libraries, and configuration—into a single, portable unit called a **container**. A container is like a lightweight virtual machine that runs the same way on your laptop, your teammate's laptop, a cloud server, or anywhere else Docker is installed.

In this lab, you will:
1. Create **two Docker containers**: one for your Flask API (serving predictions) and one for MLflow (tracking experiments).
2. Use **Docker Compose** to orchestrate these containers so they can communicate with each other.
3. Learn production-grade practices like **logging**, **volume mounting**, and **image publishing**.
4. Prepare your application for cloud deployment in Lab 05.

### Why Containerization Matters for MLOps

Containerization is the foundation of modern MLOps. Here's why:

- **Reproducibility**: Your container will run the same way today, tomorrow, and five years from now, regardless of what else is installed on the host machine.
- **Portability**: You can deploy your container to AWS, Azure, Google Cloud, your company's data center, or even a Raspberry Pi, and it will work the same way.
- **Isolation**: Multiple projects can run on the same server without conflicting dependencies (e.g., one project uses scikit-learn 1.3, another uses scikit-learn 1.5).
- **Scalability**: In production, you can run 1 container or 1,000 containers of your API, depending on traffic, using orchestration tools like Kubernetes.
- **Collaboration**: A teammate can run your entire project with a single command (`docker-compose up`) instead of following a 20-step setup guide.

### Learning Objectives

By the end of this lab, you will be able to:

- Understand core Docker concepts: images, containers, layers, and registries.
- Write production-ready Dockerfiles for Python applications.
- Use `.dockerignore` to optimize Docker build performance.
- Implement enhanced logging practices for containerized applications.
- Create a `docker-compose.yml` file to orchestrate multi-container applications.
- Configure Docker networks for inter-service communication.
- Use volume mounting to persist data and logs.
- Publish Docker images to Docker Hub for sharing and deployment.
- Test containerized applications both manually and automatically.
- Prepare your application for cloud deployment (Lab 05).

---

## What is Docker? A Quick Primer

Before we dive into the lab, let's clarify some key Docker concepts.

### Docker Images vs. Containers

- **Docker Image**: A blueprint or template for your application. It's a read-only file that contains your code, dependencies, and runtime. Think of it like a recipe.
- **Docker Container**: A running instance of an image. Think of it like a cake you baked from the recipe. You can create multiple containers (cakes) from the same image (recipe).

**Example**: If you have a `churn-prediction-api:latest` image, you can run 10 containers from it simultaneously, each serving API requests independently.

### Dockerfile

A **Dockerfile** is a text file containing instructions to build a Docker image. It's like a step-by-step recipe:

```dockerfile
FROM python:3.12-slim          # Start with a base image (Python 3.12)
WORKDIR /app                   # Set the working directory inside the container
COPY requirements.txt .        # Copy requirements file into the container
RUN pip install -r requirements.txt  # Install dependencies
COPY src/ src/                 # Copy your source code
CMD ["python", "-m", "src.app"] # Command to run when the container starts
```

Each line in a Dockerfile creates a **layer**. Docker caches layers, so if you change your code but not your dependencies, Docker can reuse the cached dependency layer, making builds faster.

### Why Multiple Dockerfiles?

In this lab, you'll create two Dockerfiles:

- **`Dockerfile.mlapp`**: For your Flask API (serves predictions)
- **`Dockerfile.mlflow`**: For MLflow UI (tracks experiments)

Why not one Dockerfile? Because these are **two different services** with different purposes:
- The API needs your code, models, and Flask.
- MLflow just needs the MLflow package and doesn't need your code.

Separating them follows the **single responsibility principle** and makes your system more modular and maintainable.

### Docker Compose

**Docker Compose** is a tool for defining and running multi-container applications. Instead of running two `docker run` commands manually (one for your API, one for MLflow), you define both services in a single `docker-compose.yml` file and start everything with one command: `docker-compose up`.

Docker Compose also handles:
- **Networking**: Automatically creates a network so your containers can talk to each other (e.g., your API can send experiment data to MLflow).
- **Volume mounting**: Lets you share data between your host machine and containers (e.g., models, logs).
- **Environment variables**: Centralized configuration for all services.
- **Startup order**: Ensures MLflow starts before your API tries to connect to it.

### Containers vs. Virtual Machines

You might wonder: "Isn't Docker just like running a virtual machine?"

No. Containers are much lighter:

| Virtual Machine | Docker Container |
|----------------|------------------|
| Runs a full OS (gigabytes) | Shares the host OS kernel (megabytes) |
| Slow to start (minutes) | Fast to start (seconds) |
| Heavy resource usage | Lightweight |
| Strong isolation | Process-level isolation |

A container is more like an isolated process on your machine, not a full virtual machine. This is why you can run 100 containers on a laptop but only 2-3 VMs.

---

## Task 1: Docker Installation and Setup

Docker is pre-installed in GitHub Codespaces, which is the primary environment for this course. However, if you're working on your local machine, you'll need to install Docker Desktop (for Windows/macOS) or Docker Engine (for Linux).

### 1.1: Verify Docker in GitHub Codespaces (Primary Method)

GitHub Codespaces comes with Docker pre-installed and ready to use. Let's verify it's working.

1. **Open your Codespace** for this project (or create a new one if needed).

2. **Open the terminal** in VS Code (it's usually at the bottom of the screen).

3. **Check Docker version**:

   ```sh
   docker --version
   ```

   **Expected Output** (version numbers may vary):
   ```
   Docker version 24.0.6, build ed223bc
   ```

4. **Check Docker Compose version**:

   ```sh
   docker-compose --version
   ```

   **Expected Output**:
   ```
   Docker Compose version v2.21.0
   ```

   If both commands return version information, you're ready to proceed! Docker and Docker Compose are installed and working.

5. **Test Docker with a "Hello World" container**:

   ```sh
   docker run hello-world
   ```

   **What this does**: Docker downloads a tiny test image and runs it. The container prints a welcome message and exits.

   **Expected Output**:
   ```
   Hello from Docker!
   This message shows that your installation appears to be working correctly.
   ...
   ```

   **Understanding the Output**: This command did a lot behind the scenes:
   1. Docker checked if the `hello-world` image exists locally (it doesn't).
   2. Docker downloaded (`pulled`) the image from Docker Hub (a public registry of images).
   3. Docker created a container from the image and ran it.
   4. The container printed a message and exited.

6. **Clean up the test container** (optional but good practice):

   ```sh
   docker rm $(docker ps -aq)
   ```

   This removes all stopped containers. Don't worry; it only removes containers, not images. We'll cover Docker cleanup commands later in the lab.

---

### 1.2: 💡 Installing Docker on Your Local Machine (Optional)

If you prefer to work on your local machine instead of Codespaces, you'll need to install Docker Desktop or Docker Engine.

#### For Windows and macOS:

1. **Download Docker Desktop**:
   - Go to [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
   - Download the installer for your operating system (Windows or macOS).

2. **Install Docker Desktop**:
   - Run the installer and follow the setup wizard.
   - **macOS users**: If you have an Apple Silicon Mac (M1/M2/M3), ensure you download the "Apple Silicon" version for better performance.

3. **Start Docker Desktop**:
   - Open Docker Desktop from your Applications folder (macOS) or Start menu (Windows).
   - Wait for the Docker icon in your system tray to show a green status (indicating Docker is running).

4. **Verify installation**:
   Open a terminal (macOS) or PowerShell (Windows) and run:

   ```sh
   docker --version
   docker-compose --version
   ```

   You should see version information for both commands.

#### For Linux:

Linux users should install Docker Engine (the command-line version) instead of Docker Desktop.

**Ubuntu/Debian**:

```sh
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
docker --version
docker compose version  # Note: On Linux, it's "docker compose" not "docker-compose"
```

**Post-installation (Linux only)**:

By default, Docker requires `sudo` to run. To run Docker without `sudo`, add your user to the `docker` group:

```sh
sudo usermod -aG docker $USER
newgrp docker  # Or log out and log back in
```

Now verify you can run Docker without `sudo`:

```sh
docker run hello-world
```

---

### 1.3: Important Notes Before Proceeding

#### Docker Disk Space

Docker images and containers can take up significant disk space. In Codespaces, you have limited storage (32GB), so be mindful:

- Each Python base image is ~100-200 MB.
- Your project images will be 300-500 MB each.
- Stopped containers and unused images accumulate over time.

**Good practices**:
- Remove stopped containers regularly: `docker container prune`
- Remove unused images: `docker image prune`
- Remove everything (nuclear option): `docker system prune -a` (use with caution!)

We'll cover cleanup commands at the end of the lab.

#### Docker and Port Conflicts

Docker containers expose ports (e.g., 5000 for your API, 5001 for MLflow). If you're already running services on these ports (like a local Flask app or MLflow UI), you'll get a "port already in use" error.

**Solution**: Stop any local services before running Docker containers, or change the host port mapping in `docker-compose.yml` (we'll cover this in Task 6).

#### Docker in Codespaces vs. Local

- **Codespaces**: You'll need to forward ports to access services in your browser (just like in Lab 03).
- **Local**: Services are immediately accessible at `http://localhost:5000` (no port forwarding needed).

---

### 1.4: Understanding Your Current Project Files

Before we start creating Dockerfiles, let's take inventory of what files your project needs to run. Understanding this will help you decide what to copy into your Docker containers.

**Essential files for the API container**:
- `src/` – Your Python source code (app.py, train.py, predict.py, etc.)
- `models/` – Pre-trained model files (model_v1.pkl, model_v2.pkl)
- `data/processed/` – Preprocessing pipeline and label encoder
- `configs/` – YAML configuration files
- `requirements.txt` – Python dependencies

**Files you do NOT need in containers**:
- `.venv/` – Virtual environment (Docker creates its own isolated environment)
- `.git/` – Git history (not needed at runtime)
- `mlruns/` – MLflow experiments (will be volume-mounted, not copied)
- `tests/` – Test files (not needed in production containers)
- `notebooks/` – Jupyter notebooks (for exploration, not deployment)
- `data/raw/` – Raw data (not needed by the API)
- `__pycache__/`, `*.pyc` – Python cache files
- `.dvc/` – DVC metadata (not needed in containers)

We'll use a `.dockerignore` file to exclude these unnecessary files from the Docker build context. This makes builds faster and images smaller.

---

## What's Next?

Now that Docker is installed and you understand the basics, we're ready to start containerizing your application. In the next section, we'll create:

1. **`Dockerfile.mlapp`** – For your Flask API
2. **`Dockerfile.mlflow`** – For MLflow UI
3. **`.dockerignore`** – To optimize builds

Continue to **Task 2: Creating the ML Application Dockerfile**.

---

**[PART 1 END - TO BE CONTINUED]**
