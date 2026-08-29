MLOps PyTorch Pipeline

An end-to-end MLOps pipeline for CIFAR-10 image classification using PyTorch, FastAPI, Docker and Kubernetes.

This assignment demonstrates how a machine-learning model can be trained, packaged into Docker containers, deployed to Kubernetes, exposed through a REST API, and managed using a structured Git/GitHub pull-request workflow.

---

1. Assignment Overview

This assignment implements a complete machine-learning workflow for image classification using the CIFAR-10 dataset.

The pipeline consists of:

1. PyTorch model training
2. Model checkpoint generation
3. Docker containerization of training and serving
4. Kubernetes-based training
5. Persistent storage for model checkpoints
6. Kubernetes-based model serving
7. FastAPI REST API
8. Kubernetes Service for exposing the model
9. Feature-branch and Pull Request based Git workflow

The final system separates the training environment from the serving environment, allowing the trained model to be packaged and deployed independently.

---

2. assignment Architecture

                         ┌─────────────────────┐
                         │         GitHub             │
                         │     Repository + PRs       │
                         └─────────┬──────────┘
                                      │
                                      ▼
                         ┌─────────────────────┐
                         │      GitHub Actions        │
                         │     Automated Tests        │
                         └───────┬─────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │         Docker             │
                         │                            │
                         │       mlops-train:v1       │
                         │       mlops-serve:v1       │
                         └──────────┬──────────┘
                                        │
                    ┌──────────────┴────────────────┐
                    │                                         │
                    ▼                                         ▼
          ┌───────────────────┐           ┌────────────────────┐
          │       Kubernetes        │           │        Kubernetes         │
          │       Training Job      │           │        Serving Deployment │
          │                         │           │                           │
          │       mlops-train:v1    │           │        mlops-serve:v1     │
          └─────────┬─────────┘           │        2 Replicas         │
                       │                     └─────────┬──────────┘
                       ▼                                  │
          ┌───────────────────┐                    ▼
          │      PersistentVolume   │             ┌────────────────┐
          │       Claim             │             │      Kubernetes     │
          │                         │             │      Service        │
          │       Checkpoints       │             │      model-serving  │
          └─────────┬─────────┘             └───────┬────────┘
                       │                                     │
                       └────────────┬───────────────┘
                                        ▼
                         ┌─────────────────────┐
                         │           FastAPI          │
                         │            :8080           │
                         │                            │
                         │        GET  /health        │
                         │        POST /predict       │
                         └─────────────────────┘

---

3. Technology Stack

Component                      |              Technology
Programming Language           |              Python 3.11
Deep Learning Framework        |              PyTorch 2.5.1
Computer Vision                |              TorchVision 0.20.1
Dataset                        |              CIFAR-10
API Framework                  |              FastAPI
Containerization               |              Docker
Orchestration                  |              Kubernetes
Kubernetes CLI                 |              kubectl
Local Kubernetes               |              Docker Desktop Kubernetes
Testing                        |              pytest
Version Control                |              Git + GitHub
API Port                       |              8080 
Kubernetes Service Port        |              80

---

4. Repository Structure

mlops-pytorch-pipeline/
├── .github/
│   └── workflows/
├── configs/
│   └── training_config.yaml
├── data/
│   └── cifar-10-batches-py
├── checkpoints/
│   └── classifier_v1.pt
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
├── scripts/
│   └── create_test_image.py
├── src/
│    ├── dataset.py
│    ├──serve.py
│    ├──train.py
│    └──model.py
├── tests/
│   └── testr_model.py
├── .gitignore
├── README.md
└── test_image.png

---

5. Prerequisites

Install the following before running the assignment:

- Python 3.11
- Git
- Docker Desktop
- Kubernetes
- kubectl

Verify the installations:

python --version
docker --version
kubectl version --client

---

6. Clone the Repository

git clone <repository-url>
cd mlops-pytorch-pipeline

Create and activate a Python virtual environment.

Windows PowerShell

python -m venv .venv
.\.venv\Scripts\Activate.ps1

---

7. Install Python Dependencies

Install the training dependencies:

pip install -r requirements/train.txt

Install pytest if it is not already included:

pip install pytest==8.3.4

---

8. Run Tests Locally

Run the assignment's test suite:

pytest -q

A successful test run confirms that the implemented assignment components pass the available automated tests.

---

9. PyTorch Training

The training component uses PyTorch to train an image-classification model on CIFAR-10.

The training process is responsible for:

1. Loading CIFAR-10 data
2. Preparing the input images
3. Creating the PyTorch model
4. Running the training loop
5. Evaluating the model
6. Saving the trained model checkpoint

The training code is contained within the assignment's "src/" and "scripts/" components.

---

10. Dockerized Training

The training environment is packaged separately using:

docker/Dockerfile.train

Build the training image:

docker build -f docker/Dockerfile.train -t mlops-train:v1 .

Verify the image:

docker images | Select-String "mlops-train"

The resulting image is:

mlops-train:v1

A local training container can be executed with mounted data and checkpoint directories.

PowerShell

docker run --rm `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\checkpoints:/app/checkpoints" `
  mlops-train:v1

The checkpoint directory is used to persist the trained model outside the container.

---

11. Dockerized Model Serving

The model-serving environment is packaged separately using:

docker/Dockerfile.serve

Build the serving image:

docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

Verify the image:

docker images | Select-String "mlops-serve"

The resulting image is:

mlops-serve:v1

Run the serving container:

docker run --rm `
  -p 8080:8080 `
  -v "${PWD}\checkpoints:/app/checkpoints:ro" `
  mlops-serve:v1

The checkpoint is mounted read-only into the serving container so that the serving application can load the trained model without modifying it.

---

12. FastAPI Model Serving

The trained model is exposed through a FastAPI application.

The API listens on:

http://localhost:8080

Health Check

Send:

curl.exe http://localhost:8080/health

The endpoint is used to verify that the serving application is running.

Prediction

The "/predict" endpoint accepts an image file.

Example:

curl.exe -X POST `
  http://localhost:8080/predict `
  -F "image=@test_image.png"

Example response:

{
  "predicted_class": "airplane",
  "probabilities": [
    {
      "class": "airplane",
      "probability": 0.500876
    }
  ]
}

The exact prediction and probabilities depend on the trained model and input image.

---

13. Kubernetes Deployment

The assignment provides Kubernetes manifests under:

k8s/

The Kubernetes resources are deployed into the:

ml-training

namespace.

---

13.1 Create the Namespace

kubectl apply -f k8s/namespace.yaml

Verify:

kubectl get namespace ml-training

---

13.2 Create the Training ConfigMap

kubectl apply -f k8s/configmap.yaml

Verify:

kubectl get configmap -n ml-training

---

14. Kubernetes Training Job

The training workload is defined in:

k8s/training-job.yaml

This manifest creates:

- A PersistentVolumeClaim
- A Kubernetes Job
- Training configuration
- Persistent storage mounts
- CPU and memory resource requirements

The training container uses:

mlops-train:v1

The Job mounts persistent storage at:

/app/data
/app/checkpoints

The Kubernetes Job also specifies:

resources:
  requests:
    cpu: "2"
    memory: 2Gi
  limits:
    cpu: "2"
    memory: 2Gi

This makes the resource requirements explicit rather than leaving them unspecified.

---

14.1 Deploy the Training Job

kubectl apply -f k8s/training-job.yaml

Check the Job:

kubectl get jobs -n ml-training

Check the Pods:

kubectl get pods -n ml-training

View the training logs:

kubectl logs job/pytorch-training -n ml-training

Describe the Job when troubleshooting:

kubectl describe job pytorch-training -n ml-training

---

15. Persistent Model Checkpoints

The training Job uses a PersistentVolumeClaim named:

mlops-data-checkpoints

The PVC requests:

10Gi

Persistent storage allows model checkpoints to survive independently of the lifecycle of the training container.

The checkpoint storage is mounted into the training container at:

/app/checkpoints

The serving workload can then use the trained checkpoint.

---

16. Kubernetes Model Serving

The serving deployment is defined in:

k8s/serving-deployment.yaml

The serving container uses:

mlops-serve:v1

The deployment is configured with:

2 replicas

Using multiple replicas demonstrates a basic horizontally scalable serving architecture.

Check the deployment:

kubectl get deployments -n ml-training

Check the serving Pods:

kubectl get pods -n ml-training

---

17. Kubernetes Service

The model-serving application is exposed through:

k8s/serving-service.yaml

The Kubernetes Service is named:

model-serving

The service exposes port:

80

and forwards traffic to the FastAPI application running on:

8080

Apply the Service:

kubectl apply -f k8s/serving-service.yaml

Verify:

kubectl get service -n ml-training

---

18. Access the Kubernetes API Locally

For local development, port-forward the Kubernetes Service:

kubectl port-forward svc/model-serving 8080:80 -n ml-training

The API is then available at:

http://localhost:8080

Test the health endpoint:

curl.exe http://localhost:8080/health

Test prediction:

curl.exe -X POST `
  http://localhost:8080/predict `
  -F "image=@test_image.png"

---


19. Complete Kubernetes Deployment Sequence

For a complete deployment, the following sequence can be used:

kubectl apply -f k8s/namespace.yaml

kubectl apply -f k8s/configmap.yaml

kubectl apply -f k8s/training-job.yaml

kubectl get jobs,pods -n ml-training

kubectl logs job/pytorch-training -n ml-training

kubectl apply -f k8s/serving-deployment.yaml

kubectl apply -f k8s/serving-service.yaml

kubectl apply -f k8s/hpa.yaml

kubectl get pods -n ml-training

kubectl get service -n ml-training

kubectl get hpa -n ml-training

To access the API:

kubectl port-forward svc/model-serving 8080:80 -n ml-training

Then:

curl.exe http://localhost:8080/health

and:

curl.exe -X POST `
  http://localhost:8080/predict `
  -F "image=@test_image.png"

---

20. Git and GitHub Workflow

The assignment follows a feature-branch based workflow.

The main development branch is:

develop

Features are developed on separate branches and merged into "develop" through Pull Requests.

The completed workflow consists of four major Pull Requests:

PR| Feature
PR #1| Assignment Structure and Initial Setup
PR #2| PyTorch Model Training and Serving
PR #3| Kubernetes Deployment Configuration
PR #4| Final README and Assignment Documentation

This workflow keeps each major development stage isolated and reviewable.

---

21. Pull Request Workflow

Create a feature branch:

git checkout -b feature/<feature-name>

Check changes:

git status

Stage changes:

git add <files>

Commit using a conventional commit message:

git commit -m "docs: finalize assignment README"

Push the branch:

git push -u origin feature/<feature-name>

Create a Pull Request on GitHub and merge it into:

develop

---

22. Docker Image Summary

The assignment uses two separate Docker images.

Training Image

mlops-train:v1

Purpose:

- Contains the training environment
- Installs training dependencies
- Runs PyTorch training
- Produces model checkpoints

Serving Image

mlops-serve:v1

Purpose:

- Contains the inference environment
- Loads the trained checkpoint
- Runs FastAPI
- Provides "/health"
- Provides "/predict"

Separating training and serving environments reduces unnecessary dependencies in the production inference container.

---

23. Kubernetes Resource Summary

Resource| Name| Purpose
Namespace| "ml-training"| Isolates ML workloads
ConfigMap| "training-config"| Provides training configuration
PVC| "mlops-data-checkpoints"| Persistent checkpoint storage
Job| "pytorch-training"| Runs model training
Deployment| Model serving deployment| Runs inference replicas
Service| "model-serving"| Exposes inference API
HPA| Serving HPA| Enables horizontal scaling

---

24. API Summary

Health Endpoint

GET /health

Used to determine whether the FastAPI serving application is running.

Prediction Endpoint

POST /predict

Accepts an image and returns the model's predicted CIFAR-10 class and associated probabilities.

Example request:

curl.exe -X POST `
  http://localhost:8080/predict `
  -F "image=@test_image.png"

---

25. Final assignment Workflow

The complete MLOps workflow can be summarized as:

CIFAR-10 Dataset
       │
       ▼
PyTorch Training
       │
       ▼
Model Checkpoint
       │
       ▼
Docker Training Image
       │
       ▼
Kubernetes Training Job
       │
       ▼
PersistentVolumeClaim
       │
       ▼
Trained Checkpoint
       │
       ▼
Docker Serving Image
       │
       ▼
Kubernetes Deployment
       │
       ▼
2 Serving Replicas
       │
       ▼
Kubernetes Service
       │
       ▼
FastAPI
       │
       ├── GET /health
       │
       └── POST /predict
       │
       ▼
CIFAR-10 Prediction

---

26. Conclusion

This assignment demonstrates an end-to-end MLOps workflow for a PyTorch image-classification application.

The solution separates model training from model serving, packages both environments using Docker, uses Kubernetes for workload orchestration and scalable inference, exposes predictions through FastAPI, and uses GitHub Actions and Pull Requests to support reproducible development and validation.

The final implementation therefore demonstrates the complete progression from:

Model Development
        ↓
Training
        ↓
Containerization
        ↓
Kubernetes Deployment
        ↓
API Serving
        ↓
Scalable Inference
        ↓
CI + GitHub PR Workflow

---

