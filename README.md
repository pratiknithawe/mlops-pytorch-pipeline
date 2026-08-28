# MLOps PyTorch Pipeline

End-to-end PyTorch CIFAR-10 image classification with Docker and Kubernetes.

## Tools Stack
Python 3.11, PyTorch 2.5.1, TorchVision 0.20.1, FastAPI, Docker, Kubernetes, GitHub Actions.

## Architecture
```text

GitHub -> 
    GitHub Actions -> 
                Docker -> 
                    Training Job -> 
                            CIFAR-10 -> 
                                    Checkpoint PVC -> 
                                            Serving Deployment(2 Replicas) -> 
                                                                        FastAPI :8080 -> 
                                                                                    Service :80
```

## Test locally
```bash
pip install -r requirements/train.txt
pip install pytest==8.3.4
pytest -q
```

## Docker
```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/checkpoints:/app/checkpoints" mlops-train:v1
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 -v "$(pwd)/checkpoints:/app/checkpoints:ro" mlops-serve:v1
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```
Windows PowerShell uses `${PWD}` instead of `$(pwd)`.

## Kubernetes
For Minikube: `minikube image load mlops-train:v1` and `minikube image load mlops-serve:v1`.

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
kubectl get jobs,pods -n ml-training
kubectl logs job/pytorch-training -n ml-training
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl get pods -n ml-training
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Git workflow
Use `develop` plus feature branches and merge each feature through a Pull Request (PR). This assignment asks atleast four merged PRs.

## Repository Structure

```text
mlops-pytorch-pipeline/
├── .github/workflows/     
├── configs/              
├── data/                  
├── checkpoints/         
├── docker/       
├── k8s/            
├── requirements/       
├── scripts/               
├── src/              
└── tests/         
```


