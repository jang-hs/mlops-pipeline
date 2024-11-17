cd train_model
docker build -f Dockerfile -t resnet18-mnist-train:1.0 . && minikube image load resnet18-mnist-train:1.0
cd ..

cd inference_model
docker build -f Dockerfile -t resnet18-mnist-inference:1.0 . && minikube image load resnet18-mnist-inference:1.0
cd ..