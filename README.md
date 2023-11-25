# Tensorflow learning

Course Info: [check here](course-link.txt)

### Environment Setup

1. [Install Conda](https://docs.anaconda.com/anaconda/install/) from the official Anaconda distribution page.

2. Then run `conda env create -f conda.yaml` to create a new conda environment from the `conda.yaml` file.

3. Activate conda environment `conda activate tensorflow-learning`

### Instruction to serve tensorflow model

1. Install `tensorflow_model_server` command using following commands

```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install tensorflow-model-server
```

2. Then serve your model using below command

`tensorflow_model_server --port=8501 --rest_api_port=8000 --model_name=model_name --model_base_path=exported_models_dir
`
