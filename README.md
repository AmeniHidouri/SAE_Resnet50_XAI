# Dataset_XAI

## How to compute criterions (!!!saliency maps only!!!)

### Prerequisites

My python version: 3.10.14. You will also need to install several packages as listed in `requirements.txt` and download the model weights and dataset files to the correct directories.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repo/Dataset_XAI.git
    cd Dataset_XAI
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Install the specific version of Hugging Face Hub:

    ```bash
    pip install huggingface-hub==0.23.4
    ```

### Dataset Setup

Download and extract the dataset:

1. Download the dataset from [this link](https://drive.google.com/file/d/1bbOUJpWHnA2bckyrm_P1IIxKwe62T0q1/view?usp=sharing).
2. Ensure that the downloaded `result_dataset` folder is placed in the `Dataset_XAI` directory:

    ```
    Dataset_XAI/
    ├── result_dataset/
    ├── ...
    ```

### Model Weights Setup

Download the model weights:

1. Download the models from [Hugging Face](https://huggingface.co/RemiKaz/Dataset_XAI/tree/main).
2. Place the weights in the `Dataset_XAI/models_pkl` directory:

    ```
    Dataset_XAI/
    ├── models_pkl/
    │   ├── best_resnet50_coco.pth
    │   └── other_models.pth
    ├── ...
    ```

### Computing Criterions

To compute specific criteria for saliency maps, run the `compute_criterion.py` script with the appropriate arguments. 

For example, to compute the faithfulness criterion on the COCO dataset using GradCAM on a ResNet-50 model, run:

```bash
python compute_criterion.py --network resnet50 --expl_method GradCAM --input_dataset toy --criterion faithfulness --sub_dataset coco
```

## Roadmap v0.1

- [x] Dataset training: pascalpart (version npy)
- [x] Dataset training: monumai (version npy)
- [x] Models: Resnet50
- [x] Models: ViT
- [x] Models: CLIP CBM
- [x] Script: train_net.py
- [x] Models: CLIP zero shot
- [x] Dataset sample: pascalpart (png+json) 
- [x] Dataset sample: monumai (png+json) 
- [x] Explanation: LIME image
- [x] Explanation: SHAP image
- [x] Explanation: GradCAM image
- [x] Explanation: CLIP-QDA sample
- [x] Explanation: CBM-lime (CLIP-QDA)
- [x] Explanation: CBM-shap (CLIP-QDA)
- [x] Script: compute_explanation.py
- [x] Utils: Perturbate image
- [x] Script: make_samples.py

## Roadmap v0.2

- [x] Explanation: LIME image (ViT)
- [x] Explanation: SHAP image (ViT)
- [x] Explanation: GradCAM image (ViT)
- [x] Explanation: LIME image (CLIP-zero-shot)
- [x] Explanation: SHAP image (CLIP-zero-shot)
- [x] Explanation: GradCAM image (CLIP-zero-shot)
- [x] Script: train_net.py (option train from concepts)
- [x] Models: LaBo
- [x] Model: CBM_supervised
- [x] Explanation: HiResCAM
- [x] Explanation: GradCAMElementWise
- [x] Explanation: GradCAM++
- [x] Explanation: XGradCAM
- [x] Explanation: AblationCAM
- [x] Explanation: ScoreCAM
- [x] Explanation: EigenCAM
- [x] Explanation: EigenGradCAM
- [x] Explanation: LayerCAM
- [x] Explanation: FullGrad
- [x] Explanation: Deep Feature Factorizations
- [x] Explanation: LaBo
- [x] Models: Yan et al.
- [x] Explanation: Yan et al.
- [x] Explanation: CBM-lime (CBM_supervised)
- [x] Explanation: CBM-shap (CBM_supervised)

## Roadmap v0.3

- [x] Dataset: COCO
- [x] Dataset: Cats/Dogs/Cars
- [x] Method : X-Nesyl
- [x] BBoxes : COCO
- [x] BBoxes : Pascalpart
- [x] BBoxes : Monumai
- [x] BBoxes : Cats/Dogs/Cars
- [x] Explanation: X-Nesyl linear

## Roadmap v0.4

- [x] Dataset: Toy example PASTA (saliency)
- [x] Criterion: variance criterion (saliency)
- [x] Criterion: complexity criterion (saliency)
- [x] Criterion: classification criterion (saliency)
- [x] Network: scoring network (saliency)
- [x] Dataset: Toy example PASTA (cbm)
- [x] Criterion: variance criterion (cbm)
- [x] Criterion: complexity criterion (cbm)
- [x] Criterion: classification criterion (cbm)
- [x] Network: scoring network (cbm)
- [x] PASTA: variant embeding text
- [x] PASTA: scoring from MLP
- [x] PASTA: training CBM + saliency
- [ ] PASTA: more options blur_image (tresholding,sigmoid ...)
- [ ] PASTA: more options cbm (only positive/negative, other templates ...)