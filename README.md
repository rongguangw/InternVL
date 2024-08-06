# InternVL Training

This repository contains a script for training the [InternVL model](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e).

## Installation

Install the required packages using `requirements.txt`.

- Create a conda virtual environment and activate it:

  ```bash
  conda create -n internvl python=3.9 -y
  conda activate internvl
  ```

- Install dependencies using `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

  By default, our `requirements.txt` file includes the following dependencies:

  - `-r requirements/internvl_chat.txt`
  - `-r requirements/streamlit_demo.txt`
  - `-r requirements/classification.txt`
  - `-r requirements/segmentation.txt`

  The `clip_benchmark.txt` is **not** included in the default installation. If you require the `clip_benchmark` functionality, please install it manually by running the following command:

  ```bash
  pip install -r requirements/clip_benchmark.txt
  ```

- Install `flash-attn`:

  ```bash
  pip install flash-attn --no-build-isolation
  ```

**Note:** You should install the `flash-attn` after running other libraries with `requirements.txt`.

## Model Download

Before training, download the InternVL model from HuggingFace. It is recommended to use the `huggingface-cli` to do this.

1. Install the HuggingFace CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

2. Download the model:

```bash
cd pretrained
# Download InternVL2-26B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-26B --local-dir InternVL2-26B
# Download InternVL2-40B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-40B --local-dir InternVL2-40B
```

## Inference

```bash
python inference.py \
 --model-name /name/of/model \
 --model-path /path/to/weight \
 --images-file /Path/to/image_folder
```
