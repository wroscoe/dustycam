# DustyCam

Open source AI camera to extract quantified information. 


This project is focused on making it very easy to create special purpose AI cameras. Everything from choosing the hardware, creating the AI model, communicating the results, to deploying the camera. All the camera software is open source and available on GitHub.

The project aims to provide reasonable defaults to accomplish the users goals while also allowing for customization and extension.


## The One Shot Workflow

Wildlife camera model.
```bash
dustycam make "A camera model that recognizes typical wyoming big game animals, people, vehicles, dogs, and cats. This model should be small enough to run at 5 frames per second on a Raspberry Pi 5."
``` 

License plate detector.
```bash
dustycam make "A camera model that reads license plates from passing cars on a city street. This model should be small enough to run at 5 frames per second on a Raspberry Pi 5."
``` 

## What DustyCam does behind the scenes. 

1. Defines what the user wants the camera to detect (ie wildlife, people, vehicles, license plates, etc).

2. Generates training data based on what the user wants to detect and scene.

3. Finetunes a model on the generated dataset.

4. Defines and tests a pipeline to run on the camera. This includes logic about when to take photos, how to process them, and how to store them.

5. Quantizes the model to run on the target camera (ie. to a Raspberry Pi or AI enabled microcontroller).

See a detailed description of the workflow in the [docs](docs/one_shot_workflow.md).



## Build Your Own DustyCam
Here's the hardware you need to get started.

- Raspberry Pi 5
- Raspberry Pi Camera (Module HQ recommended)
- microSD Card (16GB or larger)
- 3D Printed Case

* Additional hardware can enable battery and solar support. See Build Guide for details.

## Install On Your Pi

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[pi]"  # base + Pi camera deps
python3 src/dustycam/simple_capture.py # Example usage
```

Or run the helper script: `./install.sh`

**Note:** By default, this installs the CPU-only version of PyTorch to save space. 
To install with CUDA/GPU support:
`./install.sh --gpu`

## Add system service: Setup pi to take photos when there is motion.

See install-service.sh


## Test software on your computer.
For desktop development, install only base deps (no Pi extras). The Pi-only tests will auto-skip when the dependency is missing.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .  # no [pi] extra
pip install pytest

# Example dev runs
python3 -m dustycam.simple_capture --help
pytest
```


## Tips


* copy files from your pi to your computer: `rsync -avz dusty@dusty.local:~/dustycam_images ~/dustycam_images`
