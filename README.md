# DustyCam

An open source AI camera to quickly extract quantified information. 


## TODO
- Abstract the  camera so its easy to get picamer, desktop  camera or a mock camera.
- Get object detection working with YoloV8
- Setup pipeline to have gemini or openai label the images. See this example: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/bounding-box-detection


## Build Your Own DustyCam
Here's the hardware you need to get started.

- Raspberry Pi 5
- Raspberry Pi Camera (Module HQ recommended)
- microSD Card (16GB or larger)
- 3D Printed Case

* Additional hardware can enable battery and solar support. See Build Guide for details.

## Install On Your Pi

See the install.sh script.

## Add system service: Setup pi to take photos when there is motion.

See install-service.sh


## Test software on your computer.
For testing and development, you can run the software on your computer. This uses local images instead of a camera.

```bash
python dustycam.py --test
```


## Tips

* run scripts with `uv run` instead of `python`.
* copy files from your pi to your computer: `rsync -avz dusty@dusty.local:~/dustycam_images ~/dustycam_images`
