# One Shot Workflow

This is the workflow that DustyCam follows to create a camera model.

## Users command 

The user states what they want the camera to detect, where it will be deployed, and the hardware they want to run it on.

```bash
dustycam make "A camera model that recognizes typical wyoming big game animals, people, vehicles, dogs, and cats. This model should be small enough to run at 5 frames per second on a Raspberry Pi 5. Don't spend more than $3 on data and training" 
```

## Clarify what the user wants

The user's command is sent to Gemini  with a system prompt to expand on what they want. This helps ensure the user's intent is clear and the camera model will work as expected.



## Generate training data

Given the budget calculate how much training data we can generate and estimate if this is enough to train an accurate model. If not, ask the user to increase the budget. 


The clarified target detection classes and scene descriptions are sent to Gemini with a system prompt to generate a modular prompt for generating training data. 

An example result for generating training data for wildlife might look like this:

prompt = """
Create an {image_descriptors}image of a {subject_descriptor} {subject} in a {scene} {scene_descriptor}.

subject = ["deer", "elk", "antelope", "rabbit", "fox", "bear", "cougar"]
subject_descriptors = ["adult", "female", "male", "juvenal", "immature", "subadult"]

scene = ["forest", "mountain", "river", "field", "grassland", "desert"  ]
scene_descriptors = ["sunlit", "shaded", "morning", "afternoon", "evening", "night"]
image_descriptors = ["clear", "cloudy", "rainy", "snowy", "foggy", "hazy"]



## Generate training data

Use the modular prompt to generate training data using its different possible combinations.

Save the generated images to a directory.


# Label the training data
Use Gemeni structured output to get bounding box coordinates for the objects in the images. Save this data in the yolo training format.

# Train the model
Use the training data to train a model. 

# Quantize the model
Quantize the model to run on the target hardware.

# Test the model
Test the model on a few images to ensure it works as expected.

# Deploy the model
Deploy the model to the target hardware.

# Monitor the model
Monitor the model to ensure it is working as expected.

# Update the model
Update the model as needed to improve accuracy.

