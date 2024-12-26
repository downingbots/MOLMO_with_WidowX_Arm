# PYTHON-ONLY WIDOWX ARM CONTROLLER 

The Trossen WidowX series for robot arms have long been a popular 
arm for robot hobbyists and researcher.  In recent years, the widowx
controller has been used by Stanford and UC Berkeley for such projects
as REPLAB, DeepMind RT1 and RT2, and Mobile Aloha.

Language-conditioned control of robots has evolved over the years and
this repository is an evolution of two earlier experiments:
 - DeepMind_RT_X_with_WidowX_Arm: the DeepMind RT_X model control of
   a WidowX arm.  The arm controller is written in C++. The results
   were underwhelming.
 - GPT4o_with_WidowX-Arm: An early public image/text-to-text multimodal
   version of ChatGPT4o was used to control incremental movements (up/down,
   left/right,gripper open/close, forward/back) of a 
   WidowX arm.  The arm controller was ported to Python.  The experiment
   ran into limitations of the ChatGPT spacial localization.
 - MOLMO_with_WidowX_Arm: MOLMO is an open-weights multimodal language 
   model with spacial localization support with robotics in mind. The
   ability to point to locations on the image based on text prompts.
   The python arm controller was significantly enhanced, so that a
   can handle higher-level commands like pick/place/push with parameters
   specified as locations on the image. So, output from MOLMO can be
   fed directly into the arm controller. A more ambitious demo was 
   chosen, but MOLMO was not up to the task. This repository is described
   in more detail below.

Trossen provides a ROS1 controller for the widowX arm.  This repository
is a python-only controller for Trossen Robot Arm Mark II. No ROS
is required.  The controller is designed with Stanford's and UC Berkeley's RT1/RT2 
arm hardware configurations in mind (see https://rail-berkeley.github.io/bridgedata/ ).  
The controller provides an accurate point-and-click interface for robotic arm skills such as 
pick-up/place/push of objects on the tabletop in front of the arm.
With hobbyist servo-based robot arms, accuracy is more difficult due to 
gravity.  The controller tries to have the X/Y dimensions within .1 cm 
for reaches up to 24cm and .2 cm above 24cm.  The vertical gripper gamma
and the Z dimensions try to meet the specifications on a best-effort basis
to work around the joint limits.

Automated arm calibration to a fixed camera is provided. 
The calibration and arm movement attempts reduces the effect of gravity 
by emphasizing swiveling of the arm - so that a servo that is fighting
gravity with most of its available torque does not have an instant when
the torque is suddenly released upon a change of servo position.  OpenCV 
is used extensively to compute the mapping from images of a fixed 
camera to the tabletop, and include using the edge of the 
tabletop that can be used as a vanishing point for the birds-eye-view 
transformations as seen below:
<p align="center">
  <img src="https://raw.githubusercontent.com/downingbots/ALSET/Molmo_with_WidowX_Arm/image_tt_vanishing_pt_BEV.png" width="600" alt="accessibility text">
</p> 
To calibrate real-world gripper locations to image pixels, the gripper is
rotated to different positions and then the gripper is open/closed in view
of the camera.  OpenCV is then used to convert the open/closed images into
black and white and then the images are differenced to find the bounding box
of the grasping area of the gripper.  In the following composite image, the 
bottom right image shows the closed gripper; the bottom left is the open gripper;
the top left is the difference; the top right shows the grasping bounding box:
<p align="center">
  <img src="https://raw.githubusercontent.com/downingbots/ALSET/Molmo_with_WidowX_Arm/image_grasping_bb.png" width="600" alt="accessibility text">
</p> 
Combining the birds-eye-view transformations and the grasping bounding box to
pixel images, a least-squares linear equation can compute any real-world grasping
location for any pixel on the camera.  The following image illustrates the linear relationship:
<p align="center">
  <img src="https://raw.githubusercontent.com/downingbots/ALSET/Molmo_with_WidowX_Arm/image_pixel_world_calibration.png" width="600" alt="accessibility text">
</p> 
With an accurate image pixel mapping to real world mapping, higher-level functions
to do higher-level functions like pick, place, and push can be automatically executed.	
For example, a supplied demo (widowx_manual_control.py) lets click a before-and-after gripper 
position, and a cube will get pushed across a table as below:
<p align="center">
  <img src="https://raw.githubusercontent.com/downingbots/ALSET/Molmo_with_WidowX_Arm/image_push.png" width="600" alt="accessibility text">
</p> 

# USING THE MOLMO LLM FOR IMAGE+TEXT COMMAND OF THE WIDOWX ARM 

Now that we have higher-level WidowX functions, we want to see if an image+text to text large-language-model
(LLM) can be used to do more complex operatons. Molmo ( https://molmo.allenai.org/blog ) is a 
particularly interesting image+text to text model, because it is open-weights and it has been trained 
on a lot of robotics data and is able to interactively point to items in an image. So, if you want 
to find a cup, ask Molmo to "point at the cups in the image."   Molmo can also provide planning info.
This repository used the model Molmo-7B-0-0924. Information about MOLMO can found here:
https://huggingface.co/allenai/Molmo-7B-O-0924

To test with a complex demo for an LLM to control a robot, the Hasbro game of "Perfection" was modified to
become a robot arm's tabletop challenge.  The goal is to pick yellow plastic game pieces of diffent
shapes and place them in corresponding holes on a board as seen below.  A human was able to run the
pick_and_place demo to successfully play this game:
<p align="center">
  <img src="https://raw.githubusercontent.com/downingbots/ALSET/Molmo_with_WidowX_Arm/image_perfection_tt.png" width="600" alt="accessibility text">
</p> 

MOLMO was tested to try and replace the human playing perfection.  
MOLMO was easily able to identify the locations of the yellow plastic pieces ("point to the
yellow plastic pieces") and could count and identify the holes that were the destinations on
the blue board.  Unfortunately, Molmo fell short in describing/understanding the shapes of individual 
pieces/holes and was unable to match pieces to the appropriate hole.  One approach that seemed promising
when running a hosted Hugging Face MOLMO model was to choose which of two yellow pieces fits
better in a blue hole, but the approach failed as well when using the downloaded model.  
So, MOLMO was not quite up to the task to play the game of perfection.

The next experiment may try the newest OpenAI models with the higher-level WidowX Arm controller.

-
