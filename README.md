# LMO-DDPG_WZJ
An optimized DDPG algorithm for autonomous path planning of UAV in network-unavailable environment

# Note
1. Install Unreal Engine, Visual Studio Community and Python with versions of 4.26, 2019 and 3.5 respectively.
2. Download Airsim and compile it. The Plugins folder will be generated in Airsim\Unreal folder.
3. Create a new project in Unreal Engine, add the Plugins folder, compile it, and then open it in Visual Studio Community.
4. Download the scenario folder and unzip SoulCave and Forest.
5. Add scenario\SoulCave to the content folder in the project.
6. Download Landscape Mountains in Epic Game Launcher, compile and add Plugins, and replace the files in scenario\Forest into the corresponding folder of the project.
7. Run the Cave or Forest project, select the map in Unreal Engine, and select AisimGameMode in world scene settings.
8. Running arisim_ env.py file. If there is no problem, you can run the entire algorithm program.
