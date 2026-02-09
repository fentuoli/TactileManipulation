1. migrate the current ManipTrans-old repo from python3.8 to python3.11, isaacgym preview 4 to isaaclab+isaacsim5.1.0
2. create a new folder ManipTrans-new under ~/Code to serve as new repo
3. the ManipTrans-new should be run in conda env, which can be activated through "conda activate env_isaacsim"
4. The isaaclab is installed at ~/Code
5. As for the URDF converter, please check https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.converters.html#isaaclab.sim.converters.UrdfConverter, using existing API to convert URDF to USD files
6. The needed imitator checkpoints are under assets/, grab_demo_dataset is under data/
7. Replace pytorch3d with other python package or functions, because there is no suitable version of pytorch3d for current python version
8. Try to install the needed packages mentioned in README.md and requirements.txt, replace them with suitable packages. Before installing, please check if a suitable package already exists; if so, you don't need to install it.
9. Please check https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html to find official solutions on isaacgym->isaaclab migration
10. Please check https://isaac-sim.github.io/IsaacLab/main/source/api/index.html to find official definitions of new APIs
11. If you have something unsure, please find help online, don't guess
12. after finishing migration, verify your code using preprocessing, training and testing commands mentioned in README.md
13. Record key steps into Keynotes.md at fixed interval, and remember to read it
14. Try to solve problems in the simplest way
15. Only operate in `env_isaacsim` conda env, don't corrupt my local env settings
16. You only need to migrate single hand mode, i.e. you don't need migrate dexmanip_bih.py