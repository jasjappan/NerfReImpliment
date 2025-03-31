import os
import sys
import time
import json
import glob
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython import display
from tqdm.notebook import tqdm
import json
from utils.helperFunctions import HelperFunctions
import torch

torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

path = '/media/adminnio/Volume/Data_NerfRaw/Lakshwadeep/LAK/1/colmap'
poses, c2w = HelperFunctions.get_camera_poses(path)
#HelperFunctions.plot_camera_poses(poses)
with open((rf"{path}/transforms.json"), "r") as file:
    data = json.load(file)
    
    
W, H ,focalLength = data.get("w"), data.get("h"), data.get("fl_x")
W = 50
H = 50
origins = poses[:,:-3]
camera_vector = poses[1,3:]

originrays, dirrays  = HelperFunctions.generateRays(1, 10,poses, origins[1], H, W, focalLength,c2w[1])
pts = HelperFunctions.samplePoints(originrays, dirrays, numPoints=20, tn=1, tf=20)

#HelperFunctions.plot_rays(originrays, dirrays,10,10)
#HelperFunctions.plot_ray(pts, 2)
HelperFunctions.plot_rays_and_points(originrays, dirrays, pts)