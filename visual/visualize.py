import os
from os.path import join as opj
import sys
import random
#sys.path.append('/home/xuqianxu/muscles/MusclesInTime/musint')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from IPython.display import Video, display
np.bool = np.bool_
np.int = np.int_
np.float = np.float64
np.complex = np.complex128
np.object = np.object_
np.unicode = np.str_
np.str = np.str_
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
matplotlib.use("Agg")
import seaborn as sns
from matplotlib.animation import FFMpegWriter
from IPython.display import HTML, display, Video

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from visual.plotting_utils import visualize_pose
#os.environ['PYOPENGL_PLATFORM'] = 'egl'
import importlib.util

spec = importlib.util.spec_from_file_location(
    "muscle_sets", 
    "/home/xuqianxu/MMTransformer/visual/muscle_sets.py"
)
muscle_sets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(muscle_sets)

MUSCLE_SUBSETS = muscle_sets.MUSCLE_SUBSETS


import visual.amass_utils as amau

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.sans-serif": ["Helvetica"],
        "figure.figsize": (6, 4.8),  # Figure size
        "font.size": 12,  # Global font size
        "axes.titlesize": 12,  # Title font size
        "axes.labelsize": 12,  # Axes labels font size
        "xtick.labelsize": 12,  # X-tick labels font size
        "ytick.labelsize": 12,  # Y-tick labels font size
        "legend.fontsize": 12,  # Legend font size
        "figure.titlesize": 12,  # Figure title font size
    }
)


from visual.body_model import BodyModel

def build_body():
    m_bm_path = "/home/xuqianxu/muscles/models/smpl/models/SMPLX_MALE.npz"
    m_dmpl_path = None

    f_bm_path = "/home/xuqianxu/muscles/models/smpl/models/SMPLX_FEMALE.npz"
    f_dmpl_path = None
    num_betas = 10  
    num_dmpls = None  

    male_bm = BodyModel(bm_fname=m_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=m_dmpl_path).cuda()
    #faces = male_bm.f if isinstance(male_bm.f, np.ndarray) else male_bm.f.detach().cpu().numpy()
    female_bm = BodyModel(bm_fname=f_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=f_dmpl_path).cuda()
    
    return male_bm, female_bm
        

def visualize_pose_and_muscle(sample_id, male_bm, female_bm, pred0, muscle0):
    """
    Args:
        sample_id (str): relative path inside the dataset folder.
        pred: Tensor [1, T, 402]
        muscle: Tensor [1, T, 402]
    """

    target_muscles = [
            "LU_tibant_r", "LU_tibant_l",
            "LU_recfem_r", "LU_recfem_l",
            "TL_E0_R5_r", "TL_E0_R5_l"
        ]
    full_list = MUSCLE_SUBSETS["MUSINT_402"]
    muscle_idxs = [i for i, m in enumerate(full_list) if m in target_muscles]
    #full_muscle_names = [m.replace("_", " ").title() for m in target_muscles]
    full_muscle_names = target_muscles
    T = 64

    motion_root = "/home/xuqianxu/muscles/datasets/"
    print("Sample id:", sample_id)
    ind=0
    for sid in sample_id:
        pred = pred0[ind].cpu().numpy()        # [T, 402]
        muscle = muscle0[ind].cpu().numpy()    # [T, 402]
        ind+=1
        pred = pred[:, muscle_idxs]         # [T, 8]
        muscle = muscle[:, muscle_idxs]     # [T, 8]
        print("-", sid)
        pose_path = os.path.join(motion_root, sid)
        save_id = pose_path.split("/")[-1].replace(".npz", "")
        # SMPL conversion
        num_betas = 10
        body = amau.amass_to_smpl(pose_path, male_bm, female_bm, num_betas)
        
        vertices = body.v.detach().cpu().numpy()
        vertices = np.dot(vertices, np.array([[-1.0, 0.0, 0.0],
                                              [0.0, 0.0, 1.0],
                                              [0.0, 1.0, 0.0]]))  # Y-up

        if len(vertices) < T:
            print("Skipping", save_id, "due to short motion length")
            continue
        vertices = vertices[:T]

        fig = plt.figure(figsize=(8, 16))
        fig.tight_layout()
        gs = GridSpec(4, 1, height_ratios=[5,3,3,3])
        ax_3d = fig.add_subplot(gs[0], projection="3d")
        ax_bflh = fig.add_subplot(gs[1])
        ax_quad = fig.add_subplot(gs[2])
        ax_rect = fig.add_subplot(gs[3])
        time = np.arange(T)

        muscle_groups = {
            "LU_tibant": ["LU_tibant_r", "LU_tibant_l"],
            "LU_recfem": ["LU_recfem_r", "LU_recfem_l"],
            "TL_E0_R5": ["TL_E0_R5_r", "TL_E0_R5_l"]
        }
        axes = {
            "LU_tibant": ax_bflh,
            "LU_recfem": ax_quad,
            "TL_E0_R5": ax_rect
        }

        colors = {
            "LU_tibant_r": "blue", "LU_tibant_l": "cyan",
            "LU_recfem_r": "green", "LU_recfem_l": "lime",
            "TL_E0_R5_r": "red", "TL_E0_R5_l": "orange"
        }

        lines = [] 

        for group_name, muscles in muscle_groups.items():
            ax = axes[group_name]

            for side in muscles:
                idx = full_muscle_names.index(side)
                color = colors[side]

                line_gt, = ax.plot(time, muscle[:, idx], label=f"{side} GT", color=color, alpha=0.8)
                line_pred, = ax.plot(time, pred[:, idx], label=f"{side} Pred", color=color, linestyle="--", alpha=0.8)

                lines.extend([line_gt, line_pred])

            ax.set_xlim(0, T)
            min_val = min(np.min(pred[:, idx-1]), np.min(muscle[:, idx-1]))
            max_val = max(np.max(pred[:, idx-1]), np.max(muscle[:, idx-1]))
            ax.set_ylim(min_val - 0.05, max_val + 0.05)
            ax.set_title(f"{group_name} Activation Over Time")
            ax.set_xlabel("")
            ax.set_ylabel("Activation")
            ax.legend(fontsize="x-small", loc="upper right")
            ax.grid(True)

        bars = [
                ax_bflh.axvline(x=0, color="black", linestyle="--", lw=1),
                ax_quad.axvline(x=0, color="black", linestyle="--", lw=1),
                ax_rect.axvline(x=0, color="black", linestyle="--", lw=1)
            ]

        def update(frame):
            ax_3d.clear()
            visualize_pose(vertices[frame], frame, ax_3d, title=save_id.replace(".npz", ""))
            for bar in bars:
                bar.set_xdata([frame])

            return [ax_3d] + bars + lines


        ani = FuncAnimation(fig, update, frames=T, interval=50)
        #save_id = pose_path.split("/")[-1].replace(".npz", "")
        save_path = f"/home/xuqianxu/MMTransformer/outputs/{save_id}_muscle_vis.mp4"
        writer = FFMpegWriter(fps=20, bitrate=1800)
        ani.save(save_path, writer=writer)
        display(Video(save_path, embed=True))
        plt.close(fig)