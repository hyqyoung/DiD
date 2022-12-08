import math
#from urllib import urlretrieve
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F
import cv2
import pdb

def obj_loc(score, threshold):
    smax, sdis, sdim = 0, 0, score.size(0)
    minsize = int(math.ceil(sdim * 0.125))  #0.125
    snorm = (score - threshold).sign()
    snormdiff = (snorm[1:] - snorm[:-1]).abs()

    szero = (snormdiff==2).nonzero()
    if len(szero)==0:
       zmin, zmax = int(math.ceil(sdim*0.125)), int(math.ceil(sdim*0.875))
       return zmin, zmax

    if szero[0] > 0:
       lzmin, lzmax = 0, szero[0].item()
       lzdis = lzmax - lzmin
       lsmax, _ = score[lzmin:lzmax].max(0)
       if lsmax > smax:
          smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
       if lsmax == smax:
          if lzdis > sdis:
             smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if szero[-1] < sdim:
       lzmin, lzmax = szero[-1].item(), sdim
       lzdis = lzmax - lzmin
       lsmax, _ = score[lzmin:lzmax].max(0)
       if lsmax > smax:
          smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
       if lsmax == smax:
          if lzdis > sdis:
             smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if len(szero) >= 2:
       for i in range(len(szero)-1):
           lzmin, lzmax = szero[i].item(), szero[i+1].item()
           lzdis = lzmax - lzmin
           lsmax, _ = score[lzmin:lzmax].max(0)
           if lsmax > smax:
              smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
           if lsmax == smax:
              if lzdis > sdis:
                 smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if zmax - zmin <= minsize:
        pad = minsize-(zmax-zmin)
        if zmin > int(math.ceil(pad/2.0)) and sdim - zmax > pad:
            zmin = zmin - int(math.ceil(pad/2.0)) + 1
            zmax = zmax + int(math.ceil(pad/2.0))
        if zmin < int(math.ceil(pad/2.0)):
            zmin = 0
            zmax =  minsize
        if sdim - zmax < int(math.ceil(pad/2.0)):
            zmin = sdim - minsize + 1
            zmax = sdim

    return zmin, zmax

