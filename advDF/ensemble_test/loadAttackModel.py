
import torchvision
# from advDF.ensemble_test.model_for_attack.resnet import resnet_face18, resnet_face34, resnet_face50, resnet_face101, resnet_face152
from advDF.ensemble_test.model_for_attack.resnet import resnet_face18
import advDF.ensemble_test.retinanet.model as face_recognition_model
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
# from models.models import create_model
# from options.test_options import TestOptions
from advDF.ensemble_test.attacks import LinfPGDAttack  
from advDF.ensemble_test.model_for_attack import *
from torch.nn import DataParallel
import os
import pandas as pd
import advDF.ensemble_test.insight_arcface_backbones as insight_arcface_backbones
from advDF.ensemble_test.faceParsing import get_mask
import advDF.ensemble_test.get_landmark as get_landmark
from advDF.ensemble_test.path_utils import require_file
# from advDF.ensemble_test.Gender_classifier import Gender_classifier
from advDF.ensemble_test.Gender_classifier_ver2 import Gender_classifier
def get_face_recognition_model():
    classes={}
    for i in range(100):
        classes['people_'+str(i)]=i
    labels = {}
    for key, value in classes.items():
        labels[value] = key
    # dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
    #                                 transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    fr_model = face_recognition_model.resnet50(80,)
    fr_model.load_state_dict(torch.load(require_file('advDF/pytorch-retinanet/checkpoint/coco_resnet_50_map_0_335_state_dict.pt', 'RetinaNet checkpoint')))
    fr_model.training=False
    return fr_model
def load_attack_model(opt):
    face_recogntion_models=[]
    if opt.checkpoint_of_arcface_insight!=None:
        for name, weight in zip(opt.name_of_arcface_insight,opt.checkpoint_of_arcface_insight):
            if opt.lossType=='ensemble_face_recognition_inter_feature_loss' or opt.testAttackType=='style':
                insight_arcface=insight_arcface_backbones.get_model(name,fp16=False,return_intermediate_features=True)
            else:
                insight_arcface=insight_arcface_backbones.get_model(name, fp16=False,return_intermediate_features=False)
            insight_arcface.load_state_dict(torch.load(require_file(weight, f'InsightFace surrogate checkpoint for {name}')))
            insight_arcface.eval()
            face_recogntion_models.append(insight_arcface)

    # if opt.lossType=='output_id' or  opt.lossType=='internal_id' or  opt.lossType=='manipulate_id':
    #     arcface_for_attack=resnet_face18(False)
    #     arcface_for_attack=DataParallel(arcface_for_attack)
    #     arcface_for_attack.load_state_dict(torch.load('./model_for_attack/checkpoints/resnet18_110.pth'))
    # else:
    #     arcface_for_attack=None

    arcface_for_attack=resnet_face18(False)
    arcface_for_attack=DataParallel(arcface_for_attack)
    arcface_for_attack.load_state_dict(torch.load(require_file('advDF/ensemble_test/model_for_attack/checkpoints/resnet18_110.pth', 'legacy ArcFace ResNet-18 checkpoint')))
    
    if opt.lossType=='fd_loss':
        retinanet=get_face_recognition_model()
    else:
        retinanet=None
    if opt.lossType=='face_classification_loss' or opt.testAttackType=='ensemble_loss':
        # fairFace_net=FairFaceNet('advDF/FairFace/fair_face_models/fairface_alldata_4race_20191111.pt','advDF/FairFace/fair_face_models/fairface_alldata_4race_20191111.pt','cuda')
        # fairFace_net=AgeGenderEstimator()
        # fairFace_net.eval()
        fairFace_net=Gender_classifier(512*2,2)
        fairFace_net.eval()
    else:
        fairFace_net=None

    faceParsing_model=get_mask.FPModel('79999_iter.pth')
    if opt.lossType=='landmark_loss':
        landmark_model=get_landmark.LankMark()
    else:
        landmark_model=None
    
    # if opt.lossType=='ensemble_wb_test':
    #     if opt.testType=='id':
    #         arcface_for_attack=resnet_face18(False)
    #         arcface_for_attack=DataParallel(arcface_for_attack)
    #         arcface_for_attack.load_state_dict(torch.load(require_file('advDF/ensemble_test/model_for_attack/checkpoints/resnet18_110.pth', 'legacy ArcFace ResNet-18 checkpoint')))
            
    #         if opt.checkpoint_of_arcface_insight!=None:
    #             for name, weight in zip(opt.name_of_arcface_insight,opt.checkpoint_of_arcface_insight):
    #                 if opt.lossType=='ensemble_face_recognition_inter_feature_loss' :
    #                     print('yes')
    #                     exit(0)
    #                     insight_arcface=insight_arcface_backbones.get_model(name,fp16=False,return_intermediate_features=True)
    #                 else:
    #                     print('no')
    #                     exit()
    #                     insight_arcface=insight_arcface_backbones.get_model(name,fp16=False,return_intermediate_features=False)
    #                 insight_arcface.load_state_dict(torch.load(require_file(weight, f'InsightFace surrogate checkpoint for {name}')))
    #                 insight_arcface.eval()
    #                 face_recogntion_models.append(insight_arcface)
        
    return arcface_for_attack,retinanet,face_recogntion_models,face_recogntion_models,faceParsing_model,fairFace_net,landmark_model
