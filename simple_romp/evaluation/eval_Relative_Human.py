
import argparse
import os, sys
import os.path as osp
import numpy as np
import cv2
import torch
# 从安装的simple_romp库里面导入
# from romp import ResultSaver
# from romp.utils import progress_bar

# 从simple_romp/romp下面导入，需要先引入romp的目录到sys.path, 
# 加下面这个路径就会在simple_romp当前修改的文件运行，不加的话就是运行之前安装的版本，现在的修改无影响,即先引入当前目录的，没有的话再引入库的
sys.path.insert(0, osp.join(osp.dirname(os.path.abspath(__file__)), '..'))  # '/home/ssw/code/romp/simple_romp/evaluation' + '...'
from romp.utils import ResultSaver
from romp.utils import progress_bar
from RH_evaluation import RH_Evaluation

root_dir = '/data2/2020/ssw/'
user_dir = '/home/lyc/'
if 'code' in osp.dirname(os.path.abspath(__file__)):
    root_dir = '/home/ssw/code/'
    user_dir = '/home/ssw/'


set_id = 1
set_name = ['val', 'test'][set_id]

method_id = 1
method_name = ['ROMP', 'BEV'][method_id]

# model_path = {
#     'BEV': user_dir + '/.romp/BEV.pth',
# }
#   CUDA_VISIBLE_DEVICES='1' python simple_romp/evaluation/eval_Relative_Human.py


ft_model_name = 'validation_epoch_2_iter_3999_MPJPE_111.85_PA_MPJPE_73.51_tab_hrnet_cm128_V6_hrnet_relative_ft.pkl'
ft_path = root_dir  + '/romp/output_vis/checkpoints/hrnet_cm128_V6_hrnet_relative_ft_on_gpu0_val/' + ft_model_name

train_model_name = 'validation_epoch_44_iter_9999_MPJPE_120.42_PA_MPJPE_78.86_tab_hrnet_cm128_V6_hrnet_relative_train.pkl'
train_path = root_dir  + '/romp/output_vis/checkpoints/hrnet_cm128_V6_hrnet_relative_train_on_gpu0_val/' + train_model_name

path = ft_path

path = train_path


model_path = {
    'BEV': path,
}
visualize_results = False

Relative_Human_dir = root_dir +  '/dataset/Relative_human'
output_save_dir = root_dir + '/romp/simple_romp/output/Relative_results/{}_{}'.format(set_name, method_name)
#if osp.isdir(output_save_dir):
#    import shutil
#    shutil.rmtree(output_save_dir)
os.makedirs(output_save_dir,exist_ok=True)

relative_age_types = ['adult', 'teen', 'kid', 'baby']
relative_depth_types = ['eq', 'cd', 'fd']

def joint_mapping(source_format, target_format):
    mapping = np.ones(len(target_format),dtype=np.int)*-1
    for joint_name in target_format:
        if joint_name in source_format:
            mapping[target_format[joint_name]] = source_format[joint_name]
    return np.array(mapping)

Crowdpose_14 = {"L_Shoulder":0, "R_Shoulder":1, "L_Elbow":2, "R_Elbow":3, "L_Wrist":4, "R_Wrist":5,\
     "L_Hip":6, "R_Hip":7, "L_Knee":8, "R_Knee":9, "L_Ankle":10, "R_Ankle":11, "Head_top":12, "Neck_LSP":13}

SMPL_24 = {
    'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'Jaw':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23
    }
SMPL_EXTRA_30 = {
    'Nose':24, 'R_Eye':25, 'L_Eye':26, 'R_Ear': 27, 'L_Ear':28, \
    'L_BigToe':29, 'L_SmallToe': 30, 'L_Heel':31, 'R_BigToe':32,'R_SmallToe':33, 'R_Heel':34, \
    'L_Hand_thumb':35, 'L_Hand_index': 36, 'L_Hand_middle':37, 'L_Hand_ring':38, 'L_Hand_pinky':39, \
    'R_Hand_thumb':40, 'R_Hand_index':41,'R_Hand_middle':42, 'R_Hand_ring':43, 'R_Hand_pinky': 44, \
    'R_Hip': 45, 'L_Hip':46, 'Neck_LSP':47, 'Head_top':48, 'Pelvis':49, 'Thorax_MPII':50, \
    'Spine_H36M':51, 'Jaw_H36M':52, 'Head':53
    }
SMPL_ALL_54 = {**SMPL_24, **SMPL_EXTRA_30}
kp_mapper = joint_mapping(SMPL_ALL_54, Crowdpose_14)

def collect_relative_results(outputs):
    image_results = []
    person_num = len(outputs['smpl_thetas'])
    for ind in range(person_num):
        results = {}
        results['global_orient'] = outputs['smpl_thetas'][ind, :3]
        results['body_pose'] = outputs['smpl_thetas'][ind, 3:].reshape(23,3)
        results['smpl_betas'] = outputs['smpl_betas'][ind]
        results['trans'] = outputs['cam_trans'][ind]
        results['kp2ds'] = outputs['pj2d_org'][ind, kp_mapper]
        image_results.append(results)
    return image_results

@torch.no_grad()
def get_BEV_results_on_RH(set_name='test'):
    from bev import BEV
    default_eval_settings = argparse.Namespace(GPU=0, calc_smpl=True, center_thresh=0.1, nms_thresh=16,\
                render_mesh=visualize_results, renderer='pyrender', show=False, show_largest=False, \
                input=None, frame_rate=24, temporal_optimize=False, smooth_coeff=3.0, relative_scale_thresh=2, overlap_ratio=0.8,\
                mode='image', model_path=model_path[method_name], onnx=False, crowd=False,\
                save_path=osp.join(output_save_dir,'visualization'), save_video=False, show_items='mesh', show_patch_results=False, \
                smpl_path= user_dir + '/.romp/smpla_packed_info.pth', smil_path= user_dir + '/.romp/smil_packed_info.pth')

    image_folder = osp.join(Relative_Human_dir, 'images')
    annots_path = osp.join(Relative_Human_dir, '{}_annots.npz'.format(set_name))
    file_list = list(np.load(annots_path,allow_pickle=True)['annots'][()].keys())
    file_list = [os.path.join(image_folder, img_name) for img_name in file_list]
    
    model = BEV(default_eval_settings)

    all_results_dict = {}
    if visualize_results:
        saver = ResultSaver(default_eval_settings.mode, default_eval_settings.save_path, save_npz=False)
    for image_path in progress_bar(file_list):  #   file_list[80] 是复杂图片 100454.jpg
        # image_path = file_list[80]  # 直接测试80是否正确
        image = cv2.imread(image_path)
        outputs = model(image)
        if outputs is None:
            print('image_path: ', image_path)
            continue
        if visualize_results:
            saver(outputs, image_path)
        results = collect_relative_results(outputs)
        all_results_dict[osp.basename(image_path)] = results
    
    result_save_path = osp.join(output_save_dir, '{}_results.npz').format(set_name)
    np.savez(result_save_path, results=all_results_dict)
    return result_save_path

def get_ROMP_results_on_RH(set_name='test'):
    from romp import ROMP
    default_eval_settings = argparse.Namespace(GPU=0, calc_smpl=True, center_thresh=0.25, frame_rate=24, \
                input=None, mode='image', model_onnx_path= user_dir + '/.romp/ROMP.onnx', model_path= user_dir + '/.romp/ROMP.pkl', onnx=False, \
                render_mesh=False, save_path=osp.join( output_save_dir,'visualization'), save_video=False, show=False, show_largest=False, \
                smooth_coeff=3.0, smpl_path= user_dir + '/.romp/smpl_packed_info.pth', temporal_optimize=False)
    
    image_folder = osp.join(Relative_Human_dir, 'images')
    annots_path = osp.join(Relative_Human_dir, '{}_annots.npz'.format(set_name))
    file_list = list(np.load(annots_path,allow_pickle=True)['annots'][()].keys())
    file_list = [os.path.join(image_folder, img_name) for img_name in file_list]
    
    model = ROMP(default_eval_settings)

    all_results_dict = {}
    if visualize_results:
        saver = ResultSaver(default_eval_settings.mode, default_eval_settings.save_path, save_npz=False)
    for image_path in progress_bar(file_list):
        image = cv2.imread(image_path)
        outputs = model(image)
        if outputs is None:
            continue
        if visualize_results:
            saver(outputs, image_path)
        results = collect_relative_results(outputs)
        all_results_dict[osp.basename(image_path)] = results
    
    result_save_path = osp.join(output_save_dir, '{}_results.npz').format(set_name)
    np.savez(result_save_path, results=all_results_dict)
    return result_save_path

if __name__ == '__main__':
    if method_name == 'ROMP':
        result_save_path = get_ROMP_results_on_RH(set_name)
        #result_save_path = osp.join(output_save_dir, 'test_results.npz')
        RH_Evaluation(result_save_path, Relative_Human_dir, set_name, path.split('/')[-1])
    if 'BEV' in method_name:
        result_save_path = get_BEV_results_on_RH(set_name)
        #result_save_path = osp.join(output_save_dir, 'test_results.npz')
        RH_Evaluation(result_save_path, Relative_Human_dir, set_name, path.split('/')[-1])