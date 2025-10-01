import os
import json

def shapenet_classes():
    return ['ashtray0', 'bag0', 'bottle0', 'bottle1', 'bottle3', 'bowl0', 'bowl1', 'bowl2', 'bowl3', 'bowl4', 'bowl5', 'bucket0', 'bucket1', 'cap0', 'cap3', 'cap4', 'cap5', 'cup0', 'cup1', 'eraser0', 'headset0', 'headset1', 'helmet0', 'helmet1', 'helmet2', 'helmet3', 'jar0', 'microphone0', 'shelf0', 'tap0', 'tap1', 'vase0', 'vase1', 'vase2', 'vase3', 'vase4', 'vase5', 'vase7', 'vase8', 'vase9']

class MVTecSolver(object):


    def __init__(self, root='data/Real3D-AD-PCD', cls_id = 0):
        self.root = root
        self.CLSNAMES = [shapenet_classes()[cls_id]]
        # self.meta_path = f'{root}/meta.json'

    def run(self):
        # info = dict(train={}, test={})
        # anomaly_samples = 0
        # normal_samples = 0
        for cls_name in self.CLSNAMES:
            info = dict(train={}, test={})
            anomaly_samples = 0
            normal_samples = 0
            print("cls_name", cls_name, f'{self.root}/{cls_name}_meta.json')
            self.meta_path = f'{self.root}/{cls_name}_meta.json'
            # self.meta_path = f'{self.root}/meta.json'
            cls_dir = f'{self.root}/{cls_name}'
            # for phase in ['train', 'test']:
            for phase in ['test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    # img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    img_path = f'{cls_dir}/{phase}/{specie}'
                    print('img_path', img_path)
                    # d2_img_names = os.listdir(os.path.join(img_path, 'rgb'))
                    # d2_mask_names = os.listdir(os.path.join(img_path, 'gt')) if is_abnormal or phase == 'test' else None
                    d2_mask_names = os.listdir(os.path.join(img_path, 'gt_pcd'))
                    d3_pc = os.listdir(os.path.join(img_path, 'pcd'))
                    # d2_img_names.sort()
                    d2_mask_names.sort() if d2_mask_names is not None else None
                    d3_pc.sort()
                    for idx, d2_img_name in enumerate(d2_mask_names):
                        info_img = dict(
                            # d2_img_path=f'{img_path}/rgb/{d2_img_name}',
                            d2_mask_path=f'{img_path}/gt_pcd/{d2_mask_names[idx]}' if d2_mask_names else '',
                            d3_pc=f'{img_path}/pcd/{d3_pc[idx]}',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                            d2_render_img_path=f'{img_path}/2d_rendering/{d2_img_name.split(".")[0]}',
                            d2_render_gt_path=f'{img_path}/2d_gt/{d2_img_name.split(".")[0]}',
                            d2_corrdinate=f'{img_path}/2d_3d_cor/{d2_img_name.split(".")[0]}',
                            pcd_path=f'{img_path}/pcd/{d2_img_name.split(".")[0]}.pcd',
                            gt_pcd_path=f'{img_path}/gt_pcd/{d2_img_name.split(".")[0]}.pcd',
                            d2_depth_path=f'{img_path}/2d_depth/{d2_img_name.split(".")[0]}'
                        )
                        cls_info.append(info_img)
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples = anomaly_samples + 1
                            else:
                                normal_samples = normal_samples + 1
                info[phase][cls_name] = cls_info
            with open(self.meta_path, 'w') as f:
                f.write(json.dumps(info, indent=4) + "\n")
            print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("generate training dataset", add_help=True)
    # path
    parser.add_argument("--cls_id", type=int, default=3, help="select the class for training")
    args = parser.parse_args()
    for i in range(40):
        runner = MVTecSolver(root='/data1/dengzehao/new/data/Anomaly-ShapeNet', cls_id = i)
        runner.run()
