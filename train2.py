import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def back_to_3d(d2_similarity_map, d2_3d_cor, non_zero_index, ori_resolution = 336):
    # _, h, w, _ = d2_similarity_map.shape
    h = torch.sqrt(torch.tensor(d2_3d_cor.shape[2])).int()
    w = h
    b, nv, num_points, _ = d2_3d_cor.shape
    xx = d2_3d_cor[:, :, :, 0].reshape(-1).long()
    yy = d2_3d_cor[:, :, :, 1].reshape(-1).long()
    nbatch = torch.repeat_interleave(torch.arange(0, b*nv)[:,None], num_points).reshape(-1, ).cuda().long()
    d2_similarity_map = d2_similarity_map.permute(0, 3, 1, 2)
    point_logits = d2_similarity_map[nbatch, :, yy, xx]
    point_logits = point_logits.reshape(b, nv, num_points, 2)
    vweights = torch.ones((1, nv, 1, 1))
    vweights = vweights.reshape(1, -1, 1, 1).to(point_logits.device)
    is_seen = d2_3d_cor[:, :, :, 2].reshape(b, nv, num_points, 1)
    point_logits = point_logits * vweights * is_seen * non_zero_index
    mask = is_seen.bool() | (~non_zero_index.bool())
    point_logits = point_logits.sum(1)/ (mask.sum(1))
    point_logits = point_logits.reshape(b, h, w, 2)
    return point_logits


def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_transform, target_transform_pc = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("pretrained_weights/ViT-L-14-336px.pt", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()
    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, target_transform_pc = target_transform_pc, dataset_name = args.dataset, train_dataset_name = args.train_dataset_name, point_size = args.point_size)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
  ##########################################################################################
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.stage1_checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)
    
    params_to_optimize = []
    params_to_optimize.append({
        'params': model.fusion.parameters(),
        'lr': args.learning_rate  # Use the base learning rate for the new fusion module
    })
    lora_params = filter(lambda p: p.requires_grad, model.visual_depth.parameters())
    params_to_optimize.append({
        'params': lora_params,
        'lr': args.learning_rate * 0.1  # Use a smaller LR for LoRA fine-tuning
    })

    optimizer = torch.optim.Adam(params_to_optimize, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    model.eval()
    prompt_learner.eval()
    model.fusion.train()


    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        d2_pixel_loss_list = []
        d3_global_loss_list = []
        d3_point_loss_list = []
        d2_image_loss_list = []
        cons_loss_list = []
        for items in tqdm(train_dataloader):
            label =  items['anomaly']
            render_image = items['d2_render_img'].to(device)
            b, nv, c, h, w = render_image.shape
            render_image = render_image.reshape(-1, c, h, w)
            d2_render_anomaly = items['d2_render_anomaly'].to(device)
            d2_3d_cor = items['d2_3d_cor'].to(device)
            # mvtec 3d ad has some zero points
            non_zero_index = items['non_zero_index'].to(device)
            non_zero_index = non_zero_index.unsqueeze(1).repeat(1, nv, 1, 1)

            pointnet_input = items['pointnet_input'].to(device)
            depth_image = items['d2_depth_img'].to(device)
            depth_image = depth_image.reshape(-1, c, h, w)

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0


            render_gt_mask = items['d2_render_gt'].to(device)
            render_gt_mask[render_gt_mask > 0.5], render_gt_mask[render_gt_mask <= 0.5] = 1, 0

            render_gt_mask = render_gt_mask.reshape(-1, 1, h, w)
    
            with torch.no_grad():
                image_features, patch_features = model.encode_image(render_image, args.features_list, DPAM_layer = 20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) # b*nv,768
                # print('patch_features',patch_features[0].shape) # b*nv, 577, 768

                prompts, tokenized_prompts, compound_prompts_text = prompt_learner(pointnet_input)
                text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()  # 2*b,768
                # print('text_features',text_features.shape)
                text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)  # b,2,768
                # print('text_features',text_features.shape)
                text_features = text_features/text_features.norm(dim=-1, keepdim=True)  # b,2,768
                text_features = text_features.repeat_interleave(nv,dim=0) # b*nv,2,768
            
            image_features2, patch_features2 = model.encode_depth(depth_image, args.features_list)
            image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)
            image_features = model.fusion_r_d(image_features,image_features2)

           ####################################
            
            # Apply DPAM surgery
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)  # (b*nv,1,768)*(b*nv,768,2)=(b*nv,1,2) 
            # print('text_probs', text_probs.shape)
            text_probs = text_probs[:, 0, ...]/0.07
            d2_render_anomaly = d2_render_anomaly.reshape(-1,)
            d2_image_loss = F.cross_entropy(text_probs.squeeze(), d2_render_anomaly.long().cuda())
            #########################################################################
            text_probs = torch.chunk(text_probs,  nv, dim = 0)
            text_probs = torch.stack(text_probs, dim = 1).mean(1) # b,2
            # print('text_probs', text_probs.shape)
            d3_global_loss = F.cross_entropy(text_probs, label.long().cuda())
            #########################################################################
            # similarity_map_list.append(similarity_map)

            patch_feature = patch_features[0]
            patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True) # b*nv, 577, 768
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features) # b*nv, 577, 2
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
####################################################################
            d3_similarity_map = back_to_3d(similarity_map.permute(0, 2, 3, 1), d2_3d_cor, non_zero_index).permute(0, 3, 1, 2)
####################################################################

            d3_point_loss = 0
            d3_point_loss += loss_dice(d3_similarity_map[:, 1, :, :], gt)
            d3_point_loss += loss_dice(d3_similarity_map[:, 0, :, :], 1-gt)

            d2_pixel_loss = 0
            d2_pixel_loss += loss_focal(similarity_map, render_gt_mask)
            d2_pixel_loss += loss_dice(similarity_map[:, 1, :, :], render_gt_mask)
            d2_pixel_loss += loss_dice(similarity_map[:, 0, :, :], 1-render_gt_mask)

            #一致性损失
            cons_loss = 0
            multi_img_feature = image_features.reshape(b,nv,768)
            mean_img_feature = torch.mean(multi_img_feature, dim=1, keepdim=True)
            mean_img_feature = mean_img_feature.repeat(1,nv,1)
            cosine_sim = F.cosine_similarity(multi_img_feature, mean_img_feature)
            cons_loss += 1.0 - torch.mean(cosine_sim)
            cons_loss = cons_loss*0.3
        
            optimizer.zero_grad()
            (d3_point_loss + d2_image_loss + cons_loss).backward()
            optimizer.step()
            d2_pixel_loss_list.append(d2_pixel_loss.item())
            d3_point_loss_list.append(d3_point_loss.item())
            d3_global_loss_list.append(d3_global_loss.item())
            d2_image_loss_list.append(d2_image_loss.item())
            cons_loss_list.append(cons_loss.item())
        # logs
        if (epoch + 1) % args.print_freq == 0:
            # logger.info('epoch [{}/{}], d2_pixel_loss:{:.4f}, d3_point_loss:{:.4f}, d3_global_loss:{:.4f}, d2_image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(d2_pixel_loss_list), np.mean(d3_point_loss_list), np.mean(d3_global_loss_list), np.mean(d2_image_loss_list)))
            logger.info('epoch [{}/{}], d3_global_loss:{:.4f}, d2_image_loss:{:.4f}, cons_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(d3_global_loss_list), np.mean(d2_image_loss_list), np.mean(cons_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            save_obj = {
                "fusion": model.fusion.state_dict(),
                "visual_depth": model.visual_depth.state_dict()
            }
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save(save_obj, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("GD-CLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')


    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--train_dataset_name", type=str, default='cookie', help="save frequency")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--point_size", type=int, default=336, help="save frequency")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--stage1_checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
