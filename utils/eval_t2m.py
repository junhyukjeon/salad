import torch
import torch.nn.functional as F

import os
from tqdm import tqdm

from utils.metrics import *
from utils.motion_process import recover_from_ric

#
#
# def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
#     xyz = xyz[:1]
#     bs, seq = xyz.shape[:2]
#     xyz = xyz.reshape(bs, seq, -1, 3)
#     plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
#     plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
#     writer.add_video(tag, plot_xyz, nb_iter, fps=20)


@torch.no_grad()
def evaluation_vae(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                   best_top2, best_top3, best_matching, eval_wrapper, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        m_length = m_length.cuda()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # joints_num = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_dict = net(motion)
        mask = torch.arange(motion.shape[1]).unsqueeze(0).expand(motion.shape[0], -1).cuda() >= m_length.unsqueeze(1)
        pred_pose_eval = pred_pose_eval.masked_fill(mask.unsqueeze(-1), 0)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('Test/FID', fid, ep)
        writer.add_scalar('Test/Diversity', diversity, ep)
        writer.add_scalar('Test/top1', R_precision[0], ep)
        writer.add_scalar('Test/top2', R_precision[1], ep)
        writer.add_scalar('Test/top3', R_precision[2], ep)
        writer.add_scalar('Test/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vae': net.state_dict(), 'epoch': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'vae': net.state_dict(), 'epoch': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def test_vae(val_loader, net, repeat_id, eval_wrapper, num_joint, cal_mm=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    multimodality = 0
    num_poses = 0

    num_mm_batch = 3 if cal_mm else 0

    for i, batch in enumerate(tqdm(val_loader, desc="Evaluation", leave=False)):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        # GT motion
        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # predicted motion
        if i < num_mm_batch:
            motion_multimodality_batch = []
            for _ in range(30):
                pred_pose_eval, loss_dict = net(motion)
                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            pred_pose_eval, loss_dict = net(motion)
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            num_poses += gt.shape[0]

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    # msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f" % \
    #       (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
    #        R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe)
    msg = f"--> \t Eva. Repeat {repeat_id} : FID. {fid:.4f}, "\
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "\
          f"R_precision_real. ({R_precision_real[0]:.4f}, {R_precision_real[1]:.4f}, {R_precision_real[2]:.4f}), "\
          f"R_precision. ({R_precision[0]:.4f}, {R_precision[1]:.4f}, {R_precision[2]:.4f}), "\
          f"matching_real. {matching_score_real:.4f}, matching_pred. {matching_score_pred:.4f}, "\
          f"MPJPE. {mpjpe:.4f}, Multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, mpjpe, multimodality


@torch.no_grad()
def evaluation_denoiser(out_dir, val_loader, denoiser, gen_func, writer, ep,
                        best_fid, best_div, best_top1, best_top2, best_top3, best_matching,
                        eval_wrapper, save=True, draw=True, device="cuda"):
    denoiser.eval()

    motion_annotation_list = []
    motion_pred_list = []

    motion_gt_list = []
    motion_gen_list = []

    cond_list = []
    m_length_list = []

    R_precision_real = 0
    R_precision = 0

    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for batch in tqdm(val_loader, desc="Evaluation", leave=False):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.to(device, dtype=torch.float32)
        m_length = m_length.to(device, dtype=torch.long)
        m_length_list.append(m_length)

        # real motion
        motion_gt_list.append(motion)
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match

        # generated motion
        pred_pose_eval, _ = gen_func((caption, motion, m_length))
        mask = torch.arange(motion.shape[1]).unsqueeze(0).expand(motion.shape[0], -1).cuda() >= m_length.unsqueeze(1)
        pred_pose_eval = pred_pose_eval.masked_fill(mask.unsqueeze(-1), 0)
        motion_gen_list.append(pred_pose_eval)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        cond_list.extend(caption)

        nb_sample += bs

    m_length_np = torch.cat(m_length_list, dim=0).cpu().numpy()

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    motion_gt_np = torch.cat(motion_gt_list, dim=0).cpu().numpy()
    motion_gen_np = torch.cat(motion_gen_list, dim=0).cpu().numpy()

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar("Test/FID", fid, ep)
        writer.add_scalar("Test/Diversity", diversity, ep)
        writer.add_scalar("Test/top1", R_precision[0], ep)
        writer.add_scalar("Test/top2", R_precision[1], ep)
        writer.add_scalar("Test/top3", R_precision[2], ep)
        writer.add_scalar("Test/matching_score", matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({"denoiser": denoiser.state_dict_without_clip(), "epoch": ep}, os.path.join(out_dir, "net_best_fid.tar"))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({"denoiser": denoiser.state_dict_without_clip(), "epoch": ep}, os.path.join(out_dir, "net_best_matching.tar"))

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer, motion_gt_np, motion_gen_np, m_length_np, cond_list


@torch.no_grad()
def test_denoiser(val_loader, gen_func, repeat_id, eval_wrapper, num_joint, cal_mm=True):
    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    l1_dist = 0

    nb_sample = 0
    num_poses = 0
    num_mm_batch = 3 if cal_mm else 0

    pred_motion_for_rendering = []
    caption_list = []

    for i, batch in enumerate(tqdm(val_loader, desc="Evaluation", leave=False)):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
        motion = motion.cuda()
        m_length = m_length.cuda()

        # GT motion
        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # predicted motion
        if i < num_mm_batch:
            motion_multimodality_batch = []
            for _ in range(30):
                pred_pose_eval, loss_dict = gen_func((caption, motion, m_length))
                mask = torch.arange(motion.shape[1]).unsqueeze(0).expand(motion.shape[0], -1).cuda() >= m_length.unsqueeze(1)
                pred_pose_eval = pred_pose_eval.masked_fill(mask.unsqueeze(-1), 0)
                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            pred_pose_eval, loss_dict = gen_func((caption, motion, m_length))
            mask = torch.arange(motion.shape[1]).unsqueeze(0).expand(motion.shape[0], -1).cuda() >= m_length.unsqueeze(1)
            pred_pose_eval = pred_pose_eval.masked_fill(mask.unsqueeze(-1), 0)
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose
            pred_motion_for_rendering.append(pred.cpu().numpy())
            caption_list.append(caption[i])

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match

        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} : FID. {fid:.4f}, "\
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "\
          f"R_precision_real. ({R_precision_real[0]:.4f}, {R_precision_real[1]:.4f}, {R_precision_real[2]:.4f}), "\
          f"R_precision. ({R_precision[0]:.4f}, {R_precision[1]:.4f}, {R_precision[2]:.4f}), "\
          f"matching_real. {matching_score_real:.4f}, matching_pred. {matching_score_pred:.4f}, "\
          f"mae. {l1_dist:.4f}, Multimodality. {multimodality:.4f}"
    print(msg)
    return msg, fid, diversity, R_precision, matching_score_pred, l1_dist, multimodality, pred_motion_for_rendering, caption_list