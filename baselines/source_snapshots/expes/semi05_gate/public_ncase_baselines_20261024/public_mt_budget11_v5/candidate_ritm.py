def get_next_click3D_torch_ritm(prev_seg, gt_semantic_seg):
    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > mask_threshold)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    fn_mask_single = F.pad(fn_masks, (1,1,1,1,1,1), 'constant', value=0).to(torch.uint8)[0,0]
    fp_mask_single = F.pad(fp_masks, (1,1,1,1,1,1), 'constant', value=0).to(torch.uint8)[0,0]
    fn_mask_dt = torch.tensor(edt.edt(fn_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]
    fp_mask_dt = torch.tensor(edt.edt(fp_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1, 1:-1]
    fn_max_dist = torch.max(fn_mask_dt)
    fp_max_dist = torch.max(fp_mask_dt)
    is_positive = fn_max_dist > fp_max_dist # the biggest area is selected to be interaction point
    dt = fn_mask_dt if is_positive else fp_mask_dt
    to_point_mask = dt > (max(fn_max_dist, fp_max_dist) / 2.0) # use a erosion area
    to_point_mask = to_point_mask[None, None]
    # import pdb; pdb.set_trace()


    for i in range(gt_semantic_seg.shape[0]):
        # points = torch.argwhere(to_point_mask[i])
        points = torch.nonzero(to_point_mask[i], as_tuple=False)
        # points = torch.nonzero(to_point_mask[i], as_tuple=False)  ################
        if len(points) == 0:
            # Same random negative fallback as the author's sibling click function.
            point = torch.Tensor([np.random.randint(sz) for sz in fn_masks[i].size()]).to(torch.int64)
            is_positive = False
        else:
            point = points[np.random.randint(len(points))]
            # print('point.shape',point.shape,'='*100)
            if fn_masks[i, 0, point[0], point[1], point[2]]:
                is_positive = True
            else:
                is_positive = False

        bp = point[1:].clone().detach().reshape(1,1,3) 
        bl = torch.tensor([int(is_positive),]).reshape(1,1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels # , (sum(dice_list)/len(dice_list)).item()    
