[clmParams_mouth, pdm_mouth]=Load_CLM_params_mouth();
[patches_mouth]=Load_Patch_Experts('../models/hierarch/','ccnf_patches_*_mouth_mv.mat',[],[],clmParams_mouth);
clmParams_mouth.multi_modal_types=patches_mouth(1).multi_modal_types;

%嘴部特征点在全局特征点中的坐标index
mouth_inds=[49:68];

%上嘴唇index
up_inds=[49:53 61 62 62]+1-48;

%下嘴唇index
lo_inds=[55:59 65 66 67]+1-48;


%%目的：扩大bbox搜索范围，避免陷入局部误判
%张嘴上下宽度
dy=abs(shape_mouth(4,2)-shape_mouth(10,2));

%上嘴唇特征点向上移动
shape_mouth(up_inds,2)=shape_mouth(up_inds,2)-0.1*dy;

%下嘴唇特征点向下移动
shape_mouth(lo_inds,2)=shape_mouth(lo_inds,2)+0.2*dy;
