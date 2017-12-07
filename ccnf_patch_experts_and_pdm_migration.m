load ccnf_patches_1.00_mouth_mv.mat;
Write_patch_experts_ccnf('ccnf_patches_1.00_mouth_mv.txt','p1.mat',trainingScale,centers,visiIndex,patch_experts,normalizationOptions,[7 9 11 15]);
clear all;
load ccnf_patches_2.00_mouth_mv.mat;
Write_patch_experts_ccnf('ccnf_patches_2.00_mouth_mv.txt','p1.mat',trainingScale,centers,visiIndex,patch_experts,normalizationOptions,[7 9 11 15]);

clear all;
load pdm_20_mouth.mat;
FID=fopen('pdm_20_mouth.txt','wt+');
writeMatrix(FID,M,6);
writeMatrix(FID,V,6);
writeMatrix(FID,E',6);
fclose(FID);

function writeMatrix(fileID,M,type)
  fprintf(fileID,'%d\n',size(M,1));
  fprintf(fileID,'%d\n',size(M,2));
  fprintf(fileID,'%d\n',type);

  for i=1:size(M,1)
      if(type==4 || type==0)
        fprintf(fileID,'%d',M(i,:));
      else
        fprintf(fileID,'%.9f',M(i,:));
      end
      fprintf(fileID,'\n');
  end
end
