tx=tdata_maya(:,1:3:end)';
ty=tdata_maya(:,2:3:end)';
tz=tdata_maya(:,3:3:end)';

for in=1:300
  fn=sprintf('./maya/%4.4d.txt', in);
  txy=[tx(:,in) ty(:,in)];
  save(fn,'txy','-ascii');
end


obj1=VideoReader('e:/1109/1.avi');
obj2=VideoReader('e:/1109/2.avi');
obj3=VideoReader('e:/1109/3.avi');

outV=VideoWriter('e:/1109/out.mp4', 'MPEG-4');
open(outV);

for k=1:300
  img1=read(obj1,k);
  img2=read(obj2,k);
  img3=read(obj3,k);
  img1=imresize(img1,0.5);
  img2=imresize(img2,0.5);
  img3=imresize(img3,[size(img1,1) size(img1,2)]);
  img=[img1 img2 img3];
  writeVideo(outV,img);
end

close(outV);

figure,hold;
for in=1:300
  plot(240-tx(32:36,in), 120-ty(32:36,in),'ro');pause(0.4);
end

obj=VideoReader('e:/1109/video_20171109_144525.mp4')
outV=videoWriter('e:/1109/liu.avi')
open(outV)
for k=1:819
  img=read(obj,k);img=imrotate(img,90);
  writeVideo(outV,img);
end
