load 'nets.mat'

fd1=fopen('net1.bin','w+b');
fd2=fopen('net2.bin','w+b');
fd3=fopen('net3.bin','w+b');

for idx=1:31

  n1=net{idx};
  IW=n1.IW{1};
  LW=n1.LW{2,1};
  b1=n1.b{1};
  b2=n1.b{2};

  in_setting=n1.input.processSettings{1};
  out_setting=n1.output.processSettings{1};

  in_xmax=in_setting.xmax;
  in_xmin=in_setting.xmin;
  in_ymax=in_setting.ymax;
  in_ymin=in_setting.ymin;

  out_xmax=out_setting.xmax;
  out_xmin=out_setting.xmin;
  out_ymax=out_setting.ymax;
  out_ymin=out_setting.ymin;

  fd=fd1;

  fwrite(fd,IW,'double');
  fwrite(fd,LW,'double');
  fwrite(fd,b1,'double');
  fwrite(fd,b2,'double');

  fwrite(fd,in_xmax,'double');
  fwrite(fd,in_xmin,'double');
  fwrite(fd,in_ymax,'double');
  fwrite(fd,in_ymin,'double');

  fwrite(fd,out_xmax,'double');
  fwrite(fd,out_xmin,'double');
  fwrite(fd,out_ymax,'double');
  fwrite(fd,out_ymin,'double');

  n1=net2{idx};
  fd=fd2;
  %.....
  n1=net3{idx};
  fd=fd3;
  %............


end

fclose(fd1);
fclose(fd2);
fclose(fd3);
