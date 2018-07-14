n=231;
nlocal=[47 47 47 47 43];

fd=fopen('X.bin','r');
X=fread(fd,n,'double');
fclose(fd);

fd=fopen('Y.bin','r');
Y=fread(fd,n,'double');
fclose(fd);

fd=fopen('A0.bin','r');
A0=fread(fd,[nlocal(1) n],'double');
fclose(fd);

fd=fopen('A1.bin','r');
A1=fread(fd,[nlocal(2) n],'double');
fclose(fd);

fd=fopen('A2.bin','r');
A2=fread(fd,[nlocal(3) n],'double');
fclose(fd);

fd=fopen('A3.bin','r');
A3=fread(fd,[nlocal(4) n],'double');
fclose(fd);

fd=fopen('A4.bin','r');
A4=fread(fd,[nlocal(5) n],'double');
fclose(fd);

A=[A0;A1;A2;A3;A4];

