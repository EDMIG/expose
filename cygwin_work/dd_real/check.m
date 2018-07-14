n=400;
m=100;

fd=fopen('A400x400.bin','r');
A=fread(fd,[n n],'double');
fclose(fd);

fd=fopen('V400x100.bin','r');
V=fread(fd,[n m],'double');
fclose(fd);