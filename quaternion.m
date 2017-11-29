syms wx wy wz real;
syms qw qx qy qz real;
syms q0 q1 q2 q3 real;
syms t real;
syms ex ey ez real;
syms pitch roll yaw real;

%(pitch,roll,yaw)->(qw,qx,qy,qz)
cy=cos(yaw/2);
sy=sin(yaw/2);
cr=cos(roll/2);
sr=sin(roll/2);
cp=cos(pitch/2);
sp=sin(pitch/2);

qw=cy*cr*cp+sy*sr*sp;
qx=cy*sr*cp-sy*cr*sp;
qy=cy*cr*sp-sy*sr*cp;
qz=sy*cr*sp-cy*sr*sp;

%(qw,qx,qy,qz)->(pitch,roll,yaw)
%roll x-axis roation
sinr=2*(qw*qx+qy*qz);
cosr=1-2*(qx*qx+qy*qy);
roll=atan2(sinr,cosr);

%pitch y-axis rotation
sinp=2*(qw*qy-qz*qx);
pitch=asin(sinp);

%yaw z-axis rotation
siny=2*(qw*qz+qx*qy);
cosy=1-2*(qy*qy+qz*qz);
yaw=atan2(siny,cosy);

Q=[0 -wx -wy -wz; wx 0 -wz wy; wy wz 0 -wx; wz -wy wx 0];
Q2=Q/2;

%dQ2/dwx;
Q2wx=[0 -1 0 0;1 0 0 0; 0 0 0 -1; 0 0 1 0]/2;
%dQ2/dwy;
Q2wy=[0 0 -1 0; 0 0 0 1; 1 0 0 0; 0 -1 0 0]/2;
%dQ2/dyz;
Q2wz=[0 0 0 -1;0 0 -1 0; 0 1 0 0; 1 0 0 0]/2;

W1=sqrt(wx*wx+wy*wy+wz*wz);
W2=W1*W1;
W3=W1*W1*W1;

WQ2=W1/2;

cs=cos(WQ2*t);
sn=sin(WQ2*t);

I4x4=eye(4);

Qtran=cs*I4x4+1/WQ2*sn*Q2;
Qsx=-wx*t/W1/2*sn*I4x4+(wx*t/W2*cs-2*wx/W3*sn)*Q2+2/W1*sn*Q2wx;
Qsy=-wy*t/W1/2*sn*I4x4+(wy*t/W2*cs-2*wy/W3*sn)*Q2+2/W1*sn*Q2wy;
Qsz=-wz*t/W1/2*sn*I4x4+(wz*t/W2*cs-2*wz/W3*sn)*Q2+2/W1*sn*Q2wz;

%kalman filter for constant angular velocity model
w=[wx wy wz]';
q=[qw qx qy qz]';
X=[q;w];

%更新方程
F=[Qtran*q;w];
%F微分
Fk=[Qtran Qsx*q Qsy*q Qsz*q;zeros(3,4) eye(3)];

%测量矩阵
H=[I4x4 zeros(4,3)];
