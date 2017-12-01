%kalman filter for quaternion with constant angular velocity model

function [pitch_new,yaw_new,roll_new]=ekf_quaternion()

load 'tracker.param.txt';
g=tracker_param(:,35:end);

%全局参数：(s,pitch,yaw,roll,tx,ty)
g=g([1 4 5 6 2 3],:);

pitch=g(2,:)';
yaw=g(3,:)';
roll=g(4,:)';

%测量矩阵
H=[eye(4) zeros(4,3)];

%(qw,qx,qy,qz)
Q1=calc_quaternion(pitch(1),yaw(1),roll(1));
Q2=calc_quaternion(pitch(2),yaw(2),roll(2));

%计算初始角速度
wx=2*(Q2(2)-Q1(2));
wy=2*(Q2(3)-Q1(3));
wz=2*(Q2(4)-Q1(4));

%t=0,初始状态
xk=[Q1(:);wx;wy;wz];

sigma_v=10;
sigma_phi=100;
%状态covariance matrix
Q=eye(7)*sigma_v;
Q(5,5)=20；
Q(6,6)=20;
Q(7,7)=20;
%测量状态covariance matrix
R=eye(4)*sigma_phi;

Pk=Q;

nf=length(pitch);

pitch_new=zeros(1,nf);
yaw_new=zeros(1,nf);
roll_new=zeros(1,nf);

for n=1:nf
  xk1=Ft(xk);

  %(pitch,yaw,roll)->quaternion
  zk=calc_quaternion(pitch(n),yaw(n),roll(n));
  A=DF(xk);

  Pk1=A*Pk*A'+Q;

  %kalman 增益因子
  Kk=Pk*（H'/(H*Pk*H'+R));

  %state correction
  xk1=xk1+Kk*(zk-H*xk1);
  Pk=Pk1-Kk*H*Pk1;
  %quaternion -> (pitch,yaw,roll)
  [p,y,r]=calc_euler(xk(1:4));
  pitch_new(n)=p;
  yaw_new(n)=y;
  roll_new(n)=r;
end

figure,hold;
plot(pitch_new);plot(pitch,'r');
figure,hold;
plot(yaw_new);plot(yaw,'r');
figure,hold;
plot(roll_new);plot(roll,'r');
xk


%(pitch,yaw,roll)->quaternion
function q=calc_quaternion(pitch,yaw,roll)

  cy=cos(yaw/2);
  sy=sin(yaw/2);
  cr=cos(roll/2);
  sr=sin(roll/2);
  cp=cos(pitch/2);
  sp=sin(pitch/2);

  qw=cy*cr*cp+sy*sr*sp;
  qx=cy*sr*cp-sy*cr*sp;
  qy=cy*cr*sp-sy*sr*cp;
  qz=sy*cr*cp-cy*sr*sp;

  q=[qw qx qy qz]';


%quaternion->(pitch,yaw,roll)
function [pitch,yaw,roll]=calc_euler(q)

  qw=q(1);
  qx=q(2);
  qy=q(3);
  qz=q(4);

  %roll x-axis rotation
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

%状态方程
function Xt=Ft(Xt0)
  Q=Xt0(1:4);
  wx=Xt0(5);
  wy=Xt0(6);
  wz=Xt0(7);

  %skew matrix
  Q2=0.5*[0 -wx -wy -wz; wx 0 -wz wy; wy wz 0 -wx; wz -wy wx 0];
  WQ2=0.5*sqrt(wx*wx+wy*wy+wz*wz);

  t=1;
  cs=cos(WQ2*t);
  sn=sin(WQ2*t);
  Qtran=cs*eye(4)+sn/WQ2*Q2;
  Xt=[Qtran*Q(:);wx;wy;wz];

%DF 状态函数Jacobian
function Fk=DF(Xt0)

  Q=Xt0(1:4);
  wx=Xt0(5);
  wy=Xt0(6);
  wz=Xt0(7);

  %skew matrix
  Q2=0.5*[0 -wx -wy -wz; wx 0 -wz wy; wy wz 0 -wx; wz -wy wx 0];
  %dQ2/dwx
  Q2wx=[0 -1 0 0;1 0 0 0;0 0 0 -1; 0 0 1 0]/2;
  %dQ2/dwy
  Q2wy=[0 0 -1 0;0 0 0 1;1 0 0 0; 0 -1 0 0]/2;
  %dQ2/dwz
  Q2wz=[0 0 0 -1;0 0 -1 0;0 1 0 0;1 0 0 0]/2;

  W1=sqrt(wx*wx+wy*wy+wz*wz);
  W2=W1*W1;
  W3=W1*W1*W1;
  WQ2=0.5*W1;


  t=1;
  cs=cos(WQ2*t);
  sn=sin(WQ2*t);

  Qtran=cs*eye(4)+sn/WQ2*Q2;

  Qsx=-wx*t/W1/2*sn*eye(4)+(wx*t/W2*cs-2*wx/W3*sn)*Q2+2/W1*sn*Q2wx;
  Qsy=-wy*t/W1/2*sn*eye(4)+(wy*t/W2*cs-2*wy/W3*sn)*Q2+2/W1*sn*Q2wy;
  Qsz=-wz*t/W1/2*sn*eye(4)+(wz*t/W2*cs-2*wz/W3*sn)*Q2+2/W1*sn*Q2wz;

  Fk=[Qtran Qsx*Q(:) Qsy*Q(:) Qsz*Q(:); zeros(3,4) eye(3)];
