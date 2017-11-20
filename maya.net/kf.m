%kalman filter: constant velocity model

% A & H
A=[1 1; 0 1];
H=[1 0];

%Q & R
sigma_v=3;
sigma_phi=10;
R=sigma_phi;
Q=[0.25 0.5; 0.5 1];
Q=Q*sigma_v;

%[x v]
xk=[xs(1);0];
Pk=[sigma_v 0; 0 sigma_v];

xt=[];

for idx=1:length(xs)
  xt=[xt;xk];

  %测量
  zk=xs(idx);

  xk1=A*xk;
  Pk1=A*Pk*A'+Q;

  %kalman增益
  Kk=Pk*H'\(H*Pk*H'+R);
  xk1=xk1+Kk*(zk-H*xk1);

  xk=xk1;
  Pk=Pk1-Kk*H*Pk1;  
end
