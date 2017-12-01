load tracker.param.txt;
%local parameters
plocal=tracker_param(:,1:34)';
plocal_kalman=zeros(size(plocal));

sigma_v=100;
sigma_phi=10;

R=sigma_phi;
Q=[0.25 0.5;0.5 1];
Q=Q*sigma_v;

for n=1:size(plocal,1)
  A=[1 1; 0 1];
  H=[1 0];

  xs=plocal(n,:);

  xk=[xs(1);xs(2)-xs(1)];
  Pk=[sigma_v 0; 0 sigma_v];

  for idx=1:length(xs)
    plocal_kalman(n,idx)=xk(1);
    zk=xs(idx);

    xk1=A*xk;
    Pk1=A*Pk*A'+Q;

    Kk=Pk1*H'/(H*Pk1*H'+R);
    xk1=xk1+Kk*(zk-H*xk1);

    xk=xk1;
    Pk=Pk1-Kk*H*Pk1;
  end
end

g=tracker_param(:,35:end)';
g=g([1 4 5 6 2 3],:);
g_kalman=zeros(size(g));

for n=1:size(g,1)
  A=[1 1; 0 1];
  H=[1 0];

  xs=g(n,:);

  xk=[xs(1);xs(2)-xs(1)];
  Pk=[sigma_v 0; 0 sigma_v];

  for idx=1:length(xs)
    g_kalman(n,idx)=xk(1);
    zk=xs(idx);

    xk1=A*xk;
    Pk1=A*Pk*A'+Q;

    Kk=Pk1*H'/(H*Pk1*H'+R);
    xk1=xk1+Kk*(zk-H*xk1);

    xk=xk1;
    Pk=Pk1-Kk*H*Pk1;
  end
end
