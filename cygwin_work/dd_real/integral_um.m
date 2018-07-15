syms m x eta ;

digits(100)
%eta=vpa(1/3);
%m=vpa(10);
%sqrt_eta=sqrt(1-eta);
%g=sqrt_eta*cos(m*x)./sqrt(1+eta*cos(x));

m=vpa(10);
tt=[];
for k=0:100
    k
    m=vpa(k);
    t=[];
    for i=0:999
        eta=vpa(i/1000);
        sqrt_eta=sqrt(1-eta);
        g=sqrt_eta*cos(m*x)./sqrt(1+eta*cos(x));
        r=vpaintegral(g,[0 pi], 'RelTol',1e-64,'AbsTol',0);
        %i
        t=[t r];
    end
    tt=[tt;t];
end