load 't0.txt'
Loda 't3.txt'
load 't5.txt'
load 't7.txt'
load 't10.txt'

%踢掉第一个BS权重，剩下的是点坐标
%一个点有xyz三个世界坐标
x0=t0(:,2:end);

%剔除两个点
x0([7 8 9 19 20 21],:)=[];

y0=t0(:,1)';

%把中性表情加到向量内
x0=[x0;repmat(x0(:,1),1,size(x0,2))];

%x3,y3,x5,y5,x7,y7,x10,y10, 一次生成

xx=[x0 x3 x5 x10];
yy=[y0 y3 y5 y10];

%训练10个神经元的fitting网络
net=fitnet(10);
net=train(net,xx,yy);
