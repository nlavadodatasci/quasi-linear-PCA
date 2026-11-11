function [G,knots]=create_mspline(H,order,n_intknot)
% TRANSCRIPTION from thesis section 3.2.1 (pp. 70-71), listing "create_mspline.m".
%subfunctions: msplines, basis, rep
n=size(H,1);
m=size(H,2);
%constroi os percentis associados aos nos interiores
p=zeros(1,n_intknot);
p(1)=100/(n_intknot+1);
for i=2:n_intknot
    p(i)=p(1)+(i-1)*p(1);
end
p=round(p);
dimensao=order+n_intknot;
for j=1:m%corre as m variaveis de H
    h=H(:,j);
    %sequencia de nos extraida de h segundo a seq de percentis
    inner_knots=prctile(h,p);
    boundary_knots=[min(h) max(h)];
    Gaux=zeros(n,dimensao);
    boundary_knots=sort(boundary_knots);
    knotsj=[rep(boundary_knots(1), order) sort(inner_knots)...
        rep(boundary_knots(2), order)];
    for i=1:n%constroi a matriz pseudoindicatriz associada a var j
        x=h(i);
        s=msplines(x,order,inner_knots,boundary_knots,knotsj);
        Gaux(i,:)=s;
    end
    knots(:,j)=knotsj;
    if j==1 G=Gaux; else G=[G Gaux];
    end
end
