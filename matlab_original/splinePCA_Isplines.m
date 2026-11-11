function [F,X,A,eigval,knots,coef]=splinePCA_Isplines(H)
% TRANSCRIPTION of the MATLAB listings printed in section 3.2 of
% N. Lavado (2012), "Extensoes via splines da Analise em Componentes
% Principais", PhD thesis, ISCTE-IUL (thesis pp. 68-73).
% The thesis prints this function in three parts (splinePCA_Isplines1.m,
% 2.m, 3.m). Lines marked "NOT IN THESIS" are not printed there and were
% added only so that the file is syntactically complete; see docs/SPEC.md.
%
%Input: H - data matrix n,m
%Output:
%F - matriz dos dados transformados
%X - objects scores nao lineares, matrix n por p=nCP
%A - matriz dos loadings
%eigval - valores proprios
%knots - seq. de nos
%coef - coefs das cl da base de splines
%Sub functions
%   [G,knots]=create_Ispline(H,order,n_intknot)
%   MatLab functions: pcacov, svd
%28/03/2012 - Nuno Lavado .: nlavado@isec.pt
m=size(H,2);
n=size(H,1);
nCP=input('Dimensions in solution:');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Inicializacao
H=zscore(H);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%constroi a matriz difusa para H
%Ordem dos Isplines (coincide com o grau para esta base)
order=input('Spline degree:');
n_intknot=input('Interior knots:');
[G,knots]=create_Ispline(H,order,n_intknot);
dimensao=order+n_intknot;
G=zscore(G);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SCORES aleatorios
%X=rand(n,nCP);
X=G(:,1:nCP);
max_it = 1000;
it=1;
tol=1;
coef=zeros(m*dimensao,1);
F=zeros(n,m);
eigval=zeros(nCP,1);
A=corr(H,X);
communality=sum(A.^2,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---- part 2 (splinePCA_Isplines2.m) ----
while tol>0.1*10^(-5) &&  it <=max_it;
    jw=1;
    jb=dimensao;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Actualizacao das quantificacoes
    for j=1:m
       y=X*A(j,:)';
       coef(jw:jb,1)=(1/n)*(G(:,jw:jb)'*G(:,jw:jb))^(-1)*G(:,jw:jb)'*y;
       F(:,j)=G(:,jw:jb)*coef(jw:jb,1);
       F(:,j)=F(:,j)-mean(F(:,j));
       jw=jw+dimensao;
       jb=jb+dimensao;
    end
    ssq=diag(sum(F.^2,1));
    F = F*ssq^(-1/2).*sqrt(n);%F centrado e norma sqrt(n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---- part 3 (splinePCA_Isplines3.m) ----
%Actualizacao dos scores
    evprev=eigval;
    [eigvec,eigval] = pcacov(F'*F);
    eigval=eigval(1:nCP,1)./n;
    A=eigvec(:,1:nCP)*diag(eigval)^0.5;
    %A matriz dos loadings m por nCP
    X=F*A;
    [K,lambda,W] = svd(X,0);
    X=sqrt(n)*K*W';
%Exemplo com uma CP
%   eigval=eigval(1,1)/n;
%   a=eigvec(:,1).*eigval^0.5;
%   X=sum(F*diag(a),2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---- NOT IN THESIS: the convergence test and loop end are not printed.
% Section 3.2.4 says the test compares two successive values of the loss
% function against 0.1e-5. The printed code also keeps evprev=eigval, so the
% original may instead have tested the change in the eigenvalues; this is
% not known. Reconstructed here (loss-based) as:
    loss=sum(sum((X*A'-F).^2))/n;
    if it>1, tol=abs(lossprev-loss); end
    lossprev=loss;
    it=it+1;
end
