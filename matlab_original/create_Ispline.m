function [G,knots]=create_Ispline(H,order,n_intknot)
% TRANSCRIPTION from thesis section 3.2.1 (pp. 69-70), listing "create_Ispline.m".
%subfunction: create_mspline
n=size(H,1);
m=size(H,2);
dimensao=order+n_intknot;
for j=1:m%corre as m variaveis de H
    h=H(:,j);
    Gaux=zeros(n,dimensao);
    %constroi a matriz pseudoindicatriz associada a var j
    x=h;
    [s,knots2]=isplines(x,order, n_intknot);
    Gaux=s;
    knots(:,j)=knots2;
    if j==1 G=Gaux; else G=[G Gaux];
    end
end
end
function [s,knots2]=isplines(x, order, n_intknot)
dim=order+n_intknot;
[G2,knots2]=create_mspline(x,order+1,n_intknot);
n=length(x);
I=zeros(n,dim);
k=order;
for i=1:dim
    for t=1:n
        if x(t)==knots2(end)
            I(t,i)=1;
        else
        j=sum(knots2<=x(t));
            if i>j
            I(t,i)=0;
            else
                if j-k<=i<=j
                m=i+1;
                I(t,i)=((knots2(m+k+1)-knots2(m))/(k+1))*G2(t,m);
                    for m=i+2:j
                    I(t,i)=I(t,i)+((knots2(m+k+1)-knots2(m))/(k+1))*G2(t,m);
                    end
                else
                    if i<j-k
                    I(t,i)=1;
                    end
                end
            end
        end
    end
end
s=I;
end
