function s=msplines(x,order, inner_knots, boundary_knots,knots)
% TRANSCRIPTION from thesis section 3.2.1 (pp. 71-72), listing "sub_msplines.m".
% (In MATLAB each of the three functions below must live in its own file or
% be appended as local functions to create_mspline.m.)
%subfunctions: basis, rep
np=order+size(inner_knots,2);
s=rep(0, np);
for i=1:np
        s(i)=basis(x, order, i, knots);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y= basis(x, order, i, knots)
    if order==1
        if((x<knots(i+1))&&(x>=knots(i))) y=1/(knots(i+1)-knots(i));
        else y=0;
        end
    else
        if((knots(i+order)-knots(i))==0)
            y=0;
        else
        y=(order*((x-knots(i))*basis(x, (order-1), i, knots)...
            +(knots(i+order)-x)*basis(x,(order-1),(i+1),knots)))/...
            ((order-1)*(knots(i+order)-knots(i)));
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=rep(x,times)
a=x;
for i=2:times
    a=horzcat(a,x);
end
