function out=getBF(type,num,u,v,w)

if type==1  %1-(1,0,0) 2-(0,1,0) 3-(0,0,1)
    if num==1 %12
        out=[-v;u;0];
    elseif num==2
        out=[-w;0;u];
    elseif num==3
        out=[-1+v+w;-u;-u];
    elseif num==4
        out=[0;-w;v];
    elseif num==5
        out=[-v;-1+u+w;-v];
    elseif num==6
        out=[-w;-w;-1+u+v];
    end
elseif type==2
    if num==1
        out=[0;0;2];
    elseif num==2
        out=[0;-2;0];
    elseif num==3
        out=[0;2;-2];
    elseif num==4
        out=[2;0;0];
    elseif num==5
        out=[-2;0;2];
    elseif num==6
        out=[2;-2;0];
    end
end

end