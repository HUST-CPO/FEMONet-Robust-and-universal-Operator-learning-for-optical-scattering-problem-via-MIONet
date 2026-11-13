function out=BF_Ez(num,u,v)

if num==1
    out=1-u-v;
elseif num==2
    out=u;
elseif num==3
    out=v;
end

end
