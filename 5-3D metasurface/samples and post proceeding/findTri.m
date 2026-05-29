function out=findTri(index,mesh)

nbr=length(index);

index2=index(1);
out=find(mesh.DomainOfTri==index2);

if nbr>1
    for n=2:nbr
        index2=index(n);
        index3=find(mesh.DomainOfTri==index2);
        out=[out;index3];
    end
end

out=sort(out);

end