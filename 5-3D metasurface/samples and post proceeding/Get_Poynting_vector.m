function S = Get_Poynting_vector(Ex,Ey,Ez,Hx,Hy,Hz)



S = zeros(3,max(size(Ex)));

for n = 1 : max(size(Ex))

        S(1,n) = 0.5 * real ( (Ey(1,n) * conj(Hz(1,n)) - Ez(1,n) * conj(Hy(1,n))) );
        S(2,n) = 0.5 * real ( (Ez(2,n) * conj(Hx(2,n)) - Ex(2,n) * conj(Hz(2,n))) );
        S(3,n) = 0.5 * real ( (Ex(3,n) * conj(Hy(3,n)) - Ey(3,n) * conj(Hx(3,n))) );
end


end