# Draw a 2-d simplex along with the pmf's induced by binomials


using Plots
pyplot(); # Invoke pyplot backend 

tri_x = [0., 0., 1.]
tri_y = [0., 1., 0.]
plot(tri_x,
     tri_y,
     seriestype = :shape, fill=(:lightblue, 0.3), label="Possible PMFs", size=(1200,900))


# Binomial parametric family

n = 2
bin_family(p,k)=binomial(n, k)*p^k*(1-p)^(n-k)

# Plot parametric curve such as (x(p), y(p)):
plot!(p -> bin_family(p, 1), # x(p)  [k=2]
      p -> bin_family(p, 2), # y(p)  [k=2]
      0., 1.,
      line=(2,:darkred),
      legend=true, label="Binomial")

savefig("out/example.png")
