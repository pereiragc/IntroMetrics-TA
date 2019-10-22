using Statistics
using Distributions
using Plots



n1 = Normal(-1.)
n2 = Normal(1.)

pyplot();
plot(mustar -> cdf(n2, mustar),
     mustar -> 1-cdf(n1, mustar),
     -10., 10.,
     line = (2, :darkblue),
     legend = false, framestyle=:origin)

scatter!([cdf(n2, 0.)],[1-cdf(n1, 0.)], marker=(:xcross, 5, :red),
         msw=0)

savefig("out/ThresholdRisk.png")
