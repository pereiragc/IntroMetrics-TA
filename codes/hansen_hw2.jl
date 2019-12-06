using CSV
using LinearAlgebra
using StatsFuns
import ForwardDiff
using DataFrames
using DataFramesMeta
using Printf


include("function_library.jl")


# * Load data onto memory // housekeeping
# Note: to run this code, point to a valid file below
cps_df = CSV.read("data/cps09mar.txt", delim="\t", header=0)

# Variables:
varnames =  [
    "age",
    "female",
    "hisp",
    "education",
    "earnings",
    "hours",
    "week",
    "union",
    "uncov",
    "region",
    "race",
    "marital"
];
names!(cps_df, Symbol.(varnames)); # Set names

cps_q29 = @linq cps_df |> where(:hisp .== 1, :female .== 0, :race .== 1) |>
                           select(:earnings, :age, :education);

@with cps_q29 begin
    cps_q29.experience = max.(:age - :education .-6, 0)
end;

@with cps_q29 begin
    cps_q29.experiencesq = :experience.^2/100
    cps_q29.intercept = fill(1, nrow(cps_q29))
end;

# * Exercises in Q29


# Convert to matrix ------------------------------------------------------------
Y = log.(cps_q29[:, :earnings]);
vec_varnames = [:education, :experience, :experiencesq, :intercept]
X = convert(Matrix,
            cps_q29[:, vec_varnames]);
# ------------------------------------------------------------------------------

# ** OLS + pretty print
est_ols, resid, invXpX = ols(X,Y)
avar_ols  = hc0_avar(X,Y,resid, invXpX)

# Report V̂
V_hc0 = avar_ols/size(X,1)
10^5 * V_hc0

# Report standard errors
se(avar_ols, size(X,1))

coefs = fill(NaN, (length(est_ols), 1))
serrs = fill(NaN, (length(est_ols), 1))

coefs .= est_ols
serrs .= se(avar_ols, size(X,1))

print(prettyprint(coefs, serrs, vec_varnames))



# ** Theta confidence intervals
# theta function
g(b, x) = b[1]/(b[2] + b[3]x/50)

# Explicit jacobian of g
J(b, x) = [1/(b[2]+b[3]*x/50)  -b[1]/(b[2]+b[3]x/50)^2 -(x/50)*b[1]/(b[2]+b[3]x/50)^2 0.]


# Optional ----------------------------------------------------------------------
# Compare explicit jacobian with automatic (forward) differentiation
# evaluated at OLS estimator, and experience = 20

j0 = J(est_ols, 20)
j1 = ForwardDiff.gradient(b -> g(b, 20), est_ols)
# -------------------------------------------------------------------------------

# Confidence interval
function ci_theta(est_ols, Vest_ols, x, coverage=0.90)
    thetahat = g(est_ols, x)

    sq_s_hat =J(est_ols, x)*Vest_ols*J(est_ols, x)'; # This returns a matrix
    s_hat = sqrt(sq_s_hat[1,1]);
    c = norminvcdf((1+coverage)/2)

    println("Std err: $s_hat")
    thetahat - c * s_hat, thetahat + c * s_hat
end

xp = 21
ci_theta(est_ols, V_hc0, xp, 0.9)


# ** Regression function
xx = [12, 20, 4., 1.]
ss = sqrt(xx'*V_hc0*xx)
println("Regression interval: [$((xx'*est_ols)[1] - ss*1.96), $((xx'*est_ols)[1] + ss*1.96)]")


# ** Forecast interval 
E = resid
sigsq = E'*E/(length(E) - size(X, 2)); # Estimate error variance
tt = sqrt(ss^2 + sigsq[1]);

lower_endpt = (xx'*est_ols)[1] - tt*1.96
upper_endpt = (xx'*est_ols)[1] + tt*1.96

println("Forecast interval (log wage): [$lower_endpt, $upper_endpt]")
println("Forecast interval (wage): [$(exp(lower_endpt)), $(exp(upper_endpt))]")
