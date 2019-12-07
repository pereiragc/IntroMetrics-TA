using CSV
using LinearAlgebra
using StatsFuns
import ForwardDiff
using DataFrames
using Printf
using DataFramesMeta


include("function_library.jl")

# * Load data onto memory // housekeeping

# Note: to run this code, point to valid files below
cps_df = CSV.read("data/cps09mar.txt", delim="\t", header=0)
invest_df = CSV.read("data/Invest1993.txt")
nerlove_q926 = CSV.read("data/Nerlove1963.txt")


# ** CPS
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
]
names!(cps_df, Symbol.(varnames)) # Set names
cps_q819 = @linq cps_df |> where(:hisp .== 1, :female .== 0, :race .== 1) |>
                           select(:earnings, :education, :age, :marital);

# Add necessary columns ---------------------------------------------------------
@with cps_q819 begin
    cps_q819.intercept = fill(1, nrow(cps_q819))
    cps_q819.logwage = log.(:earnings)
    cps_q819.experience = max.(:age - :education .- 6, 0)
    cps_q819.married1 = :marital .== 1
    cps_q819.married2 = :marital .== 2
    cps_q819.married3 = :marital .== 3
    cps_q819.widowed =  :marital .== 4
    cps_q819.divorced = :marital .== 5
    cps_q819.separated = :marital .== 6
end;

@with cps_q819 begin
    cps_q819.experiencesq_scale = :experience.^2 / 100
end
# -------------------------------------------------------------------------------

# ** Investment
invest_q925 = @linq invest_df |> where(:year .== 1987) |>
                           select(:year, :inva, :vala, :cfa, :debta);
@with invest_q925 begin
    invest_q925.intercept = fill(1, nrow(invest_q925))
end;

# ** Nerlove

@with nerlove_q926 begin
    nerlove_q926.intercept = fill(1, nrow(nerlove_q926))
    nerlove_q926.logC = log.(:Cost)
    nerlove_q926.logQ = log.(:output)
    nerlove_q926.logPL = log.(:Plabor)
    nerlove_q926.logPK = log.(:Pcapital)
    nerlove_q926.logPF = log.(:Pfuel)
end




# * Chapter 8, Q19

# From data frame to matrix -----------------------------------------------------
vec_varnames = [:education, :experience, :experiencesq_scale, :married1,
            :married2, :married3, :widowed, :divorced,
            :separated, :intercept]
Y, X = dataframe_to_mat(cps_q819, :logwage, vec_varnames)
# -------------------------------------------------------------------------------

# ** OLS
est_ols,resid,serrs,avar_ols,invXpX = estimate(Y, X, OLS);

# ** CLS
W = X'X/length(Y)

# Constrained least squares (b)
n_constraints = 2;
R = fill(0, (size(X, 2), n_constraints));
R[4,1] = 1; R[7,1] = -1; # First constraint
R[8,2] = 1; R[9,2] = -1; # Second constraint

c = fill(0, n_constraints)
est_cls, lm_cls, avar_cls = md_linear(Y, X, W, R, c, est_ols, avar_ols)

# ** Efficient minimum distance
W = inv(avar_ols)
est_md, lm, avar_md = md_linear(Y, X, W, R, c, est_ols, avar_ols)

# ** Inequality constraint
# Solve the problem with the extra constraint β₂ = β₃
S = fill(0, (size(X,2), n_constraints + 1));
S[:, 1:2] = R;
S[2,3] = 1
S[3,3] = 1

est_md_ptE, lm_1, avar_md_1 = md_linear(Y, X, W, S, fill(0, n_constraints + 1), est_ols, avar_ols)

# ** Report
n = length(Y)
coefs = [est_ols est_cls est_md est_md_ptE]
serrs = [se(avar_ols, n) se(avar_cls, n) se(avar_md, n) se(avar_md_ptE, n)]

print(prettyprint(coefs, serrs, vec_varnames))

# * Chapter 9, Q25

vec_varnames = [:vala, #r morgulis
                :cfa, :debta, :intercept]
Y, X = dataframe_to_mat(invest_q925, :inva, vec_varnames)

# Construct quadratic regressors
XX, quad_varnames = quadratic_expand(X, String.(vec_varnames))


est_ols,resid,serrs,avar_ols,invXpX = estimate(Y, X, OLS);
tstat = est_ols[1]/serrs[1]  # T-statistic of `vala`

# Build restriction matrix for part (c)
R = fill(0, (size(X,2), 2))
R[2,1]=1
R[3,2]=1
w = wald_stat(R, R'*zero(est_ols), est_ols, avar_ols, length(Y))

# Quadratic components (part e)
est_ols_quad,resid_quad,serrs_quad, avar_ols_quad,invXpX_quad=estimate(Y, XX, OLS);

# Build restriction matrix (part e)
regressors_quadratic = [1,2,3,5,6,8] # which regressors are qudratic?
R_quad = fill(0, (size(XX,2),length(regressors_quadratic)))
for (j_quad, quad_term) in enumerate(regressors_quadratic)
    R_quad[quad_term, j_quad] = 1
end

w_quad = wald_stat(R_quad, R_quad'*zero(est_ols_quad), est_ols_quad,
                   avar_ols_quad, length(Y))


# ** Report


# *** Part (b)
z = norminvcdf(0.975)
for i in eachindex(est_ols)
    coef = est_ols[i]
    sd = serrs[i]

    str_print = " | "*String(vec_varnames[i])*" | "
    str_print *= @sprintf "[% 1.4f, % 1.4f]" (coef-z*sd) (coef+z*sd)
    println(str_print)
end

# *** Part (c)
crit = chisqinvcdf(2, 0.95)
println("Reject coef of cfa and debta are jointly zero if  ")
println("wald statistic w=$w > $crit.")
println("In the current case: reject = $(w > crit)")

# *** Part (d)
print(prettyprint(est_ols_quad, serrs_quad, quad_varnames,22))
println("Reject quadratic coefficients are jointly zero if  ")
println("wald statistic w=$w_quad > $crit.")
println("In the current case: reject = $(w > crit)")


# * Chapter 9, Q26
vec_varnames = [:intercept, :logQ, :logPL, :logPK, :logPF]
Y, X = dataframe_to_mat(nerlove_q926, :logC, vec_varnames)

# ** OLS
est_ols,resid,serrs,avar_ols,invXpX = estimate(Y, X, OLS;
                                               varnames=vec_varnames,
                                               print_table=true);
# ** CLS/EMD
R = fill(0, length(est_ols));
R[3]=R[4]=R[5]=1;
c=1

est_cls, lm_cls, avar_cls = md_linear(Y, X, X'X, R, c, est_ols, avar_ols)
est_md, lm_md, avar_md = md_linear(Y, X, inv(avar_ols), R, c, est_ols, avar_ols)

# Wald statistic
w = wald_stat(R, c, est_ols, avar_ols, length(Y))

# Minimum distance statistic
md = mdstat(est_md, est_ols, inv(avar_ols), length(Y))

# * Chapter 9, Q27
