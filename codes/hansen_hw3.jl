using CSV
using LinearAlgebra
using StatsFuns
import ForwardDiff
using DataFrames
using Printf
using DataFramesMeta


include("function_library.jl")

# * Load data onto memory // housekeeping
# Load data sets used throughout the homework.

# Note: to run this code, point to valid files below
cps_df = CSV.read("data/cps09mar.txt", delim="\t", header=0)
invest_df = CSV.read("data/Invest1993.txt") # Has column name headers



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

# * Chapter 8, Q19

# From data frame to matrix -----------------------------------------------------
vec_varnames = [:education, :experience, :experiencesq_scale, :married1,
            :married2, :married3, :widowed, :divorced,
            :separated, :intercept]
Y = log.(cps_q819[:, :earnings])
X = convert(Matrix,
            cps_q819[:, vec_varnames])
# -------------------------------------------------------------------------------



# ** OLS
# Save ols estimator, residuals and X'X inverse
est_ols, resid, invXpX = ols(X, Y);
est_ols

# Asymptotic variance
avar_ols = hc0_avar(X, Y, resid, invXpX)

# ** CLS
W = X'X/length(Y)

# Constrained least squares (b)
n_constraints = 2;
R = fill(0, (size(X, 2), n_constraints));
R[4,1] = 1; R[7,1] = -1;
R[8,2] = 1; R[9,2] = -1;

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

coefs = [est_ols est_cls est_md est_md_ptE]


n = length(Y)
serrs = [se(avar_ols, n) se(avar_cls, n) se(avar_md, n) se(avar_md_ptE, n)]


print(prettyprint(coefs, serrs, vec_varnames))

# * Chapter 9, Q25

# From data frame to matrix -----------------------------------------------------
vec_varnames = [:vala, #r morgulis
                :cfa, :debta, :intercept]
Y = log.(invest_q925[:,:inva])
X = convert(Matrix,
            invest_q925[:, vec_varnames])
XX, mod_varnames = quadratic_expand(X, String.(vec_varnames))


# -------------------------------------------------------------------------------

est_ols, resid, invXpX = ols(X, Y);
avar_ols = hc0_avar(X, Y, resid, invXpX)

# ** Report
print(prettyprint(est_ols, se(avar_ols, length(Y)), vec_varnames))
