using CSV
using LinearAlgebra
using StatsFuns
import ForwardDiff

# Note: to run this code, point to a valid file below
cps_df = CSV.read("data/cps09mar.txt", delim="\t", header=0)

function prepare_data(cps_df, educ_idx, age_idx, wage_idx)
    # Use `cps_df` to fill `X` and `Y`, respectively the matrix of regressors
    # and dependent variable. 
    #
    # `{educ,age,wage}_idx` are the (column) indices of the respective variables
    # in the cps data



    # Pre-allocate `X`and `Y`
    X = fill(1., (size(cps_df,1), 4)) # Already fill ones
    Y = fill(0., (size(cps_df,1), 1))

    # Loop over observations
    for i in 1:size(X,1)
        educ = cps_df[i, educ_idx]
        age =  cps_df[i, age_idx]
        xp = age - educ - 6  # Check formula
        wage = cps_df[i, wage_idx]
        X[i,2] = educ
        X[i,3] = xp
        X[i,4] = xp^2/100
        Y[i] =  log(wage)
    end
    return Y, X
end


# * Housekeeping
# From Hansen's CPS documentation
educ_idx = 4
age_idx = 1
wage_idx = 5

# Get regressor/dependent variable matrices X and Y
Y, X = prepare_data(cps_df, educ_idx, age_idx, wage_idx);

# * OLS estimation
# Estimate OLS coefficients
Qxx = X'*X          # Store for later use
ols = Qxx \ (X'*Y); # OLS estimator

print("
OLS coefficients:
(education)        β_1 = $(ols[2])
(experience)       β_2 = $(ols[3])
(expereince²/100)  β_3 = $(ols[4])
(constant term)    β_4 = $(ols[1])
")



# * OLS standard errors
E = Y - X*ols; # Residuals


# Compute OLS standard errors
bread = inv(Qxx);  # Bread
meat = sum( [X[i,:]*X[i, :]'*E[i]^2 for i in 1:size(X,1)] ); # ∑ xᵢ xᵢ' êᵢ²
V_hc0 = bread * meat * bread;


# * Theta confidence intervals
# theta function
g(b, x) = b[1]/(b[2] + b[3]x/50)

# Explicit jacobian of g
J(b, x) = [1/(b[2]+b[3]*x/50)  -b[1]/(b[2]+b[3]x/50)^2 -(x/50)*b[1]/(b[2]+b[3]x/50)^2 0.]

# Compare explicit jacobian with automatic (forward) differentiation
# evaluated at OLS estimator, and experience = 20
j0 = J(ols, 20)
j1 = ForwardDiff.gradient(b -> g(b, 20), ols)


# Confidence interval
function ci(ols, Vols, x, coverage=0.90)
    thetahat = g(ols, x)

    sq_s_hat =J(ols, x)*Vols*J(ols, x)'; # This returns a matrix
    s_hat = sqrt(sq_s_hat[1,1]);
    c = norminvcdf((1+coverage)/2)


    thetahat - c * s_hat, thetahat + c * s_hat
end

xp = 20
ci(ols, V_hc0, xp, 0.9)

