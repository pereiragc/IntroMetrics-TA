
# * Estimation
" Compute OLS estimator, residuals and (X'X)⁻¹ "
function ols(X,Y)
    invXpX = pinv(X'*X)
    ols_est = invXpX*X'*Y
    resid = Y - X*ols_est
    return ols_est, resid, invXpX
end

" Calculate HC0 estimate of OLS asymptotic variance "
hc0_avar(X, Y, resid, invXpX) = invXpX * hc0_meat(X, resid) * invXpX * (length(Y)^2)
function hc0_avar(X, Y)
    ols_est, resid, invXpX = ols(X,Y)
    hc0_avar(X, Y, resid, invXpX)
end

" 'Meat' term of HC0 variance "
function hc0_meat(X, resid)
    k = size(X,2)
    mat_meat = fill(0., (k,k))
    for i in eachindex(resid)
        mat_meat += X[i, :] * X[i, :]' * resid[i]^2
    end
    return mat_meat/length(resid)
end


"""
     md_linear(Y, X, W, R, c, est_ols, avar_ols)

Compute the minimizer of J(β) = (β̂ - β)' W (β̂ -β) subject to R'β = c. The term β̂
above is passed as `est_ols`. Also compute the estimator of the associated
asymptotic variance and the Lagrange multiplier associated with the constraints.
"""
function md_linear(Y, X, W, R, c, est_ols, avar_ols)
    n = length(Y)
    invW = inv(W)

    bread_md = inv(R' * invW * R)
    meat_md = R' * avar_ols * R


    lm  = n * bread_md * (R' * est_ols - c)
    est_md = est_ols - invW * R * lm / n


    # Asymptotic variance
    V = avar_ols
    V -= invW * R * bread_md * R' * avar_ols
    V -= avar_ols * R * bread_md * R'*invW
    V += invW * R * bread_md * meat_md * bread_md * R' * invW

    est_md, lm, V
end


function cap_string(symb_str, n)
    str = String(symb_str)
    lpad(str[1:min(n, lastindex(str))], n)
end

" Build a table from column-wise stacked vectors of coefficients and matrices "
function prettyprint(coefs::Matrix{Float64}, serrs::Matrix{Float64},
                     vec_varnames, cap_strname=15)
    str0 = ""
    for i in 1:size(coefs,1)
        strtmp_coef = "| "*cap_string(vec_varnames[i], cap_strname)*" | "
        strtmp_se = "| "*" "^cap_strname*" | "
        for k in 1:size(coefs, 2)
            strtmp_coef *= @sprintf "  % 2.5f  |" coefs[i, k]
            strtmp_se *= @sprintf " (%2.5f)  |" serrs[i, k]
        end
        str0 *= strtmp_coef * "\n"
        str0 *= strtmp_se * "\n"
    end
    return str0
end

function prettyprint(coefs::Vector{Float64}, serrs::Vector{Float64}, vec_varnames, cap_strname=15)
    n = length(coefs)
    # Build matrices based on arrays
    mcoefs = fill(0., (n, 1))
    mserrs = fill(0., (n, 1))

    mcoefs .= coefs
    mserrs .= serrs

    prettyprint(mcoefs, mserrs, vec_varnames, cap_strname)
end

" Finds standard errors from asymptotic variance matrix "
se(avarmat, n) = sqrt.(diag(avarmat))/sqrt(n)

function quadratic_expand(X, labels)
    k = size(X,2)
    n = size(X, 1)

    expanded_ncols = k*(k+1)÷2

    XX = fill(zero(eltype(X)), (n, expanded_ncols))
    newlabels = fill("", expanded_ncols)

    p = 1
    for i in 1:k
        for j in i:k
            XX[:, p] .= X[:, j].*X[:, i]
            newlabels[p] = "$(labels[j]) × $(labels[i])"
            p += 1
        end
    end
    return XX, newlabels
end

function dataframe_to_mat(df, yname, xnames)
    Y = df[:, yname]
    X = convert(Matrix,
                df[:, xnames])
    return Y, X
end


struct OLS end
struct CLS end
struct MD end

function estimate(Y, X, ::Type{OLS}; varnames=nothing, print_table=false)
    est_ols, resid, invXpX = ols(X, Y);

    avar_ols = hc0_avar(X, Y, resid, invXpX)
    serrs = se(avar_ols, length(Y))

    if print_table
        isnothing(vec_varnames) &&
            error("In order to print a table you need to supply `vec_varnames`")

        print(prettyprint(est_ols, serrs,vec_varnames))
    end

    est_ols, resid, serrs, avar_ols, invXpX
end

mdstat(b0, bols, W, n) = n*(b0 - bols)'*W*(b0 - bols)


" Wald statistic from linear constraint R'β = c"
function wald_stat(R, c, vec_coef, V, n)
    diff_theta = (R'*vec_coef-c)
    meat=inv(R'*V*R)
    n*diff_theta'*meat*diff_theta
end
