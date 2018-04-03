module uncons

	# imports: which packages are we going to use in this module?
	using Distributions, Optim, Plots, DataFrames

	"""
    `input(prompt::AbstractString="")`

    Read a string from STDIN. The trailing newline is stripped.

    The prompt string, if given, is printed to standard output without a
    trailing newline before reading input.
    """
    #function input(prompt::AbstractString="")
    #    print(prompt)
    #    return chomp(readline())
    #end

    #export maximize_like_grad, runall, makedata

	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm)
	# true coeff vector, number of obs, data matrix X, response vector y, and a type of parametric distribution for G.
	beta = [1; 1.5; -0.5]

	function makedata(n=10_000)
		srand(1234)
		numobs = n
		σ = Matrix([1. 0. 0.;
			0. 1. 0.;
			0. 0. 1.])
		μ = Vector([0.;0.;0.])
		dist = MvNormal(μ, σ)
		X = transpose(rand(dist, n))
		d = Normal()
		function Φ(sthg)
			return(cdf(d,sthg))
		end
		Prob = [Φ(transpose(X[i,:])* beta) for i in 1:n]
		Uni = Uniform(0,1)
		y = [rand(Uni) < Prob[i] for i in 1:n]
		y = Float64.(y)
		return Dict("beta"=>beta, "n"=>numobs, "X"=>X, "y"=>y, "dist"=>d)
	end


	# log likelihood function at x
	# function loglik(betas::Vector,X::Matrix,y::Vector,distrib::UnivariateDistribution)
	function loglik(betas::Vector, d::Dict)
		X = d["X"]
		y = d["y"]
		n = d["n"]
		dist = Normal()
		function Φ(sthg)
			return(cdf(dist,sthg))
		end
		objective = sum([y[i] * log(Φ(transpose(X[i, :]) * betas)) + (1-y[i]) * log(1 - Φ(transpose(X[i, :]) * betas)) for i in 1:n])
		return objective
	end

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global maximum of the likelihood at the true value.
	function plotLike(n = 10000)
		data = makedata()
		truev = [loglik(beta, data)]
		beta1 = ones(301)
		beta2 = 1.5 * ones(301)
		beta3 = -0.5 * ones(301)
		params = collect(-1:0.01:2)
		x1 = hcat(params, beta2, beta3)
		x2 = hcat(beta1, params, beta3)
		x3 = hcat(beta1, beta2, params)
		y1 = [loglik(x1[i,:], data) for i in 1:301]
		y2 = [loglik(x2[i,:], data) for i in 1:301]
		y3 = [loglik(x3[i,:], data) for i in 1:301]
		plot1 = plot(params, y1, labels = "beta_1")
		scatter!(plot1, [1], truev, labels = "true value")
		plot2 = plot(params, y2, labels = "beta_2")
		scatter!(plot2, [1.5], truev, labels = "true value")
		plot3 = plot(params, y3, labels = "beta_3")
		scatter!(plot3, [-0.5], truev, labels = "true value")
		plot_total = plot(plot1, plot2, plot3, layout=(1,3))
	end

	function maximize_like(x0=[0.8,1.0,-0.1],meth=NelderMead())
		d = makedata(10000)
		res = optimize(arg->-loglik(arg,d),x0,meth, Optim.Options(iterations = 500,g_tol=1e-20))
		return res
	end
	function maximize_like_helpNM(x0=[ 1; 1.5; -0.5 ],meth=NelderMead())
		d = makedata(10000)
		res = optimize(arg->-loglik(arg,d),x0,meth, Optim.Options(iterations = 500,g_tol=1e-20))
		return res
	end

	#gradient of the likelihood at x
	function grad!(storage::Vector,betas::Vector,d)
		xbeta     = d["X"]*betas	# (n,1)
		G_xbeta   = cdf.(d["dist"],xbeta)	# (n,1)
		g_xbeta   = pdf.(d["dist"],xbeta)	# (n,1)
		storage[:]= -sum((d["y"] .* g_xbeta ./ G_xbeta - (1-d["y"]) .* g_xbeta ./ (1-G_xbeta)) .* d["X"],1)
		return nothing
	end

	# function plotGrad()
	# 	d = makedata(10000)
	# end

	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=LBFGS())
		d = makedata()
		function g!(storage::Vector, betas::Vector)
			grad!(storage, betas, d)
		end
		res = optimize(arg->-loglik(arg,d), g!, x0,meth, Optim.Options(iterations = 500,g_tol=1e-20))
		return res
	 end

	# hessian of the likelihood at x
	function hessian!(storage::Matrix,betas::Vector,d)
		xbeta     = d["X"]*betas	# (n,1)
		G_xbeta   = cdf.(d["dist"],xbeta)	# (n,1)
		g_xbeta   = pdf.(d["dist"],xbeta)
		g_xbeta_sq   = pdf.(d["dist"],xbeta).^2
		G_xbeta_sq = cdf.(d["dist"],xbeta).^2
		function phideriv(x)
			- x / √(2*π) * exp(-x^2/2)
		end
		gprime_xbeta = 	phideriv.(xbeta) # (n,1)
		storage[:] = -sum(((d["y"] .* (gprime_xbeta .* G_xbeta - g_xbeta_sq) ./ G_xbeta_sq) - (1-d["y"]) .* (gprime_xbeta .* G_xbeta + g_xbeta_sq )./ (1-G_xbeta).^2)[i] * d["X"][i,:]) * transpose(d["X"][i,:]) for i in 1:d["n"])
		return nothing
	end

	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],meth=Newton())
		d = makedata()
		function g!(storage::Vector, betas::Vector)
			grad!(storage, betas, d)
		end
		function h!(storage::Matrix, betas::Vector)
			hessian!(storage, betas, d)
		end
		res = optimize(arg->-loglik(arg,d), g!, h!, x0,meth, Optim.Options(iterations = 500,g_tol=1e-20))
		return res
	end
	#

	#
	# function info_mat(betas::Vector,d)
	# end
	#
	# function inv_Info(betas::Vector,d)
	# end
	#
	#
	#
	# """
	# inverse of observed information matrix
	# """
	# function inv_observedInfo(betas::Vector,d)
	# end
	#
	# """
	# standard errors
	# """
	# function se(betas::Vector,d::Dict)
	# 	# sqrt.(diag(inv_observedInfo(betas,d)))
	# 	sqrt.(diag(inv_Info(betas,d)))
	# end

	# function that maximizes the log likelihood without the gradient
	# with a call to `optimize` and returns the result

	#

	# function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=BFGS())
	# end
	#

	# visual diagnostics
	# ------------------
	function runall()
		plotLike()
		#plotGrad()
		m1 = maximize_like()
		m2 = maximize_like_grad()
		m3 = maximize_like_grad_hess()
		#m4 = maximize_like_grad_se()
		println("results are:")
		println("maximize_like optimizer: $(m1.minimizer)")
		println("maximize_like iterations: $(Optim.iterations(m1))")
		println("maximize_like_grad: $(Optim.minimizer(m2))")
		println("maximize_like_grad iterations: $(Optim.iterations(m2))")
		println("maximize_like_grad_hess: $(m3.minimizer)")
		println("maximize_like_grad_hess iterations: $(Optim.iterations(m3))")
		# println("maximize_like_grad_se: $m4)")
		println("")
		println("running tests:")
		include("test/runtests_ML.jl")
		println("")
		if isinteractive()
			ok = input("enter y to close this session.")
			if ok == "y"
				quit()
			end
		end
	end
end
