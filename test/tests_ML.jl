using Base.Test

@testset "basics" begin
	@testset "Test Data Construction" begin
		@test size(uncons.makedata()["X"]) == (10000, 3)
		@test size(uncons.makedata()["y"]) == (10000,)
		@test mean(uncons.makedata()["X"][:,1]) ≈ 0.0 atol = 0.1
		@test mean(uncons.makedata()["X"][:,2]) ≈ 0.0 atol = 0.1
		@test mean(uncons.makedata()["X"][:,3]) ≈ 0.0 atol = 0.1
		@test uncons.makedata()["n"] == 10000
	end
	@testset "Test Return value of likelihood" begin
		@test uncons.loglik(uncons.makedata()["beta"], uncons.makedata()) < 0
		@test uncons.loglik(uncons.makedata()["beta"], uncons.makedata()) > uncons.loglik([0,0,0], uncons.makedata())
		@test uncons.loglik(uncons.makedata()["beta"], uncons.makedata()) > uncons.loglik([1,1,1], uncons.makedata())
		@test uncons.loglik(uncons.makedata()["beta"], uncons.makedata()) > uncons.loglik([-0.5,-0.5,-0.5], uncons.makedata())
	end
# 	@testset "Test return value of gradient" begin
# 	end
end

@testset "test maximization results" begin
	@testset "maximize returns approximate result" begin
		@test uncons.maximize_like().minimizer[1] ≈ 1 atol = 0.1
		@test uncons.maximize_like().minimizer[2] ≈ 1.5 atol = 0.1
		@test uncons.maximize_like().minimizer[3] ≈ -0.5 atol = 0.1
	end
	@testset "maximize_grad returns accurate result" begin
		@test uncons.maximize_like_grad().minimizer[1] ≈ 1 atol = 0.1
		@test uncons.maximize_like_grad().minimizer[2] ≈ 1.5 atol = 0.1
		@test uncons.maximize_like_grad().minimizer[3] ≈ -0.5 atol = 0.1
	end
	@testset "maximize_grad_hess returns accurate result" begin
		minimizer = uncons.maximize_like_grad_hess().minimizer
		@test minimizer[1] ≈ 1 atol = 0.1
		@test minimizer[2] ≈ 1.5 atol = 0.1
		@test minimizer[3] ≈ -0.5 atol = 0.1
		hessian = Matrix{Float64}(3,3)
		uncons.hessian!(hessian, minimizer, uncons.makedata())
		eigva = eig(hessian)[1]
		@test eigva[1] > 0
		@test eigva[2] > 0
		@test eigva[3] > 0
	end
	@testset "gradient is close to zero at max like estimate" begin
		grad = Vector(3)
		uncons.grad!(grad, uncons.maximize_like_grad().minimizer, uncons.makedata())
		@test grad[1] ≈ 0 atol = 1.0e-5
		@test grad[2] ≈ 0 atol = 1.0e-5
		@test grad[3] ≈ 0 atol = 1.0e-5
	end
end


# @testset "test against GLM" begin
# 	@testset "estimates vs GLM" begin
# 	end
# 	@testset "standard errors vs GLM" begin
# 	end
# end
