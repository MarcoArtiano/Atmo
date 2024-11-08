# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Abstract base type for time integration schemes of explicit strong stability-preserving (SSP)
# Runge-Kutta (RK) methods. They are high-order time discretizations that guarantee the TVD property.
abstract type SimpleAlgorithmIMEX end

"""
    SimpleIMEX(; stage_callbacks=())

## References

- missing

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct SimpleIMEX{StageCallbacks} <: SimpleAlgorithmIMEX
    beta::Matrix{Float64}
    alfa::Matrix{Float64}
    gamma::Matrix{Float64}
    d::SVector{5, Float64}

    rkstages::Int64

    stage_callbacks::StageCallbacks
    FastMethod::String
    Mpw::Matrix{Float64}
    Mwp::Matrix{Float64}
    Mww::Matrix{Float64}
    equations_vertical::AbstractEquations
    function SimpleIMEX(; FastMethod = "RK4", Mwp, Mpw, Mww, equations_vertical, stage_callbacks = ())
        rkstages = 5
        beta = zeros(rkstages, rkstages)
        alfa = zeros(rkstages, rkstages)
        gamma = zeros(rkstages, rkstages)
        d = zeros(rkstages, 1)
        beta[2, 1] = 0.38758444641450318
        beta[3, 1] = -2.5318448354142823E-002
        beta[3, 2] = 0.38668943087310403
        beta[4, 1] = 0.20899983523553325
        beta[4, 2] = -0.45856648476371231
        beta[4, 3] = 0.43423187573425748
        beta[5, 1] = -0.10048822195663100
        beta[5, 2] = -0.46186171956333327
        beta[5, 3] = 0.83045062122462809
        beta[5, 4] = 0.27014914900250392

        alfa[3, 2] = 0.52349249922385610
        alfa[4, 2] = 1.1683374366893629
        alfa[4, 3] = -0.75762080241712637
        alfa[5, 2] = -3.6477233846797109E-002
        alfa[5, 3] = 0.56936148730740477
        alfa[5, 4] = 0.47746263002599681

        gamma[3, 2] = 0.13145089796226542
        gamma[4, 2] = -0.36855857648747881
        gamma[4, 3] = 0.33159232636600550
        gamma[5, 2] = -6.5767130537473045E-002
        gamma[5, 3] = 4.0591093109036858E-002
        gamma[5, 4] = 6.4902111640806712E-002

        d2 = beta[2, 1]
        d3 = beta[3, 1] + beta[3, 2]
        d4 = beta[4, 1] + beta[4, 2] + beta[4, 3]
        d5 = beta[5, 1] + beta[5, 2] + beta[5, 3] + beta[5, 4]
        d = SVector(0.0, d2, d3, d4, d5)
        new{typeof(stage_callbacks)}(beta, alfa, gamma, d, rkstages, stage_callbacks, FastMethod, Mwp, Mpw, Mww, equations_vertical)
    end
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct SimpleIntegratorIMEXOptions{Callback, TStops}
    callback::Callback # callbacks; used in Trixi
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    tstops::TStops # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function SimpleIntegratorIMEXOptions(callback, tspan; maxiters = typemax(Int),
                                     kwargs...)
    tstops_internal = BinaryHeap{eltype(tspan)}(FasterForward())
    # We add last(tspan) to make sure that the time integration stops at the end time
    push!(tstops_internal, last(tspan))
    # We add 2 * last(tspan) because add_tstop!(integrator, t) is only called by DiffEqCallbacks.jl if tstops contains a time that is larger than t
    # (https://github.com/SciML/DiffEqCallbacks.jl/blob/025dfe99029bd0f30a2e027582744528eb92cd24/src/iterative_and_periodic.jl#L92)
    push!(tstops_internal, 2 * last(tspan))
    SimpleIntegratorIMEXOptions{typeof(callback), typeof(tstops_internal)}(callback,
                                                                           false, Inf,
                                                                           maxiters,
                                                                           tstops_internal)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct SimpleIntegratorIMEX{RealT <: Real, uType, Params, Sol, F1, F2, Alg,
                                    SimpleIntegratorIMEXOptions}
    u::uType
    u_tmp::uType
    Zn0::uType
    dZn::uType
    du::uType
    r0::uType
    t::RealT
    tdir::RealT
    dt::RealT # current time step
    dtcache::RealT # manually set time step
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f1::F1
    f2::F2
    alg::Alg
    opts::SimpleIntegratorIMEXOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
end

"""
    add_tstop!(integrator::SimpleIntegratorSSP, t)
Add a time stop during the time integration process.
This function is called after the periodic SaveSolutionCallback to specify the next stop to save the solution.
"""
function add_tstop!(integrator::SimpleIntegratorIMEX, t)
    integrator.tdir * (t - integrator.t) < zero(integrator.t) &&
        error("Tried to add a tstop that is behind the current time. This is strictly forbidden")
    # We need to remove the first entry of tstops when a new entry is added.
    # Otherwise, the simulation gets stuck at the previous tstop and dt is adjusted to zero.
    if length(integrator.opts.tstops) > 1
        pop!(integrator.opts.tstops)
    end
    push!(integrator.opts.tstops, integrator.tdir * t)
end

has_tstop(integrator::SimpleIntegratorIMEX) = !isempty(integrator.opts.tstops)
first_tstop(integrator::SimpleIntegratorIMEX) = first(integrator.opts.tstops)

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::SimpleIntegratorIMEX, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

"""
    solve(ode, alg; dt, callbacks, kwargs...)

The following structures and methods provide the infrastructure for SSP Runge-Kutta methods
of type `SimpleAlgorithmSSP`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function solve(ode::ODEProblem, alg::T;
               dt, callback = nothing, kwargs...) where {T <: SimpleAlgorithmIMEX}
    u = copy(ode.u0)
    u_tmp = similar(u)
    Zn0 = similar(u)
    dZn = similar(u)
    du = similar(u)
    r0 = similar(u)
    t = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0
    integrator = SimpleIntegratorIMEX(u, u_tmp, Zn0, dZn, du, r0, t, tdir, dt, dt, iter,
                                      ode.p,
                                      (prob = ode,), ode.f.f1, ode.f.f2, alg,
                                      SimpleIntegratorIMEXOptions(callback, ode.tspan;
                                                                  kwargs...),
                                      false, true, false)

    # resize container
    resize!(integrator.p, nelements(integrator.p.solver1, integrator.p.cache1))

    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            error("unsupported")
        end
        foreach(callback.discrete_callbacks) do cb
            #   cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    elseif !isnothing(callback)
        error("unsupported")
    end

    for stage_callback in alg.stage_callbacks
        init_callback(stage_callback, integrator.p)
    end

    solve!(integrator)
end

function solve!(integrator::SimpleIntegratorIMEX)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback
    semi = integrator.p
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    integrator.finalstep = false
    @unpack equations1, equations2 = semi
    a = equations1.U
    b = equations2.cs
    dts = integrator.dt * a / b

    Mpw = sparse(alg.Mpw)
    Mwp = sparse(alg.Mwp)
    Mww = sparse(alg.Mww)
    Mpwwp = sparse(alg.Mpw*alg.Mwp*0.25f0)
    @trixi_timeit timer() "main loop" while !integrator.finalstep
        if isnan(integrator.dt)
            error("time step size `dt` is NaN")
        end

        modify_dt_for_tstops!(integrator)
        # if the next iteration would push the simulation beyond the end time, set dt accordingly
        if integrator.t + integrator.dt > t_end ||
           isapprox(integrator.t + integrator.dt, t_end)
            integrator.dt = t_end - integrator.t
            terminate!(integrator)
        end

        t_stage = integrator.t + integrator.dt

        Yn = [zeros(size(integrator.u)) for _ in 1:(alg.rkstages)]
        Sdu = [zeros(size(integrator.u)) for _ in 1:(alg.rkstages)]
        for stage in 1:(alg.rkstages)
            integrator.Zn0 .= integrator.u

            InitialConditionMIS!(integrator.Zn0, Yn, integrator.u, alg.alfa, stage)

            integrator.dZn .= 0

            PreparationODEMIS!(integrator.dZn, Yn, Sdu, integrator.u, alg.d, alg.gamma,
                               alg.beta, integrator.dt, stage)

            if stage == 1
                Yn[stage] .= integrator.u
            else
                if alg.FastMethod == "RK4"
                solveODEMIS_RK4!(Yn, integrator.dZn, integrator.Zn0, dts,
                                alg.d[stage] * integrator.dt, integrator, prob, stage,
                                semi)
                elseif alg.FastMethod == "SE"
                    solveODEMIS_SE!(Yn, integrator.dZn, integrator.Zn0, dts,
                                alg.d[stage] * integrator.dt, integrator, prob, stage,
                                semi)
                elseif alg.FastMethod == "MP"
                    solveODEMIS_MP!(Yn, integrator.dZn, integrator.Zn0, dts,
                                alg.d[stage] * integrator.dt, integrator, prob, stage,
                                semi, Mwp, Mpw, Mww, Mpwwp, alg.equations_vertical)                  
                end

            end

            integrator.u_tmp .= Yn[stage]
            integrator.f1(integrator.du, integrator.u_tmp, prob.p,
                          integrator.t + integrator.dt)
            Sdu[stage] .= integrator.du
        end

        integrator.u .= Yn[end]
        integrator.iter += 1
        println(integrator.iter)
        integrator.t += integrator.dt
        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    #              cb.affect!(integrator)
                end
            end
        end

        # respect maximum number of iterations
        if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
            @warn "Interrupted. Larger maxiters is needed."
            terminate!(integrator)
        end
    end

    # Empty the tstops array.
    # This cannot be done in terminate!(integrator::SimpleIntegratorSSP) because DiffEqCallbacks.PeriodicCallbackAffect would return at error.
    extract_all!(integrator.opts.tstops)

    for stage_callback in alg.stage_callbacks
        finalize_callback(stage_callback, integrator.p)
    end

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u), prob)
end

function debuggingprint(u, mesh, equations, solver, cache)
    u_wrap = wrap_array(u, mesh, equations, solver, cache)
    @show u_wrap[1, :, :]
    @show u_wrap[2, :, :]
    return nothing
end

function debuggingprintstage(u, mesh, equations, solver, cache, stage)
    u_stage = copy(u[stage])
    u_wrap = wrap_array(u_stage, mesh, equations, solver, cache)
    @show u_wrap[1, :, :]
    @show u_wrap[2, :, :]
    return nothing
end

function InitialConditionMIS!(Zn0, Yn, u, alfa, stage)
    for j in 1:(stage - 1)
        @threaded for i in eachindex(Zn0)
            Zn0[i] = Zn0[i] + alfa[stage, j] * (Yn[j][i] - u[i])
        end
    end
end

function PreparationODEMIS!(dZn, Yn, Sdu, u, d, gamma, beta, dtL, stage)
    for j in 1:(stage - 1)
        @threaded for i in eachindex(dZn)
            dZn[i] = dZn[i] +
                     1 / d[stage] * (1 / dtL * gamma[stage, j] * (Yn[j][i] - u[i]) +
                      beta[stage, j] * Sdu[j][i])
        end
    end
end

function solveODEMIS_RK4!(Yn, dZn, Zn0, dt, dtL, integrator, prob, stage, semi)
    yn = copy(Zn0)
    y = zeros(size(yn))
    du = [zeros(size(dZn)) for _ in 1:4]

    numit = round(dtL / dt)
    dtloc = dtL / numit
    A = zeros(4, 4)
    b = zeros(4, 1)
    A[2, 1] = 0.5
    A[3, 2] = 0.5
    A[4, 3] = 1
    b[1] = 1 / 6
    b[2] = 1 / 3
    b[3] = 1 / 3
    b[4] = 1 / 6

    for ii in 1:numit
        for s in 1:4
            y .= yn
            for i in 1:(s - 1)
                y .= y + dtloc * A[s, i] * du[i]
            end
            integrator.u_tmp .= y
            integrator.f2(integrator.du, integrator.u_tmp, prob.p,
                          integrator.t + integrator.dt)
            du[s] .= integrator.du + dZn
            if s == 4
                y .= yn
                for i in 1:4
                    y .= y + dtloc * b[i] * du[i]
                end
            end
        end
        yn .= y
    end
    Yn[stage] .= yn
end

function solveODEMIS_SE!(Yn, dZn, Zn0, dt, dtL, integrator, prob, stage, semi)
    numit = round(dtL / dt)
    dtloc = dtL / numit
    integrator.u_tmp .= Zn0

    for ii in 1:numit
        # Compute RHS u
        integrator.f2(integrator.du, integrator.u_tmp, prob.p,
                      integrator.t + integrator.dt)
        time_stepping_symplectic!(integrator.u_tmp, integrator.du, dtloc, semi, dZn, 1)
        integrator.f2(integrator.du, integrator.u_tmp, prob.p,
                      integrator.t + integrator.dt)
        time_stepping_symplectic!(integrator.u_tmp, integrator.du, dtloc, semi, dZn, 3)
    end
    Yn[stage] .= integrator.u_tmp
end

function solveODEMIS_MP!(Yn, dZn, Zn0, dt, dtL, integrator, prob, stage, semi, Mwp, Mpw, Mww, Mpwwp, equations_vertical)
    numit = round(dtL / dt)
    dtloc = dtL / numit
    integrator.u_tmp .= Zn0
    LA1 = LinearAlgebra.lu(sparse(I - dtloc^2*Mpwwp))
    for ii in 1:numit
        # Compute RHS u
        # We perform Euler, then we simply adjust p and w that are the "vertical variables"
        integrator.f2(integrator.du, integrator.u_tmp, prob.p,
                      integrator.t + integrator.dt) # compute rhs u^n

        time_stepping_midpoint!(integrator.u_tmp, integrator.du, dtloc, semi, dZn) # We just compute u^n+1 = rhs(u^n) for u and b
        
        time_stepping_p_w!(integrator.u_tmp, integrator.du, dtloc, semi, dZn, Mwp, Mpw, Mww, integrator, equations_vertical, LA1) # We adjust the p and w variables according to mid point 
    end

    Yn[stage] .= integrator.u_tmp
end

function time_stepping_symplectic!(u, du, dt, semi, dZn, var)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)

    u_wrap = Trixi.wrap_array(u, semi)
    du_wrap = Trixi.wrap_array(du, semi)
    dZn_wrap = Trixi.wrap_array(dZn, semi)
    perform_time_step_se!(u_wrap, du_wrap, dZn_wrap, var, dt, semi, solver, cache,
                          equations, mesh)
    return nothing
end

function time_stepping_midpoint!(u, du, dt, semi, dZn)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)

    u_wrap = Trixi.wrap_array(u, semi)
    du_wrap = Trixi.wrap_array(du, semi)
    dZn_wrap = Trixi.wrap_array(dZn, semi)
    perform_time_step_mp!(u_wrap, du_wrap, dZn_wrap, dt, semi, solver, cache,
                          equations, mesh)
    return nothing
end

function time_stepping_p_w!(u, du, dt, semi, dZn, Mwp, Mpw, Mww, integrator, equations_vertical, LA1)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
   @unpack source_terms2, boundary_conditions2, solver2, cache2, equations2, initial_condition = semi
    nvar = length(eachnode(solver))*length(eachnode(solver))*length(eachelement(solver,cache))
   
    # ## Checking if the two procedures are the same
    # semi = SemidiscretizationHyperbolic(mesh, equations2, initial_condition, solver2; boundary_conditions = boundary_conditions2)  
    # J = jacobian_ad_forward(semi)

    # du2  = copy(du)
    # du2 .= J*u
    # du3 = copy(du)
    # ## Posso chiamare rhs! tante volte quanto è la lunghezza di u e costruire la matrice J come prima colonna data da u 1 e tutti 0, seconda colonna secondo u 1 e tutti gli altri 0 etc.
    # u_wrap = Trixi.wrap_array(u, semi)
    # du_wrap = Trixi.wrap_array(du3, semi)
    # rhs!(du_wrap, u_wrap, integrator.t, mesh, equations2, boundary_conditions2, source_terms2, solver2, cache2)

    # println(maximum(du3-du2))
    # println(minimum(du3-du2))
    ## --- > They are the same the max(abs(min,max)) = 1e-16 roughly
    wn = view(u,2:4:4*nvar)    
    pn = view(u,3:4:4*nvar)
    dw = view(du,2:4:4*nvar)
    dp = view(du,3:4:4*nvar)
    Sw = view(dZn,2:4:4*nvar)
    Sp = view(dZn,3:4:4*nvar)
    
    pnew = LA1\(pn + dt*(dp + Sp) + dt^2*0.5f0*Mpw*(dw + Sw - Mwp*pn*0.5f0))
    duv = copy(du)
    uv = copy(u)
    pnv = view(uv,3:4:4*nvar)
    pnv = (pnv + pnew)*0.5
    uv_wrap = Trixi.wrap_array(uv, semi)
    duv_wrap = Trixi.wrap_array(duv, semi)

    rhs!(duv_wrap, uv_wrap, integrator.t, mesh, equations2, boundary_conditions2, source_terms2, solver2, cache2)
    dwv = view(duv,2:4:4*nvar)
    # println("Checking values")
    # println(maximum(dwv -(0.5*Mwp*(pnew+pn) + Mww*wn)))
    # println(minimum(dwv -(0.5*Mwp*(pnew+pn) + Mww*wn)))
    # println(maximum(dwv -(dw + 0.5*Mwp*(pnew-pn))))
    # println(minimum(dwv -(dw + 0.5*Mwp*(pnew-pn))))
    wn .= wn + dt*(Sw + dwv)
    #wn .= wn + dt*(dw + Sw + 0.5f0*Mwp*(pnew - pn))
    pn .= pnew

    #du2 == du?

    # println(pnew - pn)
    # println(maximum(pnew-pn))
    # println(minimum(pnew-pn))
    # LA2 = LinearAlgebra.lu(sparse(I - dt^2/4*Mwp*Mpw))
    # wnew = LA2\(wn + dt*(dw + Sw) + dt^2*0.5f0*Mwp*(dp - Mpw*wn*0.5f0 + Sp))
    # println(pnew - pn)
    # println(maximum(wnew-wn))
    # println(minimum(wnew-wn))
     
    # pn .= pn + dt*(dp + Sp + 0.5f0*Mpw*(wnew - wn))
    # wn .= wnew
    
    # @unpack prob = integrator.sol
    # tmp = TimeIntegratorSolution((first(prob.tspan), integrator.t),
    #                               (prob.u0, u), prob)
    # pd = PlotData2D(tmp)
    # a = Plots.plot(pd["w"],aspect_ratio = 10)
    # Plots.savefig(a,"debugging_w.pdf")  
    # a = Plots.plot(pd["p"],aspect_ratio = 10)
    # Plots.savefig(a,"debugging_p.pdf")       
           
    return nothing
end

function perform_time_step_se!(u, du, dZn, lvar, dt, semi, solver, cache, equations,
    mesh::Union{TreeMesh{1},P4estMesh{1}, StructuredMesh{1}})
    for element in eachelement(solver, cache)
        for i in eachnode(solver)
            for var in lvar:(lvar+1)
            u[var, i, element] = u[var, i, element] +
                                 dt * (du[var, i, element] + dZn[var, i, element])
            end
        end
    end

    return nothing
end

function perform_time_step_se!(u, du, dZn, lvar, dt, semi, solver, cache, equations,
    mesh::Union{TreeMesh{2}, P4estMesh{2}, StructuredMesh{2}})
for element in eachelement(solver, cache)
    for j in eachnode(solver), i in eachnode(solver)
        for var in lvar:(lvar+1)
u[var, i,j, element] = u[var, i, j, element] +
      dt * (du[var, i,j, element] + dZn[var, i,j, element])
        end
    end
end

return nothing
end

function perform_time_step_mp!(u, du, dZn, dt, semi, solver, cache, equations,
    mesh::Union{TreeMesh{2}, P4estMesh{2}, StructuredMesh{2}})
for element in eachelement(solver, cache)
    for j in eachnode(solver), i in eachnode(solver)
        for var in (1,4)
u[var, i,j, element] = u[var, i, j, element] +
      dt * (du[var, i,j, element] + dZn[var, i,j, element])
        end
    end
end

return nothing
end

# get a cache where the RHS can be stored
get_du(integrator::SimpleIntegratorIMEX) = integrator.du
#get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.Zn0,)
#get_tmp_cache(integrator::SimpleIntegratorIMEX) = (integrator.dZn,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SimpleIntegratorIMEX, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::SimpleIntegratorIMEX, dt)
    (integrator.dt = dt; integrator.dtcache = dt)
end

# used by adaptive timestepping algorithms in DiffEq
function get_proposed_dt(integrator::SimpleIntegratorIMEX)
    return ifelse(integrator.opts.adaptive, integrator.dt, integrator.dtcache)
end

# stop the time integration
function terminate!(integrator::SimpleIntegratorIMEX)
    integrator.finalstep = true
end

"""
    modify_dt_for_tstops!(integrator::SimpleIntegratorSSP)
Modify the time-step size to match the time stops specified in integrator.opts.tstops.
To avoid adding OrdinaryDiffEq to Trixi's dependencies, this routine is a copy of
https://github.com/SciML/OrdinaryDiffEq.jl/blob/d76335281c540ee5a6d1bd8bb634713e004f62ee/src/integrators/integrator_utils.jl#L38-L54
"""
function modify_dt_for_tstops!(integrator::SimpleIntegratorIMEX)
    if has_tstop(integrator)
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = first_tstop(integrator)
        if integrator.opts.adaptive
            integrator.dt = integrator.tdir *
                            min(abs(integrator.dt), abs(tdir_tstop - tdir_t)) # step! to the end
        elseif iszero(integrator.dtcache) && integrator.dtchangeable
            integrator.dt = integrator.tdir * abs(tdir_tstop - tdir_t)
        elseif integrator.dtchangeable && !integrator.force_stepfail
            # always try to step! with dtcache, but lower if a tstop
            # however, if force_stepfail then don't set to dtcache, and no tstop worry
            integrator.dt = integrator.tdir *
                            min(abs(integrator.dtcache), abs(tdir_tstop - tdir_t)) # step! to the end
        end
    end
end

# used for AMR

function Base.resize!(semi::SemidiscretizationHyperbolicSplit, new_size)
    resize!(semi, semi.solver1.volume_integral, new_size)
end

function Base.resize!(integrator::SimpleIntegratorIMEX, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.r0, new_size)

    # Resize container
    # new_size = n_variables * n_nodes^n_dims * n_elements
    n_elements = nelements(integrator.p.solver, integrator.p.cache)
    resize!(integrator.p, n_elements)
end

end # @muladd
