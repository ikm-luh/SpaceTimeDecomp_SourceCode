using DrWatson
@quickactivate "SpaceTimeDecomp"

using Ferrite, SparseArrays
using LinearAlgebra
using Printf
using Tensors
using Tullio
using DataFrames
using CSV

include(srcdir("inp_reader.jl"))

struct MaterialParameters
    λ::Float64
    μ::Float64
    S::Matrix{Float64}
    E::Matrix{Float64}
    η::Float64
end

#Material State
mutable struct MaterialState
    xv::Array{Float64, 2}
end

function MaterialState(n_modes::Int)
    return MaterialState(
        0.1*ones(n_modes, 6)
    )
end

function compute_sigma(iter_modes::Int, ϵx::Array{Float64}, τ, τv, α, αv, βv, material::MaterialParameters, state::MaterialState)
    E = material.E
    S = material.S
    η = material.η

    if abs(αv[iter_modes]) < 1E-50 && abs(βv[iter_modes]) < 1E-50
        #do nothing, Tv is zero -> xv can be anything
        xv = state.xv[iter_modes, :]
    else
        #fixed from past modes
        rhs = zeros(6)
        for i = 1:iter_modes-1
            rhs += 1/η * S * E * ϵx[i, :] * α[i] - state.xv[i, :] * βv[i] - 1/η * S * E * state.xv[i, :] * αv[i]
        end
        lhs = (I*βv[iter_modes] + 1/η * S * E * αv[iter_modes])
        xv = (lhs \ (1/η * S * E * ϵx[iter_modes, :] * α[iter_modes] + rhs))
    end

    sigma = E * (ϵx[iter_modes, :] * τ[iter_modes] - xv * τv[iter_modes])    
    #add past modes contribution
    for i = 1:iter_modes-1
        sigma += E * (ϵx[i, :] * τ[i] - state.xv[i, :] * τv[i])
    end   

    state.xv[iter_modes, :] = xv
    return sigma
end

function compute_sigma_tangent(iter_modes::Int, τ, τv, α, αv, βv, material::MaterialParameters)
    E = material.E    
    S = material.S
    η = material.η
    
    if abs(αv[iter_modes]) < 1E-50 && abs(βv[iter_modes]) < 1E-50
        return E * τ[iter_modes]    
    else
        return E * τ[iter_modes] - E * ((I*βv[iter_modes] + 1/η * S * E * αv[iter_modes]) \ (1/η * S * E * α[iter_modes]) * τv[iter_modes])
    end
end

function symmetrize_lower!(K)
    for i in 1:size(K,1)
        for j in i+1:size(K,1)
            K[i,j] = K[j,i]
        end
    end
end

function setup(λ, μ, η)
    S = [2/3 -1/3 -1/3 0 0 0;
    -1/3 2/3 -1/3 0 0 0;
    -1/3 -1/3 2/3 0 0 0;
    0 0 0 1 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;]

    E = [λ+2*μ λ λ  0 0 0;
        λ λ+2*μ λ 0 0 0;
        λ λ λ+2*μ 0 0 0;
        0 0 0 μ 0 0; 
        0 0 0 0 μ 0;
        0 0 0 0 0 μ;]
    
    material = MaterialParameters(λ, μ, S, E, η)

    #Main Computation
    grid, cell_type = mesh_from_inp(datadir("meshes/Spring/", "SpringFine.inp"))  
    addfaceset!(grid, "left", (x) -> x[3] < 0.0001)
    addfaceset!(grid, "right", (x) -> x[3] > 0.0599)

    dim = 3
    ip = Lagrange{dim, RefTetrahedron, 2}()
    qr = QuadratureRule{dim, RefTetrahedron}(2)
    cellvalues = CellVectorValues(qr, ip)
    
    #DoF
    dh = DofHandler(grid)
    push!(dh, :u, dim, ip)  #change
    close!(dh); 
    
    #Sparsity Pattern
    K = create_sparsity_pattern(dh);
    
    #Constraints
    ch_zero = ConstraintHandler(dh);
    ∂Ωr = union(getfaceset.((grid, ), ["right"])...);
    dbc_r = Ferrite.Dirichlet(:u, ∂Ωr, (x, t) -> [0.0], [3]);
    ∂Ωl = union(getfaceset.((grid, ), ["left"])...);
    dbc_l = Ferrite.Dirichlet(:u, ∂Ωl, (x, t) -> [0.0 0.0 0.0], [1, 2, 3]);


    add!(ch_zero, dbc_r);
    add!(ch_zero, dbc_l); 
    close!(ch_zero);

    ch = ConstraintHandler(dh);
    ∂Ωr = union(getfaceset.((grid, ), ["right"])...);
    dbc_r = Ferrite.Dirichlet(:u, ∂Ωr, (x, t) -> [1.0], [3]);
    ∂Ωl = union(getfaceset.((grid, ), ["left"])...);
    dbc_l = Ferrite.Dirichlet(:u, ∂Ωl, (x, t) -> [0.0 0.0 0.0], [1, 2, 3]);

    add!(ch, dbc_l);
    add!(ch, dbc_r); 
    close!(ch);

    #Boundary with nonzero boundary condition -> moving, since ∫_Ω F dV = 0 -> only reaction force or the external force
    ch_nzBc = ConstraintHandler(dh);
    add!(ch_nzBc, dbc_r); 
    close!(ch_nzBc);

    return material, grid, cellvalues, dh, K, ch, ch_zero, ch_nzBc
end

function doassemble_KR(iter_modes::Int, cellvalues::CellVectorValues{dim}, K::SparseMatrixCSC, x, dh::DofHandler, τ, τv, α, αv, βv, states, material) where {dim}
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    re = zeros(n_basefuncs)

    r = zeros(ndofs(dh))
    assembler = start_assemble(K, r)

    F_all = zeros(ndofs(dh))

    @inbounds for (cell, state) in zip(CellIterator(dh), states)
        fill!(Ke, 0)
        fill!(re, 0)
        reinit!(cellvalues, cell)

        eldofs = celldofs(cell)
        for q_point in 1:getnquadpoints(cellvalues)     #For each integration point
            dΩ = getdetJdV(cellvalues, q_point)
              
            ϵx = zeros(iter_modes, 6)
            for i = 1:iter_modes
                xe = x[i, eldofs]
                ϵx[i, :] = tovoigt(function_symmetric_gradient(cellvalues, q_point, xe))   #for all eps[iter_modes]
            end
            
            σ = compute_sigma(iter_modes, ϵx, τ, τv, α, αv, βv, material, state[q_point])
            dσ = compute_sigma_tangent(iter_modes, τ, τv, α, αv, βv, material)


            for i in 1:n_basefuncs                      #For each u
                N = shape_value(cellvalues, q_point, i)        
                B = tovoigt(shape_symmetric_gradient(cellvalues, q_point, i))
                re[i] += transpose(B) * σ * dΩ 

                for j in 1:i
                    Bj = tovoigt(shape_symmetric_gradient(cellvalues, q_point, j))
                    Ke[i, j] += transpose(B) * dσ * Bj * dΩ       
                end
            end
        end
        symmetrize_lower!(Ke)
        assemble!(assembler, celldofs(cell), re, Ke)
    end

    return K, r
end

function compute_x_funcs(iter_modes::Int, cellvalues::CellVectorValues{dim}, x::Array{Float64, 2}, dh::DofHandler, states, material) where {dim}
    n_basefuncs = getnbasefunctions(cellvalues)
    
    Ψ = zeros(iter_modes)
    Ψv = zeros(iter_modes)
    av = zeros(iter_modes)
    b = zeros(iter_modes)
    bv = zeros(iter_modes)


    @inbounds for (cell, state) in zip(CellIterator(dh), states)
        reinit!(cellvalues, cell)

        eldofs = celldofs(cell)
        
        for q_point in 1:getnquadpoints(cellvalues)     #For each integration point
            dΩ = getdetJdV(cellvalues, q_point)
            
            ϵx = zeros(iter_modes, 6)
            for i = 1:iter_modes
                xe = x[i, eldofs]
                ϵx[i, :] = tovoigt(function_symmetric_gradient(cellvalues, q_point, xe))   
            end

            for i = 1:iter_modes
                Ψ[i] += ϵx[iter_modes, :]' * material.E * ϵx[i, :] * dΩ
                Ψv[i] += ϵx[iter_modes, :]' * material.E * state[q_point].xv[i, :] * dΩ
                av[i] += state[q_point].xv[iter_modes, :]' * state[q_point].xv[i, :] * dΩ
                b[i] += 1/material.η * state[q_point].xv[iter_modes, :]' * material.S * material.E * ϵx[i, :] * dΩ
                bv[i] += 1/material.η * state[q_point].xv[iter_modes, :]' * material.S * material.E * state[q_point].xv[i, :] * dΩ
            end
        end
    end

    return Ψ, Ψv, av, b, bv	
end

function compute_stress(n_modes::Int, n_steps::Int, cellvalues::CellVectorValues{dim}, X::Array{Float64, 2}, T::Array{Float64, 2}, Tv::Array{Float64, 2}, states, material) where {dim}
    σ_data = zeros(n_steps, getncells(grid), 6)
    ϵv_data = zeros(n_steps, getncells(grid), 6)
    ϵ_data = zeros(n_steps, getncells(grid), 6)

    for iter_step = 1:n_steps
        iter_cell = 0
        @inbounds for (cell, state) in zip(CellIterator(dh), states)
            iter_cell += 1
            reinit!(cellvalues, cell)

            eldofs = celldofs(cell)
            
            for q_point in 1:getnquadpoints(cellvalues)     #For each integration point
                dΩ = getdetJdV(cellvalues, q_point)
                
                ϵ = zeros(6)
                for i = 1:n_modes
                    xe = X[i, eldofs]
                    ϵ = tovoigt(function_symmetric_gradient(cellvalues, q_point, xe)) * T[i, iter_step]   
                    σ_data[iter_step, iter_cell, :] += 1/getnquadpoints(cellvalues) * material.E * (ϵ - state[q_point].xv[i, :] * Tv[i, iter_step])
                    ϵv_data[iter_step, iter_cell, :] += 1/getnquadpoints(cellvalues) * state[q_point].xv[i, :] * Tv[i, iter_step]
                    ϵ_data[iter_step, iter_cell, :] += 1/getnquadpoints(cellvalues) * ϵ
                end
            end
        end
    end    

    return σ_data, ϵ_data, ϵv_data
end

function compute_force(n_steps::Int, dh::DofHandler, ch_nzBc, σ_data)
    F_sum = zeros(n_steps, 3)
    n_basefuncs = getnbasefunctions(cellvalues)

    for iter_step = 1:n_steps
        iter_cell = 0
        F_all = zeros(ndofs(dh))    
        @inbounds for cell in CellIterator(dh)
            iter_cell += 1
            reinit!(cellvalues, cell)
            eldofs= celldofs(cell)

            Fe_all = zeros(length(eldofs))
            for q_point in 1:getnquadpoints(cellvalues)
                dΩ = getdetJdV(cellvalues, q_point)
                for i in 1:n_basefuncs
                    B = tovoigt(shape_symmetric_gradient(cellvalues, q_point, i))
                    Fe_all[i] += transpose(B) * σ_data[iter_step, iter_cell, :] * dΩ
                end
            end
            F_all[eldofs] += Fe_all 
        end

        F_all[Ferrite.free_dofs(ch_nzBc)] .= 0
        F_sum[iter_step, :] = [sum(F_all[1:3:end]), sum(F_all[2:3:end]), sum(F_all[3:3:end])]
    end

    return F_sum
end

#calculate finite differences of vector
function diff_vec(vec::Vector{Float64}, Δt::Float64)
    res = zeros(length(vec))

    res[1] = 1/Δt * (vec[2] - vec[1])       #forward differences
    res[end] = 1/Δt *(vec[end]-vec[end-1])  #backward differences

    for i = 2:length(vec)-1
        res[i] = 1/(2*Δt) * (vec[i+1]-vec[i-1]) #midpoint differences
    end
    return res
end

#integration of vector based on simpsons rule
function integrate_vec(vec::Vector{Float64}, Δt::Float64)
    res = 0

    res = sum(vec)

    return res
end

function compute_residuum_error(dh, cellvalues, ch, X, T, Tv, states, material, iter_modes)
    error = 0.0

    for iter_step = 1:n_steps
        r_step = zeros(ndofs(dh))

        n_basefuncs = getnbasefunctions(cellvalues)
        re = zeros(n_basefuncs)


        @inbounds for (cell, state) in zip(CellIterator(dh), states)
            fill!(re, 0)
            reinit!(cellvalues, cell)

            eldofs = celldofs(cell)

            for q_point in 1:getnquadpoints(cellvalues)    
                dΩ = getdetJdV(cellvalues, q_point)

                σ = zeros(6)
                for i = 1:n_modes
                    xe = X[i, eldofs]
                    ϵ = tovoigt(function_symmetric_gradient(cellvalues, q_point, xe)) * T[i, iter_step]   
                    σ += 1/getnquadpoints(cellvalues) * material.E * (ϵ - state[q_point].xv[i, :] * Tv[i, iter_step])
                end

                for i in 1:n_basefuncs                      #For each u
                    N = shape_value(cellvalues, q_point, i)        
                    B = tovoigt(shape_symmetric_gradient(cellvalues, q_point, i))
                    re[i] += transpose(B) * σ * dΩ 
                end
            end
            r_step[eldofs] += re
        end
        apply_zero!(r_step, ch)
        error += norm(r_step)
    end

    return error
end

###################################################################################################################################
n_modes = 12
n_fix_point_iter = 5
n_steps = 101
Δt = 1.0/(n_steps-1)
λ = 1000E6
μ = 800E6
η = 500E6

material, grid, cellvalues, dh, K, ch, ch_zero, ch_nzBc = setup(λ, μ, η)

nqp = getnquadpoints(cellvalues)
states = [[MaterialState(n_modes) for _ in 1:nqp] for _ in 1:getncells(grid)]

X = zeros(n_modes, ndofs(dh))
T = ones(n_modes, n_steps)  #reasonable initial condition
Tv = zeros(n_modes, n_steps)
u = zeros(n_steps, ndofs(dh))
u_last = zeros(n_steps, ndofs(dh))  #save last state
error_ind = zeros(n_modes)

#scalars 
τ = zeros(n_modes)
τv = zeros(n_modes)
α = zeros(n_modes)
αv = zeros(n_modes)
βv = zeros(n_modes)
Ψ = zeros(n_modes)

#Non-homogeneous dirichlet boundary conditions in first mode
iter_modes = 1
X[1, :] = apply!(X[1, :], ch) 
for iter_step = 1:n_steps
    T[1, iter_step] = 0.1
end
T[:, 1] .= 0.0


##################################
##iterations
println("#################################### Iterations ################################################")
total_iterations = 0

for iter_modes = 2:n_modes
    println("Mode: ", iter_modes)

    #viscous
    for fix_point_iter = 1:n_fix_point_iter
        ##compute x, t and update xv, tv     
        #compute x
        for i = 1:iter_modes
            τ[i] = integrate_vec(T[iter_modes, :].*T[i, :], Δt) 
            τv[i] = integrate_vec(T[iter_modes, :].*Tv[i, :], Δt)
            α[i] = integrate_vec(Tv[iter_modes, :].*T[i, :], Δt)
            αv[i] = integrate_vec(Tv[iter_modes, :].*Tv[i, :], Δt)
            βv[i] = integrate_vec(Tv[iter_modes, :].*diff_vec(Tv[i, :], Δt), Δt)
        end
        

        for iter_newton = 1:2   #convergence in two steps
            global K, R = doassemble_KR(iter_modes, cellvalues, K, X, dh, τ, τv, α, αv, βv, states, material)
            apply_zero!(K, R, ch_zero)
            ΔX = K\R
            apply_zero!(ΔX, ch_zero)
            global X[iter_modes, :] = X[iter_modes, :] - ΔX
        end
   
        #update t
        Ψ, Ψv, av, b, bv = compute_x_funcs(iter_modes, cellvalues, X, dh, states, material)

        lhs = Ψ[iter_modes] - Ψv[iter_modes]*(av[iter_modes] + Δt * bv[iter_modes])^(-1) * Δt * b[iter_modes]
        rhs_factor = Ψv[iter_modes] * (av[iter_modes] + Δt * bv[iter_modes])^(-1) * av[iter_modes]
        

        for ts = 2:n_steps
            fixed_part = 0
            for i = 1:iter_modes-1
                fixed_part += Ψv[iter_modes]*(av[iter_modes] + Δt * bv[iter_modes])^(-1)*Δt*b[i]*T[i, ts] 
                fixed_part -= Ψv[iter_modes]*(av[iter_modes] + Δt * bv[iter_modes])^(-1)*(av[i]*(Tv[i, ts] - Tv[i, ts-1]) + Δt*bv[i]*T[i, ts])
                fixed_part -= (Ψ[i]*T[i, ts] - Ψv[i]*Tv[i, ts])
            end
            global T[iter_modes, ts] = lhs \ (rhs_factor*Tv[iter_modes, ts-1] + fixed_part)      #T is fixed as first mode is boundary condition
        end
            
        #if T is zero everywhere, it is set arbitrarily such that X will be zero. Letting T equal zero breaks stuff
        if sum(T[iter_modes, :]) == 0.0
            T[iter_modes, :] .= 1.0
        end

        #normalize
        sum_T_mode = sum(T[iter_modes, :])
        T[iter_modes, :] = T[iter_modes, :] ./ sum_T_mode
    

        for ts = 2:n_steps
            fixed_part_2 = 0
            for i = 1:iter_modes-1
                fixed_part_2 += Δt*b[i]*T[i, ts] - (av[i]*(Tv[i, ts] - Tv[i, ts-1]) + Δt*bv[i]*Tv[i, ts])
            end
            global Tv[iter_modes, ts] = (av[iter_modes] + Δt * bv[iter_modes])^(-1) * (av[iter_modes] * Tv[iter_modes, ts-1] + Δt * b[iter_modes]*T[iter_modes, ts] + fixed_part_2)    
        end

        sum_Tv_mode = sum(Tv[iter_modes, :])
        Tv[iter_modes, :] = Tv[iter_modes, :] ./ sum_Tv_mode
     
        global total_iterations += 1
        
        @tullio u_cur[j, i] := X[iter_modes, i] * T[iter_modes, j]
        @tullio u_1[j, i] := X[2, i] * T[2, j]

        rel_error = norm(u_cur - u_last)/norm(u_last)
        error_ind[iter_modes] = compute_residuum_error(dh, cellvalues, ch, X, T, Tv, states, material, iter_modes) 

        println("_________________________________________________________________________")
        println("Iteration: ", fix_point_iter)
        println("Rel_error: ", rel_error)
        println("Residuum: ", error_ind[iter_modes])
        println("_________________________________________________________________________")

        global u_last = u_cur

        if fix_point_iter > 2 && rel_error < 1E-4
            break
        end
    end
end


 
for iter = 1:n_modes
    println(iter)
    @tullio u[j, i] += X[$iter, i] * T[$iter, j]
end

@tullio u1[j, i] := X[1, i] * T[1, j]
@tullio u2[j, i] := X[2, i] * T[2, j]
@tullio u3[j, i] := X[3, i] * T[3, j]
@tullio u4[j, i] := X[4, i] * T[4, j]
@tullio u5[j, i] := X[5, i] * T[5, j]

σ, ϵ, ϵv = compute_stress(n_modes, n_steps, cellvalues, X, T, Tv, states, material)
F = compute_force(n_steps, dh, ch_nzBc, σ)

println("Total iterations: ", total_iterations)

folderName = "sims/Spring/"*string(η)*"_"*string(n_modes)*"_"*string(total_iterations)
mkdir(datadir(folderName, ))


SaveDict = Dict(
    "n_modes" => n_modes,
    "n_fix_point_iter" => n_fix_point_iter,
    "Total_iterations" => total_iterations,
    "n_steps" => n_steps,
    "Δt" => Δt,
    "λ" => λ,
    "μ" => μ,
    "η" => η
)
CSV.write(datadir(folderName, "Info.csv"), DataFrame(SaveDict))


pvd = paraview_collection(datadir(folderName, "STD_ve.pvd"))
for t in 1:n_steps
    vtk_grid(datadir(folderName, "STD_res-$t"), dh) do vtk
        vtk_point_data(vtk, dh, u[t, :], "Displacement")
        vtk_point_data(vtk, dh, u1[t, :], "1st mode")
        vtk_point_data(vtk, dh, u2[t, :], "2nd mode")
        vtk_point_data(vtk, dh, u3[t, :], "3rd mode")
        vtk_point_data(vtk, dh, u4[t, :], "4th mode")
        vtk_point_data(vtk, dh, u5[t, :], "5th mode")
        vtk_cell_data(vtk, transpose(σ[t, :, :]), "Stress")
        vtk_cell_data(vtk, transpose(ϵ[t, :, :]), "Strain")
        vtk_cell_data(vtk, transpose(ϵv[t, :, :]), "Viscous Strain")
        vtk_save(vtk)
        pvd[t] = vtk
    end
end

save(datadir(folderName, "Sigma_STD.jld2"), "Stress", σ)
save(datadir(folderName, "Disp_STD.jld2"), "Displacement", u)

df = DataFrame(F, :auto)
CSV.write(datadir(folderName, "Force_STD.csv"), df)