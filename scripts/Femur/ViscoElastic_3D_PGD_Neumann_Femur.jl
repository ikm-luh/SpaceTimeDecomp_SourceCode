using DrWatson
@quickactivate "SpaceTimeDecomp_SourceCode"

using Ferrite 
using SparseArrays
using LinearAlgebra
using Printf
using Tensors
using Tullio
using DataFrames
using CSV
using FerriteMeshParser

# Define material parameters for the viscoelastic model
struct MaterialParameters
    λ::Float64  # Lamé's first parameter
    μ::Float64  # Shear modulus
    S::Matrix{Float64}  # Deviator matrix
    E::Matrix{Float64}  # Elasticity matrix
    η::Float64  # Viscosity coefficient
end

# Define the material state, which stores internal variables
mutable struct MaterialState
    xv::Array{Float64, 2}  # Internal state variable for each mode
end

# Constructor for initializing the material state
function MaterialState(n_modes::Int)
    return MaterialState(
        0.1 * ones(n_modes, 6)  # Initialize with small values
    )
end

# Compute stress (σ) based on strain (ϵx) and material properties
function compute_sigma(iter_modes::Int, ϵx::Array{Float64}, τ, τv, α, αv, βv, material::MaterialParameters, state::MaterialState)
    E = material.E  # Elasticity matrix
    S = material.S  # Deviator matrix
    η = material.η  # Viscosity coefficient

    if abs(αv[iter_modes]) < 1E-50 && abs(βv[iter_modes]) < 1E-50
        # If αv and βv are negligible, no update is needed
        xv = state.xv[iter_modes, :]
    else
        # Compute the right-hand side (rhs) for the current mode
        rhs = zeros(6)
        for i = 1:iter_modes-1
            rhs += 1/η * S * E * ϵx[i, :] * α[i] - state.xv[i, :] * βv[i] - 1/η * S * E * state.xv[i, :] * αv[i]
        end
        # Compute the left-hand side (lhs) and solve for xv
        lhs = (I * βv[iter_modes] + 1/η * S * E * αv[iter_modes])
        xv = (lhs \ (1/η * S * E * ϵx[iter_modes, :] * α[iter_modes] + rhs))
    end

    # Compute stress for the current mode
    sigma = E * (ϵx[iter_modes, :] * τ[iter_modes] - xv * τv[iter_modes])
    # Add contributions from previous modes
    for i = 1:iter_modes-1
        sigma += E * (ϵx[i, :] * τ[i] - state.xv[i, :] * τv[i])
    end

    # Update the internal state variable
    state.xv[iter_modes, :] = xv
    return sigma
end

# Compute the tangent stiffness matrix for the stress-strain relationship
function compute_sigma_tangent(iter_modes::Int, τ, τv, α, αv, βv, material::MaterialParameters)
    E = material.E  # Elasticity matrix
    S = material.S  # Deviator matrix
    η = material.η  # Viscosity coefficient

    if abs(αv[iter_modes]) < 1E-50 && abs(βv[iter_modes]) < 1E-50
        # If αv and βv are negligible, return a simplified tangent
        return E * τ[iter_modes]
    else
        # Compute the tangent stiffness matrix
        return E * τ[iter_modes] - E * ((I * βv[iter_modes] + 1/η * S * E * αv[iter_modes]) \ (1/η * S * E * α[iter_modes]) * τv[iter_modes])
    end
end

# Symmetrize the lower triangular part of a matrix
function symmetrize_lower!(K)
    for i in 1:size(K, 1)
        for j in i+1:size(K, 1)
            K[i, j] = K[j, i]  # Copy upper triangular values to the lower triangular part
        end
    end
end

# Setup function to initialize material properties, grid, and other components
function setup(λ, μ, η)
    # Define the compliance matrix (S) and elasticity matrix (E)
    S = [2/3 -1/3 -1/3 0 0 0;
         -1/3 2/3 -1/3 0 0 0;
         -1/3 -1/3 2/3 0 0 0;
         0 0 0 1 0 0;
         0 0 0 0 1 0;
         0 0 0 0 0 1]
    E = [λ+2*μ λ λ  0 0 0;
         λ λ+2*μ λ 0 0 0;
         λ λ λ+2*μ 0 0 0;
         0 0 0 μ 0 0;
         0 0 0 0 μ 0;
         0 0 0 0 0 μ]

    # Initialize material parameters
    material = MaterialParameters(λ, μ, S, E, η)

    # Load the finite element grid from an input file
    grid = get_ferrite_grid(datadir("meshes/Femur/Job-1.inp"))
    addfaceset!(grid, "bottom", (x) -> x[1] > 156.24)  # Define boundary conditions

    # Define interpolation and quadrature rules
    dim = 3
    ip = Lagrange{dim, RefTetrahedron, 2}()  # Quadratic interpolation
    qr = QuadratureRule{dim, RefTetrahedron}(2)  # Quadrature rule for volume integration
    cellvalues = CellVectorValues(qr, ip)

    qr_face = QuadratureRule{dim-1, RefTetrahedron}(2)  # Quadrature rule for surface integration
    facevalues = FaceVectorValues(qr_face, ip)

    # Define degrees of freedom (DoF) handler
    dh = DofHandler(grid)
    push!(dh, :u, dim, ip)  # Add displacement DoFs
    close!(dh)

    # Create sparsity pattern for the stiffness matrix
    K = create_sparsity_pattern(dh)

    # Define constraints for boundary conditions
    ch_zero = ConstraintHandler(dh)
    ∂Ωb = union(getfaceset.((grid,), ["bottom"])...)
    dbc_b = Ferrite.Dirichlet(:u, ∂Ωb, (x, t) -> [0.0 0.0 0.0], [1, 2, 3])
    add!(ch_zero, dbc_b)
    close!(ch_zero)

    ch = ConstraintHandler(dh)
    add!(ch, dbc_b)
    close!(ch)

    # Define nonzero boundary conditions for moving boundaries
    ch_nzBc = ConstraintHandler(dh)
    add!(ch_nzBc, dbc_b)
    close!(ch_nzBc)

    return material, grid, cellvalues, facevalues, dh, K, ch, ch_zero, ch_nzBc
end

# Assemble the stiffness matrix (K) and residual vector (r) for the current mode
function doassemble_KR(iter_modes::Int, cellvalues::CellVectorValues{dim}, K::SparseMatrixCSC, x, dh::DofHandler, τ, τv, α, αv, βv, states, material) where {dim}
    n_basefuncs = getnbasefunctions(cellvalues)  # Number of basis functions per element
    Ke = zeros(n_basefuncs, n_basefuncs)  # Element stiffness matrix
    re = zeros(n_basefuncs)  # Element residual vector

    r = zeros(ndofs(dh))  # Global residual vector
    assembler = start_assemble(K, r)  # Initialize assembler

    F_all = zeros(ndofs(dh))  # Placeholder for external forces

    @inbounds for (cell, state) in zip(CellIterator(dh), states)
        fill!(Ke, 0)  # Reset element stiffness matrix
        fill!(re, 0)  # Reset element residual vector
        reinit!(cellvalues, cell)  # Reinitialize cell values for the current cell

        eldofs = celldofs(cell)  # Get degrees of freedom for the current cell

        for q_point in 1:getnquadpoints(cellvalues)  # Loop over quadrature points
            dΩ = getdetJdV(cellvalues, q_point)  # Compute the Jacobian determinant

            ϵx = zeros(iter_modes, 6)  # Strain tensor for all modes
            for i = 1:iter_modes
                xe = x[i, eldofs]  # Extract displacement for the current mode
                ϵx[i, :] = tovoigt(function_symmetric_gradient(cellvalues, q_point, xe))  # Compute strain
            end

            σ = compute_sigma(iter_modes, ϵx, τ, τv, α, αv, βv, material, state[q_point])  # Compute stress
            dσ = compute_sigma_tangent(iter_modes, τ, τv, α, αv, βv, material)  # Compute tangent stiffness

            for i = 1:n_basefuncs  # Loop over basis functions
                B = tovoigt(shape_symmetric_gradient(cellvalues, q_point, i))  # Compute strain-displacement matrix
                re[i] += transpose(B) * σ * dΩ  # Assemble residual vector

                for j in 1:i  # Loop over basis functions for stiffness matrix
                    Bj = tovoigt(shape_symmetric_gradient(cellvalues, q_point, j))
                    Ke[i, j] += transpose(B) * dσ * Bj * dΩ  # Assemble stiffness matrix
                end
            end
        end
        symmetrize_lower!(Ke)  # Symmetrize the stiffness matrix
        assemble!(assembler, celldofs(cell), re, Ke)  # Assemble into global system
    end

    return K, r
end

# Compute the functions Ψ, Ψv, av, b, and bv for the current mode
function compute_x_funcs(iter_modes::Int, cellvalues::CellVectorValues{dim}, x::Array{Float64, 2}, dh::DofHandler, states, material) where {dim}
    n_basefuncs = getnbasefunctions(cellvalues)  # Number of basis functions per element
    
    Ψ = zeros(iter_modes)  # Initialize Ψ
    Ψv = zeros(iter_modes)  # Initialize Ψv
    av = zeros(iter_modes)  # Initialize av
    b = zeros(iter_modes)  # Initialize b
    bv = zeros(iter_modes)  # Initialize bv

    @inbounds for (cell, state) in zip(CellIterator(dh), states)
        reinit!(cellvalues, cell)  # Reinitialize cell values for the current cell

        eldofs = celldofs(cell)  # Get degrees of freedom for the current cell
        
        for q_point in 1:getnquadpoints(cellvalues)  # Loop over quadrature points
            dΩ = getdetJdV(cellvalues, q_point)  # Compute the Jacobian determinant
            
            ϵx = zeros(iter_modes, 6)  # Strain tensor for all modes
            for i = 1:iter_modes
                xe = x[i, eldofs]  # Extract displacement for the current mode
                ϵx[i, :] = tovoigt(function_symmetric_gradient(cellvalues, q_point, xe))  # Compute strain
            end

            for i = 1:iter_modes
                Ψ[i] += ϵx[iter_modes, :]' * material.E * ϵx[i, :] * dΩ  # Compute Ψ
                Ψv[i] += ϵx[iter_modes, :]' * material.E * state[q_point].xv[i, :] * dΩ  # Compute Ψv
                av[i] += state[q_point].xv[iter_modes, :]' * state[q_point].xv[i, :] * dΩ  # Compute av
                b[i] += 1/material.η * state[q_point].xv[iter_modes, :]' * material.S * material.E * ϵx[i, :] * dΩ  # Compute b
                bv[i] += 1/material.η * state[q_point].xv[iter_modes, :]' * material.S * material.E * state[q_point].xv[i, :] * dΩ  # Compute bv
            end
        end
    end

    return Ψ, Ψv, av, b, bv	
end

# Compute stress, strain, and viscous strain for all time steps
function compute_stress(n_modes::Int, n_steps::Int, cellvalues::CellVectorValues{dim}, X::Array{Float64, 2}, T::Array{Float64, 2}, Tv::Array{Float64, 2}, states, material) where {dim}
    σ_data = zeros(n_steps, getncells(grid), 6)  # Initialize stress data
    ϵv_data = zeros(n_steps, getncells(grid), 6)  # Initialize viscous strain data
    ϵ_data = zeros(n_steps, getncells(grid), 6)  # Initialize strain data

    for iter_step = 1:n_steps
        iter_cell = 0
        @inbounds for (cell, state) in zip(CellIterator(dh), states)
            iter_cell += 1
            reinit!(cellvalues, cell)  # Reinitialize cell values for the current cell

            eldofs = celldofs(cell)  # Get degrees of freedom for the current cell
            
            for q_point in 1:getnquadpoints(cellvalues)  # Loop over quadrature points
                dΩ = getdetJdV(cellvalues, q_point)  # Compute the Jacobian determinant
                
                ϵ = zeros(6)  # Initialize strain tensor
                for i = 1:n_modes
                    xe = X[i, eldofs]  # Extract displacement for the current mode
                    ϵ = tovoigt(function_symmetric_gradient(cellvalues, q_point, xe)) * T[i, iter_step]  # Compute strain
                    σ_data[iter_step, iter_cell, :] += 1/getnquadpoints(cellvalues) * material.E * (ϵ - state[q_point].xv[i, :] * Tv[i, iter_step])  # Compute stress
                    ϵv_data[iter_step, iter_cell, :] += 1/getnquadpoints(cellvalues) * state[q_point].xv[i, :] * Tv[i, iter_step]  # Compute viscous strain
                    ϵ_data[iter_step, iter_cell, :] += 1/getnquadpoints(cellvalues) * ϵ  # Compute strain
                end
            end
        end
    end    

    return σ_data, ϵ_data, ϵv_data
end

# Compute the total force for all time steps
function compute_force(n_steps::Int, dh::DofHandler, ch_nzBc, σ_data)
    F_sum = zeros(n_steps, 3)  # Initialize total force
    n_basefuncs = getnbasefunctions(cellvalues)  # Number of basis functions per element

    for iter_step = 1:n_steps
        iter_cell = 0
        F_all = zeros(ndofs(dh))  # Initialize force vector
        @inbounds for cell in CellIterator(dh)
            iter_cell += 1
            reinit!(cellvalues, cell)  # Reinitialize cell values for the current cell
            eldofs= celldofs(cell)  # Get degrees of freedom for the current cell

            Fe_all = zeros(length(eldofs))  # Initialize element force vector
            for q_point in 1:getnquadpoints(cellvalues)  # Loop over quadrature points
                dΩ = getdetJdV(cellvalues, q_point)  # Compute the Jacobian determinant
                for i in 1:n_basefuncs  # Loop over basis functions
                    B = tovoigt(shape_symmetric_gradient(cellvalues, q_point, i))  # Compute strain-displacement matrix
                    Fe_all[i] += transpose(B) * σ_data[iter_step, iter_cell, :] * dΩ  # Compute element force
                end
            end
            F_all[eldofs] += Fe_all  # Assemble element force into global force vector
        end

        F_all[Ferrite.free_dofs(ch_nzBc)] .= 0  # Apply boundary conditions
        F_sum[iter_step, :] = [sum(F_all[1:3:end]), sum(F_all[2:3:end]), sum(F_all[3:3:end])]  # Compute total force
    end

    return F_sum
end

# Assemble traction forces
function assemble_external_forces(dh, facevalues, fx)
    f = zeros(ndofs(dh))  # Initialize force vector

    for (cellcount, cell) in enumerate(CellIterator(dh))
        eldofs = celldofs(cell)  # Get degrees of freedom for the current cell

        for face in 1:nfaces(cell)
            if (cellid(cell), face) ∈ getfaceset(grid, "Set-1")
                reinit!(facevalues, cell, face)  # Reinitialize face values for the current face
                for q_point in 1:getnquadpoints(facevalues)  # Loop over quadrature points
                    dΓ = getdetJdV(facevalues, q_point)  # Compute the Jacobian determinant
                    for i in 1:getnbasefunctions(facevalues)  # Loop over basis functions
                        δu = shape_value(facevalues, q_point, i)  # Compute shape function value
                        f[eldofs[i]] += δu ⋅ fx * dΓ  # Compute force
                    end
                end
            end
        end
    end
    return f
end

# Compute the potential energy of external forces
function compute_varphi_f(dh, facevalues, fx, Xn)
    φf = 0  # Initialize potential energy

    for (cellcount, cell) in enumerate(CellIterator(dh))
        eldofs = celldofs(cell)  # Get degrees of freedom for the current cell

        for face in 1:nfaces(cell)
            if (cellid(cell), face) ∈ getfaceset(grid, "Set-1")
                reinit!(facevalues, cell, face)  # Reinitialize face values for the current face

                for q_point in 1:getnquadpoints(facevalues)  # Loop over quadrature points
                    dΓ = getdetJdV(facevalues, q_point)  # Compute the Jacobian determinant
                    φf += function_value(facevalues, q_point, Xn[eldofs]) ⋅ fx * dΓ  # Compute potential energy
                end
            end
        end
    end
    return φf
end

# Calculate finite differences of a vector
function diff_vec(vec::Vector{Float64}, Δt::Float64)
    res = zeros(length(vec))  # Initialize result vector

    res[1] = 1/Δt * (vec[2] - vec[1])  # Forward differences
    res[end] = 1/Δt *(vec[end]-vec[end-1])  # Backward differences

    for i = 2:length(vec)-1
        res[i] = 1/(2*Δt) * (vec[i+1]-vec[i-1])  # Midpoint differences
    end
    return res
end

# Integrate a vector using the trapezoidal rule
function integrate_vec(vec::Vector{Float64}, Δt::Float64)
    res = sum(vec) * Δt  # Trapezoidal rule integration
    return res
end

###################################################################################################################################
n_modes = 4                 # number of modes
n_fix_point_iter = 5        # number of iterations for fixed point iteration
n_steps = 51                # number of time steps
Δt = 10.0/(n_steps-1)       # time step size
λ = 1000E6                  # Lamé's first parameter  
μ = 800E6                   # Shear modulus
η = 10000E6                 # Viscosity coefficient

material, grid, cellvalues, facevalues, dh, K, ch, ch_zero, ch_nzBc = setup(λ, μ, η)    # Initialize material and grid

nqp = getnquadpoints(cellvalues)           # Number of quadrature points
states = [[MaterialState(n_modes) for _ in 1:nqp] for _ in 1:getncells(grid)]   # Initialize material states for each cell and quadrature point

X = zeros(n_modes, ndofs(dh))   # Initialize mode shapes
T = ones(n_modes, n_steps)  #reasonable initial condition for the time modes
Tv = zeros(n_modes, n_steps)    # Initialize viscous time modes
u = zeros(n_steps, ndofs(dh))
u_last = zeros(n_steps, ndofs(dh))  #save last state

#Scalars as in paper
τ = zeros(n_modes)
τv = zeros(n_modes)
α = zeros(n_modes)
αv = zeros(n_modes)
βv = zeros(n_modes)
Ψ = zeros(n_modes)

# Non-homogeneous dirichlet boundary conditions in first mode
iter_modes = 1
X[1, :] = apply!(X[1, :], ch) 
for iter_step = 1:n_steps
    T[1, iter_step] = 0.01*(iter_step-1)/100
end
T[2, 1:end] .= 0.1
T[:, 1] .= 0.0
Tv[:, 1] .= 0.0

# Neumann forces
bt = zeros(n_steps)
ft = ones(n_steps)  #ft is constant in time
ft[1] = 0.0
fx = [2000.0, 10000.0, 0.0]

ftrac = assemble_external_forces(dh, facevalues, fx)

##################################
## Fixed point iterations to compute space and time modes
println("#################################### Iterations ################################################")
total_iterations = 0

for iter_modes = 1:n_modes
    println("Mode: ", iter_modes)

    for fix_point_iter = 1:n_fix_point_iter
        #compute space modes X, Xv
        for i = 1:iter_modes
            τ[i] = integrate_vec(T[iter_modes, :].*T[i, :], Δt) 
            τv[i] = integrate_vec(T[iter_modes, :].*Tv[i, :], Δt)
            α[i] = integrate_vec(Tv[iter_modes, :].*T[i, :], Δt)
            αv[i] = integrate_vec(Tv[iter_modes, :].*Tv[i, :], Δt)
            βv[i] = integrate_vec(Tv[iter_modes, :].*diff_vec(Tv[i, :], Δt), Δt)
        end
        τb = integrate_vec(T[iter_modes, :].*bt, Δt)
        τf = integrate_vec(T[iter_modes, :].*ft, Δt)
        
        for iter_newton = 1:2   #convergence in two steps
            global K, R = doassemble_KR(iter_modes, cellvalues, K, X, dh, τ, τv, α, αv, βv, states, material)
            R -= ftrac * τf     # add the Neumann BC part
            apply_zero!(K, R, ch_zero)
            ΔX = K\R
            global X[iter_modes, :] = X[iter_modes, :] - ΔX
        end
   
        #compute time modes T, Tv
        Ψ, Ψv, av, b, bv = compute_x_funcs(iter_modes, cellvalues, X, dh, states, material)
        φf = compute_varphi_f(dh, facevalues, fx, X[iter_modes, :])

        lhs = Ψ[iter_modes] - Ψv[iter_modes]*(av[iter_modes] + Δt * bv[iter_modes])^(-1) * Δt * b[iter_modes]
        rhs_factor = Ψv[iter_modes] * (av[iter_modes] + Δt * bv[iter_modes])^(-1) * av[iter_modes]

        for ts = 2:n_steps
            fixed_part = 0
            for i = 1:iter_modes-1
                fixed_part += Ψv[iter_modes]*(av[iter_modes] + Δt * bv[iter_modes])^(-1)*Δt*b[i]*T[i, ts] 
                fixed_part -= Ψv[iter_modes]*(av[iter_modes] + Δt * bv[iter_modes])^(-1)*(av[i]*(Tv[i, ts] - Tv[i, ts-1]) + Δt*bv[i]*T[i, ts])
                fixed_part -= (Ψ[i]*T[i, ts] - Ψv[i]*Tv[i, ts])
            end
            fixed_part += φf * ft[ts]
            global T[iter_modes, ts] = lhs \ (rhs_factor*Tv[iter_modes, ts-1] + fixed_part)      #T is fixed as first mode is boundary condition
        end
            
        #if T is zero everywhere, it is set arbitrarily such that X will be zero
        if sum(abs.(T[iter_modes, :])) == 0.0
            T[iter_modes, :] .= 1.0
        end

        #normalize time modes
        sum_T_mode = sum(T[iter_modes, :])
        T[iter_modes, :] = T[iter_modes, :] ./ sum_T_mode
    
        for ts = 2:n_steps  #go over all time steps and update Tv
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
        rel_error = norm(u_cur - u_last)/norm(u_last)

        println("_________________________________________________________________________")
        println("Iteration: ", fix_point_iter)
        println("Rel_error: ", rel_error)
        println("_________________________________________________________________________")

        global u_last = u_cur

        if fix_point_iter > 2 && rel_error < 1E-4       #Early stopping criterion
            break
        end
    end
end

# Build up full solution for the displacement field
for iter = 1:n_modes
    println(iter)
    @tullio u[j, i] += X[$iter, i] * T[$iter, j]
end

@tullio u1[j, i] := X[1, i] * T[1, j]
@tullio u2[j, i] := X[2, i] * T[2, j]
@tullio u3[j, i] := X[3, i] * T[3, j]
@tullio u4[j, i] := X[4, i] * T[4, j]
@tullio u5[j, i] := X[5, i] * T[5, j]

# compute stress and reaction forces
σ, ϵ, ϵv = compute_stress(n_modes, n_steps, cellvalues, X, T, Tv, states, material)
F = compute_force(n_steps, dh, ch_nzBc, σ)

println("Total iterations: ", total_iterations)

#and save them in a folder
folderName = "sims/Femur/"*string(η)*"_"*string(n_modes)*"_"*string(total_iterations)
mkdir(datadir(folderName, ))

pvd = paraview_collection(datadir(folderName, "STD_ve.pvd"))
for t in 1:n_steps
    vtk_grid(datadir(folderName, "STD_res-$t"), dh) do vtk
        vtk_point_data(vtk, dh, u[t, :], "Displacement")
        vtk_point_data(vtk, dh, u1[t, :], "1st mode")
        vtk_point_data(vtk, dh, u2[t, :], "2nd mode")
        vtk_point_data(vtk, dh, u3[t, :], "3rd mode")
        vtk_point_data(vtk, dh, u4[t, :], "4th mode")
        vtk_point_data(vtk, dh, u5[t, :], "5th mode")
        vtk_point_data(vtk, dh, X[1, :], "1st mode shape")
        vtk_point_data(vtk, dh, X[2, :], "2nd mode shape")
        vtk_cell_data(vtk, transpose(σ[t, :, :]), "Stress")
        vtk_cell_data(vtk, transpose(ϵ[t, :, :]), "Strain")
        vtk_cell_data(vtk, transpose(ϵv[t, :, :]), "Viscous Strain")
        vtk_save(vtk)
        pvd[t] = vtk
    end
end

save(datadir(folderName, "T_modes_STD.jld2"), "Time modes", T)
save(datadir(folderName, "Sigma_STD.jld2"), "Stress", σ)
save(datadir(folderName, "Disp_STD.jld2"), "Displacement", u)

df = DataFrame(F, :auto)
CSV.write(datadir(folderName, "Force_STD.csv"), df)