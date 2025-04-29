using DrWatson
@quickactivate "SpaceTimeDecomp"

using Ferrite, SparseArrays
using LinearAlgebra
using Printf
using TensorOperations
using DataFrames
using CSV

include(srcdir("inp_reader.jl"))

#Material
struct MaterialParameters
    λ::Float64
    μ::Float64
    η::Float64
    E::Matrix{Float64}
end

#Material State
mutable struct MaterialState
    temp_ϵᵛ::Vector{Float64}
    ϵᵛ::Vector{Float64}
    σ::Vector{Float64}
    ϵ::Vector{Float64}
end

function MaterialState()
    return MaterialState(
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    )
end

function update_state!(state::MaterialState)
    state.ϵᵛ = state.temp_ϵᵛ
end

#Sigma, Sigma_Tangent
function compute_sigma_vc(ϵ, material::MaterialParameters, state::MaterialState, S, Δt)
    η = material.η
    E = material.E

    ϵᵛ = inv(I + Δt/η * S * E) * (state.ϵᵛ + Δt/η * S * E * ϵ)   
    sigma = E * (ϵ - ϵᵛ)
    state.temp_ϵᵛ = ϵᵛ
    return sigma
end

function compute_sigma_tangent_vc(material::MaterialParameters, S, Δt)
    η = material.η
    E = material.E
    
    return E - Δt/η * E * inv(I + Δt/η * S * E) * S * E
end

function symmetrize_lower!(K)
    for i in 1:size(K,1)
        for j in i+1:size(K,1)
            K[i,j] = K[j,i]
        end
    end
end;

#Assembly
function doassemble(cellvalues::CellVectorValues{dim}, K::SparseMatrixCSC, u::Vector{Float64}, dh::DofHandler, ch_nzBc::ConstraintHandler, states, material, S, Δt) where {dim}
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
        ue = u[eldofs]
        Fe_all = zeros(size(F_all[eldofs])[1])  #weirdly written

        for q_point in 1:getnquadpoints(cellvalues)     #For each integration point
            dΩ = getdetJdV(cellvalues, q_point)
                
            ϵ = tovoigt(function_symmetric_gradient(cellvalues, q_point, ue))
            σ = compute_sigma_vc(ϵ, material, state[q_point], S, Δt)
            dσ = compute_sigma_tangent_vc(material, S, Δt)

            state[q_point].σ = σ
            state[q_point].ϵ = ϵ

            for i in 1:n_basefuncs                      #For each u
                N = shape_value(cellvalues, q_point, i)        
                B = tovoigt(shape_symmetric_gradient(cellvalues, q_point, i))
                re[i] += transpose(B) * σ * dΩ 
                Fe_all[i] += transpose(B) * σ * dΩ

                for j in 1:i
                    Bj = tovoigt(shape_symmetric_gradient(cellvalues, q_point, j))
                    Ke[i, j] += transpose(B) * dσ * Bj * dΩ
                end
            end
        end
        F_all[eldofs] += Fe_all
        symmetrize_lower!(Ke)
        assemble!(assembler, celldofs(cell), re, Ke)
    end

    F_all[Ferrite.free_dofs(ch_nzBc)] .= 0
    F_sum = [sum(F_all[1:3:end]), sum(F_all[2:3:end]), sum(F_all[3:3:end])]
    return K, r, F_sum
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
    material = MaterialParameters(λ, μ, η, E)

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
    push!(dh, :u, dim, ip)
    close!(dh); 
    
    #Sparsity Pattern
    K = create_sparsity_pattern(dh);
    
    #Constraints
    ch = ConstraintHandler(dh);
    ∂Ωr = union(getfaceset.((grid, ), ["right"])...);
    dbc_r = Ferrite.Dirichlet(:u, ∂Ωr, pres_displacement, [3]);
    ∂Ωl = union(getfaceset.((grid, ), ["left"])...);
    dbc_l = Ferrite.Dirichlet(:u, ∂Ωl, (x, t) -> [0.0 0.0 0.0], [1, 2, 3]);
    add!(ch, dbc_r);
    add!(ch, dbc_l);
    close!(ch);

    #Boundary with nonzero boundary condition -> moving, since ∫_Ω F dV = 0 -> only reaction force or the external force
    ch_nzBc = ConstraintHandler(dh);
    add!(ch_nzBc, dbc_r); 
    close!(ch_nzBc);

    return S, material, grid, cellvalues, dh, K, ch, ch_nzBc
end


function solve(grid, cellvalues, dh, K, ch, ch_nzBc, nsteps, Δt)
    #Create material states
    nqp = getnquadpoints(cellvalues)
    states = [[MaterialState() for _ in 1:nqp] for _ in 1:getncells(grid)]

    #Solution
    n_dofs = ndofs(dh)
    u = zeros(n_dofs)
    Δu = zeros(n_dofs)

    #Newton-Raphson loop
    NEWTON_TOL = 1E-10 #1N
    println("Starting Newton iterations: ")

    ϵ_data = zeros(nsteps, getncells(grid), 6)
    ϵᵛ_data = zeros(nsteps, getncells(grid), 6)
    σ_data = zeros(nsteps, getncells(grid), 6)

    F_data = zeros(nsteps, 3)
    u_data = zeros(nsteps, n_dofs)
    F = zeros(3)

    for n_step = 1:nsteps
        println("Timestep: $n_step")
        newton_iter = 0
        t = (n_step-1) * Δt;

        println(t)
        update!(ch, t);
        apply!(u, ch)
        
        while newton_iter < 2
            newton_iter += 1 
            K, r, F = doassemble(cellvalues, K, u, dh, ch_nzBc, states, material, S, Δt);
            norm_r = norm(r[Ferrite.free_dofs(ch)])    
        
            println("Iteration: $newton_iter \tresidual: $(@sprintf("%.8f", norm_r))")
            if norm_r < NEWTON_TOL
                break
            end
        
            apply_zero!(K, r, ch)
            Δu = Symmetric(K) \ r
            u -= Δu
        end

        u_data[n_step, :] = u
        F_data[n_step, :] = F

        for cell_states in states
            foreach(update_state!, cell_states)
        end

        for (el, cell_states) in enumerate(states)
            for state in cell_states    
                ϵ_data[n_step, el, :] += state.ϵ[:]
                ϵᵛ_data[n_step, el, :] += state.ϵᵛ[:]
                σ_data[n_step, el, :] += state.σ[:]
            end 
            ϵ_data[n_step, el, :] /= length(cell_states) 
            ϵᵛ_data[n_step, el, :] /= length(cell_states) 
            σ_data[n_step, el, :] /= length(cell_states)
        end
    end

    println("Finished")
    return σ_data, ϵ_data, ϵᵛ_data, F_data, u_data
end

function calculate_area(grid::Grid, dh::DofHandler, cellvalues::CellVectorValues)
    cellvalues = deepcopy(cellvalues)
    CellIter = CellIterator(dh)
    ncells = getncells(grid)
    nqp = getnquadpoints(cellvalues)

    states = zeros(ncells, nqp)

    area = 0
    area_per_element = zeros(ncells)

    #For all quadrature points 
    @inbounds for cell in CellIter
        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            area += dΩ
            states[getindex(cellid(cell)), q_point] = dΩ
        end        
    end
    
    for el in 1:ncells
        area_per_element[el] = sum(states[el, :])
    end

    return area, area_per_element
end

function pres_displacement(x, t)
    return 0.01*t
end

λ = 1000E6
μ = 800E6
η = 500E6 
n_steps = 101
Δt = 1.0/(n_steps-1)

S, material, grid, cellvalues, dh, K, ch, ch_nzBc = setup(λ, μ, η,)
σ, ϵ, ϵᵛ, F_res, u = solve(grid, cellvalues, dh, K, ch, ch_nzBc, n_steps, Δt)
Ω, area_per_element = calculate_area(grid, dh, cellvalues)

folderName = "sims/Spring2/"*string(η)*"_NR"
mkdir(datadir(folderName, ))

SaveDict = Dict(
    "n_steps" => n_steps,
    "Δt" => Δt,
    "λ" => λ,
    "μ" => μ,
    "η" => η
)
CSV.write(datadir(folderName, "Info.csv"), DataFrame(SaveDict))

pvd = paraview_collection(datadir(folderName, "NR.pvd"))
for t in 1:n_steps
    vtk_grid(datadir(folderName, "nr_res-$t"), dh) do vtk
        vtk_cell_data(vtk, transpose(area_per_element), "Area")
        vtk_point_data(vtk, dh, u[t, :], "Displacement")
        vtk_cell_data(vtk, transpose(ϵ[t, :, :]), "True Strain")
        vtk_cell_data(vtk, transpose(ϵᵛ[t, :, :]), "True Visous Strain")
        vtk_cell_data(vtk, transpose(σ[t, :, :]), "True Stress")
        vtk_point_data(vtk, ch)
        vtk_save(vtk)
        pvd[t] = vtk
    end
end

save(datadir(folderName, "Disp_NR.jld2"), "Displacement", u)
save(datadir(folderName, "Sigma_NR.jld2"), "Stress", σ)
df = DataFrame(F_res, :auto)
CSV.write(datadir(folderName, "Force_NR.csv"), df)
